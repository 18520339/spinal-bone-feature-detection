import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.detection.map import mean_average_precision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from abc import abstractmethod, ABCMeta # For define pure virtual functions
from tqdm.notebook import tqdm
from loader import concat_batch


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, model, train_loader, val_loader, num_epochs, optimizer, scheduler=None, best_model_path='best_model.pth'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_train_batches = len(self.train_loader)
        self.total_val_batches = len(self.val_loader)
        self.device = next(model.parameters()).device
        
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter('results')
        self.history = {'train_total_losses': [], 'val_total_losses': []}
        self.best_val_total_loss = torch.inf
        self.best_model_path = best_model_path
        
    @abstractmethod
    def train(self):
        raise NotImplementedError('Subclasses should implement this method')
    
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError('Subclasses should implement this method')
        
        
class FasterRCNNTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, num_epochs, optimizer, scheduler=None, best_model_path='best_model.pth'):
        super().__init__(model, train_loader, val_loader, num_epochs, optimizer, scheduler, best_model_path)

    def train(self):
        for epoch in range(self.num_epochs): # Training loop
            train_total_loss, train_cls_loss, train_box_loss, train_edge_loss, val_total_loss = 0, 0, 0, 0, 0
            self.model.train()

            # Iterate over training batches
            for batch_idx, (names, images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                batch_total_loss = sum(loss for loss in loss_dict.values())
                train_total_loss += batch_total_loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad() # Zero gradients
                batch_total_loss.backward()
                self.optimizer.step()
            train_total_loss = train_total_loss / self.total_train_batches

            with torch.no_grad(): # Validation loop
                for names, images, targets in self.val_loader:
                    images = images.to(self.device)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(images, targets)
                    batch_total_loss = sum(loss for loss in loss_dict.values())
                    val_total_loss += batch_total_loss.item()
            val_total_loss /= self.total_val_batches
            
            # Logging and saving
            self.history['train_total_losses'].append(train_total_loss)
            self.history['val_total_losses'].append(val_total_loss)
            self.writer.add_scalars('Loss/Total', {'train': train_total_loss, 'val': val_total_loss}, epoch)
            print(f'[EPOCH {epoch+1}/{self.num_epochs}] Train Loss: {train_total_loss:.4f} - Val Loss: {val_total_loss:.4f}')
            if val_total_loss < self.best_val_total_loss: # Save the best model based on validation loss
                self.best_val_total_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_model_path)
            if self.scheduler: self.scheduler.step()
                
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            metric = MeanAveragePrecision(class_metrics=True)
            mAP_by_step, mAP_preds, mAP_targets = [], [], []
            all_preds, all_targets = [], []

            for names, images, targets in self.val_loader:
                images, targets, boxes_by_images, labels_by_images, all_boxes, all_labels = concat_batch(images, targets, self.device)
                outputs = self.model(images)
                all_preds.extend(outputs[0]['labels'].cpu().numpy())
                all_targets.extend(all_labels.cpu().numpy())
                
                mAP_preds.append({'boxes': outputs[0]['boxes'], 'labels': outputs[0]['labels'], 'scores': outputs[0]['scores']})
                mAP_targets.append({'boxes': all_boxes, 'labels': all_labels})
                mAP_dict = metric([mAP_preds[-1]], [mAP_targets[-1]])
                # mAP_by_step.append({k: v for k, v in mAP_dict.items() if k in ['map', 'map_50', 'map_75']})
                mAP_by_step.append(mAP_dict)

        # metric.plot(mAP_by_step)
        return mAP_preds, mAP_targets, {k: v for k, v in mean_average_precision(mAP_preds, mAP_targets, class_metrics=True).items()}


class MultiTaskModelTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, num_epochs, optimizer, scheduler=None, best_model_path='best_model.pth'):
        super().__init__(model, train_loader, val_loader, num_epochs, optimizer, scheduler, best_model_path)
        self.history.update({
            'train_cls_losses': [], 'train_box_losses': [], 'train_edge_losses': [],
            'val_cls_losses': [], 'val_box_losses': [], 'val_edge_losses': [],
        })
        
    def compute_step(self, images, targets, epoch_total_loss, epoch_cls_loss, epoch_box_loss, epoch_edge_loss):
        images, targets, boxes_by_images, labels_by_images, all_boxes, all_labels = concat_batch(images, targets, self.device)
        batch_total_loss, cls_loss, box_loss, edge_loss, pred_boxes, pred_labels = self.model(images, boxes_by_images, all_boxes, all_labels)
        epoch_cls_loss += cls_loss.item()
        epoch_box_loss += box_loss.item()
        epoch_edge_loss += edge_loss.item()
        epoch_total_loss += batch_total_loss.item()
        # scores, pred_labels_idxs = F.softmax(pred_labels, dim=-1).max(dim=1) 
        return (
            batch_total_loss, epoch_total_loss, epoch_cls_loss, epoch_box_loss, epoch_edge_loss,
            # scores, pred_boxes, pred_labels_idxs, all_boxes, all_labels
        )
        
    def train(self): # Train a multi-task model for object detection and clustering
        for epoch in range(self.num_epochs): # Training loop
            # loop = tqdm(enumerate(train_loader), total=self.total_train_batches)
            train_total_loss, train_cls_loss, train_box_loss, train_edge_loss = 0, 0, 0, 0
            val_total_loss, val_cls_loss, val_box_loss, val_edge_loss = 0, 0, 0, 0
            self.model.train()

            # Iterate over training batches
            mAP_preds, mAP_targets = [], []
            for batch_idx, (names, images, targets) in enumerate(self.train_loader):
                (batch_total_loss, train_total_loss, train_cls_loss, train_box_loss, train_edge_loss,
                # scores, pred_boxes, pred_labels_idxs, all_boxes, all_labels
                ) = self.compute_step(images, targets, train_total_loss, train_cls_loss, train_box_loss, train_edge_loss)
                # mAP_preds.append({'boxes': pred_boxes, 'labels': pred_labels_idxs, 'scores': scores})
                # mAP_targets.append({'boxes': all_boxes, 'labels': all_labels})

                # Backward pass and optimization
                self.optimizer.zero_grad() # Zero gradients
                batch_total_loss.backward()
                self.optimizer.step()

                # Update the progress bar
                # loop.set_description(f'[EPOCH {epoch+1}/{num_epochs}] {batch_idx + 1}/{self.total_train_batches}')
                # loop.set_postfix(loss=batch_total_loss.item(), accuracy=(correct / total).item())
                # loop.update()

            train_total_loss /= self.total_train_batches
            train_cls_loss /= self.total_train_batches
            train_box_loss /= self.total_train_batches
            train_edge_loss /= self.total_train_batches
            # train_mAPs = mean_average_precision(mAP_preds, mAP_targets, class_metrics=True)

            mAP_preds, mAP_targets = [], []
            with torch.no_grad(): # Validation loop
                for _, images, targets in self.val_loader:
                    (batch_total_loss, val_total_loss, val_cls_loss, val_box_loss, val_edge_loss,
                    # scores, pred_boxes, pred_labels_idxs, all_boxes, all_labels
                    ) = self.compute_step(images, targets, val_total_loss, val_cls_loss, val_box_loss, val_edge_loss)
                    # mAP_preds.append({'boxes': pred_boxes, 'scores': scores, 'labels': pred_labels_idxs})
                    # mAP_targets.append({'boxes': all_boxes, 'labels': all_labels})

            val_total_loss /= self.total_val_batches
            val_cls_loss /= self.total_val_batches
            val_box_loss /= self.total_val_batches
            val_edge_loss /= self.total_val_batches
            val_mAPs = {} # mean_average_precision(mAP_preds, mAP_targets, class_metrics=True)
            
            # Logging and saving
            self.history['train_total_losses'].append(train_total_loss)
            self.history['train_cls_losses'].append(train_cls_loss)
            self.history['train_box_losses'].append(train_box_loss)
            self.history['train_edge_losses'].append(train_edge_loss)
            self.history['val_cls_losses'].append(val_cls_loss)
            self.history['val_box_losses'].append(val_box_loss)
            self.history['val_edge_losses'].append(val_edge_loss)
            self.history['val_total_losses'].append(val_total_loss)

            self.writer.add_scalars('Loss/Total', {'train': train_total_loss, 'val': val_total_loss}, epoch)
            self.writer.add_scalars('Loss/CLS', {'train': train_cls_loss, 'val': val_cls_loss}, epoch)
            self.writer.add_scalars('Loss/BOX', {'train': train_box_loss, 'val': val_box_loss}, epoch)
            self.writer.add_scalars('Loss/EDGE', {'train': train_edge_loss, 'val': val_edge_loss}, epoch)
            print( # Print all the results
                f'[EPOCH {epoch + 1}/{self.num_epochs}] '
                f'Train Loss: {train_total_loss:.4f} (CLS: {train_cls_loss:.4f}, BOX: {train_box_loss:.4f}, EDGE: {train_edge_loss:.4f})'
                # f' - Train mAP@0.5: {train_mAPs['map_50']:.4f}'
                f' - Val Loss: {val_total_loss:.4f} (CLS: {val_cls_loss:.4f}, BOX: {val_box_loss:.4f}, EDGE: {val_edge_loss:.4f})'
                # f' - Val mAP@0.5: {val_mAPs['map_50']:.4f}'
                # f' - LR: {self.scheduler.get_last_lr()[0]:.6f}'
            )
            if val_total_loss < self.best_val_total_loss: # Save the best model based on validation loss
                self.best_val_total_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_model_path)
            if self.scheduler: self.scheduler.step()
            
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            metric = MeanAveragePrecision(class_metrics=True)
            mAP_by_step, mAP_preds, mAP_targets = [], [], []

            for names, images, targets in self.val_loader:
                images, targets, boxes_by_images, labels_by_images, all_boxes, all_labels = concat_batch(images, targets, self.device)
                predictions = self.model(images, boxes_by_images, all_boxes) # Get predictions for all images in batch
                start_idx = 0
                
                for i, boxes in enumerate(boxes_by_images): # Process each image separately
                    end_idx = start_idx + len(boxes)
                    
                    # Ground truth for this image
                    gt_boxes = all_boxes[start_idx:end_idx]
                    gt_labels = all_labels[start_idx:end_idx]
                    
                    mAP_preds.append(predictions[i])
                    mAP_targets.append({'boxes': gt_boxes, 'labels': gt_labels})
                    mAP_dict = metric([mAP_preds[-1]], [mAP_targets[-1]]) # Compute mAP for this step
                    # mAP_by_step.append({k: v for k, v in mAP_dict.items() if k in ['map', 'map_50', 'map_75']})
                    mAP_by_step.append(mAP_dict)
                    start_idx = end_idx
                
        # metric.plot(mAP_by_step)
        return mAP_preds, mAP_targets, {k: v for k, v in mean_average_precision(mAP_preds, mAP_targets, class_metrics=True).items()}