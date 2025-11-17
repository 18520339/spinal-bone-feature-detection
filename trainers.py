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
    def __init__(
        self, model, train_loader, val_loader, num_epochs=None, optimizer=None, 
        scheduler=None, patience=40, min_delta=0.0, best_model_path='best_model.pth'):
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
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_without_improvement = 0
        
    @abstractmethod
    def train(self):
        raise NotImplementedError('Subclasses should implement this method')
    
    @abstractmethod
    def get_predictions(self, images, boxes_by_images, all_boxes, all_labels):
        # Get predictions from the model. Returns (pred_boxes, pred_labels, pred_scores)
        raise NotImplementedError('Subclasses should implement this method')
    
    def evaluate(self):  # Common evaluation logic for all trainers
        self.model.eval()
        with torch.no_grad():
            metric = MeanAveragePrecision(class_metrics=True)
            mAP_preds, mAP_targets = [], []

            for names, images, targets in self.val_loader:
                images, targets, boxes_by_images, labels_by_images, all_boxes, all_labels = concat_batch(images, targets, self.device)
                pred_boxes, pred_labels, pred_scores = self.get_predictions(images, boxes_by_images, all_boxes, all_labels)
                start_idx = 0
                
                for i, boxes_by_image in enumerate(boxes_by_images): # Convert targets back to per-image format
                    end_idx = start_idx + len(boxes_by_image)
                    mAP_preds.append({'boxes': pred_boxes[i], 'labels': pred_labels[i], 'scores': pred_scores[i]})
                    mAP_targets.append({'boxes': all_boxes[start_idx:end_idx], 'labels': all_labels[start_idx:end_idx]})
                    metric.update([mAP_preds[-1]], [mAP_targets[-1]]) # Compute mAP for this step
                    start_idx = end_idx

        overall = metric.compute() # Aggregate over all batches
        return mAP_preds, mAP_targets, {k: v for k, v in overall.items()}
    
        
class FasterRCNNTrainer(BaseTrainer):
    def __init__(
        self, model, train_loader, val_loader, num_epochs=None, optimizer=None, 
        scheduler=None, patience=40, min_delta=0.0, best_model_path='best_faster_rcnn.pth'):
        super().__init__(model, train_loader, val_loader, num_epochs, optimizer, scheduler, patience, min_delta, best_model_path)


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
            
            if val_total_loss < self.best_val_total_loss - self.min_delta: # Save the best model based on validation loss
                print(f'[EPOCH {epoch + 1}/{self.num_epochs}] Train Loss: {train_total_loss:.4f} - Val Loss: {val_total_loss:.4f}')
                self.epochs_without_improvement = 0
                self.best_val_total_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_model_path)
            else: self.epochs_without_improvement += 1
                
            # Early stopping check
            if self.patience and self.epochs_without_improvement >= self.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs. No improvement for {self.patience} epochs.')
                break
            if self.scheduler: self.scheduler.step()
            
    
    def get_predictions(self, images, boxes_by_images, all_boxes, all_labels): # Get predictions from Faster R-CNN model
        outputs = self.model(images)
        pred_boxes = [output['boxes'] for output in outputs]
        pred_labels = [output['labels'] for output in outputs]
        pred_scores = [output['scores'] for output in outputs]
        return pred_boxes, pred_labels, pred_scores
    

class MultiTaskModelTrainer(BaseTrainer):
    def __init__(
        self, model, train_loader, val_loader, num_epochs=None, optimizer=None, 
        scheduler=None, patience=40, min_delta=0.0, best_model_path='best_multitask_model.pth'):
        super().__init__(model, train_loader, val_loader, num_epochs, optimizer, scheduler, patience, min_delta, best_model_path)
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
            
            if val_total_loss < self.best_val_total_loss - self.min_delta: # Save the best model based on validation loss
                print( # Only print when a new best model is found
                    f'[EPOCH {epoch + 1}/{self.num_epochs}] '
                    f'Train Loss: {train_total_loss:.4f} (CLS: {train_cls_loss:.4f}, BOX: {train_box_loss:.4f}, EDGE: {train_edge_loss:.4f})'
                    # f' - Train mAP@0.5: {train_mAPs['map_50']:.4f}'
                    f' - Val Loss: {val_total_loss:.4f} (CLS: {val_cls_loss:.4f}, BOX: {val_box_loss:.4f}, EDGE: {val_edge_loss:.4f})'
                    # f' - Val mAP@0.5: {val_mAPs['map_50']:.4f}'
                    # f' - LR: {self.scheduler.get_last_lr()[0]:.6f}'
                )
                self.epochs_without_improvement = 0
                self.best_val_total_loss = val_total_loss
                torch.save(self.model.state_dict(), self.best_model_path)
            else: self.epochs_without_improvement += 1
                
            # Early stopping check
            if self.patience and self.epochs_without_improvement >= self.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs. No improvement for {self.patience} epochs.')
                break
            if self.scheduler: self.scheduler.step()
    
    
    def get_predictions(self, images, boxes_by_images, all_boxes, all_labels): # Get predictions from multi-task model
        return self.model(images, boxes_by_images, all_boxes, all_labels)