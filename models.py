import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.ops import RoIAlign, nms
from collections import OrderedDict


def custom_resnet18_backbone():
    backbone = resnet18()
    original_conv1 = backbone.conv1
    backbone = nn.Sequential(*list(backbone.children())[:-2]) # Remove the classification head
    backbone[0] = nn.Conv2d( # Modify the first conv layer to accept 2 channels
        in_channels=2,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )
    backbone.out_channels = 512 # Define number of output channels (ResNet-18's final layer before avgpool has 512)
    return backbone


class EdgeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(1,)):
        super(EdgeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        layer_list = OrderedDict()
        self.device = device

        for l in range(len(num_features_list)):
            layer_list['conv%d' % l] = nn.Conv2d(
                in_channels = num_features_list[l - 1] if l > 0 else in_features,
                out_channels = num_features_list[l],
                kernel_size = 1, bias = False
            )
            layer_list['norm%d' % l] = nn.BatchNorm2d(num_features=num_features_list[l])
            layer_list['relu%d' % l] = nn.LeakyReLU()

        # Add final similarity kernel
        layer_list['conv_out'] = nn.Conv2d(in_channels=num_features_list[-1], out_channels=1, kernel_size=1)
        self.sim_network = nn.Sequential(layer_list).to(device)


    def forward(self, node_feat):
        node_feat = node_feat.unsqueeze(dim=0) # (1, bs, dim)
        num_tasks = node_feat.size(0) # 1
        num_data = node_feat.size(1) # bs

        x_i = node_feat.unsqueeze (2) # (1, bs, 1, dim)
        x_j = torch.transpose(x_i, 1, 2) # (1, 1, bs, dim)
        x_ij = torch.abs(x_i - x_j) # (1, bs, bs, dim)
        x_ij = torch.transpose(x_ij, 1, 3) # (1, dim, bs, bs)

        # Compute similarity / dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = (torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0).to(self.device)) # (bs, bs)

        # Normalize affinity matrix
        force_edge_feat = (torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)) # (1, bs, bs)
        edge_feat = sim_val + force_edge_feat # (bs, bs)
        edge_feat = edge_feat + 1e-6 # Add small value to avoid nan
        edge_feat = edge_feat / torch.sum(edge_feat, dim =1).unsqueeze (1) # Normalize
        return edge_feat, sim_val # (bs, bs), (bs, bs)


class NodeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio =(1,)):
        super(NodeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        layer_list = OrderedDict()
        self.device = device

        for l in range(len(num_features_list)):
            layer_list['conv%d' % l] = nn.Conv2d(
                in_channels = num_features_list[l - 1] if l > 0 else in_features * 2,
                out_channels = num_features_list[l],
                kernel_size = 1,
                bias = False,
            )
            layer_list['norm%d' % l] = nn.BatchNorm2d(num_features=num_features_list[l])

            if l < len(num_features_list) - 1: layer_list['relu%d' % l] = nn.LeakyReLU()
        self.network = nn.Sequential(layer_list).to(device)


    def forward(self, node_feat, edge_feat):
        # node_feat: (bs, dim), edge_feat: (bs, bs)
        node_feat = node_feat.unsqueeze(dim=0) # (1, bs, dim)
        num_tasks = node_feat.size(0) # 1
        num_data = node_feat.size(1) # bs

        # Get eye matrix(batch_size x node_size x node_size) only use inter dist.
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device) # (1, bs, bs)

        # Set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1) # (bs, bs)

        # Compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat) # (bs, dim)
        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2) # (1, 2 * dim, bs)

        # Non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2) # (1, bs, dim)
        return node_feat.squeeze(-1).squeeze(0) # (bs, dim)
    
    
class GNN(nn.Module):
    def __init__(self, in_features, edge_features, out_features, device, ratio=(1,)):
        super(GNN, self).__init__()
        self.edge_net = EdgeNet(in_features=in_features, num_features=edge_features, device=device, ratio=ratio)
        self.node_net = NodeNet(in_features=in_features, num_features=out_features, device=device, ratio=ratio) # Set edge to node
        self.mask_val = -1 # mask value for no-gradient edges

    def label2edge(self, targets): # Convert node labels to affinity mask for backprop
        num_sample = targets.size()[1]
        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float()
        target_edge_mask = (torch.eq(label_i, self.mask_val) + torch.eq(label_j, self.mask_val)).type(torch.bool)
        source_edge_mask = ~target_edge_mask
        edge *= source_edge_mask.float()
        return edge[0], source_edge_mask

    def forward(self, node_feat):
        edge_feat, edge_sim = self.edge_net(node_feat) # Compute normalized and not normalized affinity matrix
        logits_gnn = self.node_net(node_feat, edge_feat) # Get edge feature and class logits
        return logits_gnn, edge_sim
    

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes=2, criterion_box=None, max_detections=40, score_threshold=0.05, nms_threshold=0.5, device=None):
        super(MultiTaskModel, self).__init__()
        self.backbone = custom_resnet18_backbone()
        self.backbone2roi_proj = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        self.roi_align = RoIAlign(output_size=(5, 5), spatial_scale=1.0 / 32.0, sampling_ratio=-1)
        self.gnn = GNN(in_features=64 * 5 * 5, edge_features=512, out_features=num_classes, device=device, ratio=(1,))
        self.box_regressor = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.criterion_box = criterion_box # DIoU or CIoU loss
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_edge = nn.BCELoss()
        self.max_detections = max_detections
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.to(device)
        

    def forward(self, images, boxes_by_images, all_boxes, all_labels=None):
        # Create batch indices for each box and combine batch_indices and all_boxes into a single rois tensor
        batch_indices = torch.cat([
            torch.full((len(b),), i, dtype=torch.int64)
            for i, b in enumerate(boxes_by_images)
        ], dim=0).to(images.device) # Shape: (total_boxes,)
        rois = torch.cat([batch_indices.unsqueeze(1).float(), all_boxes], dim=1) # Shape: (total_boxes, 5)

        # Apply RoIAlign and flatten the pooled features
        feature_maps = self.backbone(images)                                 # Shape: (batch_size, 512, H/32, W/32)
        feature_maps = self.backbone2roi_proj(feature_maps)                  # Shape: (batch_size, 64, H/32, W/32)
        pooled_features = self.roi_align(feature_maps, rois)                 # Shape: (total_boxes, 64, 5, 5)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Shape: (total_boxes, 64 * 5 * 5)
        pred_boxes = self.box_regressor(pooled_features)
        pred_labels, edge_sim = self.gnn(pooled_features)
        
        if self.training and all_labels is not None: # Calculate Multi-task loss
            box_loss = self.criterion_box(pred_boxes, all_boxes, reduction='mean')  # Box loss
            cls_loss = self.criterion_cls(pred_labels, all_labels) # Cls loss
            edge_gt, edge_mask = self.gnn.label2edge(all_labels.unsqueeze(dim=0))
            edge_loss = self.criterion_edge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))
            total_loss = cls_loss + box_loss + edge_loss
            return total_loss, cls_loss, box_loss, edge_loss, pred_boxes, pred_labels
        return self._post_process_predictions(pred_boxes, pred_labels, boxes_by_images) # Post-process predictions during inference
    
    
    def _post_process_predictions(self, pred_boxes, pred_labels, boxes_by_images):
        # Split concatenated predictions back into per-image format and apply NMS
        pred_boxes_list, pred_labels_list, pred_scores_list = [], [], []
        start_idx = 0
        
        for boxes_by_image in boxes_by_images:
            end_idx = start_idx + len(boxes_by_image)
            
            # Get predictions for this image
            boxes = pred_boxes[start_idx:end_idx]
            logits = pred_labels[start_idx:end_idx]
            scores, labels = F.softmax(logits, dim=-1).max(dim=1)
            
            # Apply score thresholding
            score_mask = scores >= self.score_threshold
            boxes, scores, labels = boxes[score_mask], scores[score_mask], labels[score_mask]
            
            # Apply NMS per class
            keep_indices = []
            for class_id in labels.unique(): 
                class_mask = labels == class_id
                class_boxes, class_scores = boxes[class_mask], scores[class_mask]
                
                if len(class_boxes) > 0:
                    keep = nms(class_boxes, class_scores, self.nms_threshold)
                    keep_indices.extend(torch.where(class_mask)[0][keep].tolist())
            
            # Limit to max detections per image
            if len(keep_indices) > self.max_detections: 
                keep_scores = scores[keep_indices]
                top_k = keep_scores.topk(self.max_detections).indices
                keep_indices = [keep_indices[i] for i in top_k.tolist()]
            
            pred_boxes_list.append(boxes[keep_indices])
            pred_labels_list.append(labels[keep_indices])
            pred_scores_list.append(scores[keep_indices])
            start_idx = end_idx
        return pred_boxes_list, pred_labels_list, pred_scores_list
    
    
class CustomFasterRCNN(FasterRCNN):
    def forward(self, images, targets=None):
        if self.training and targets is None: raise ValueError('In training mode, targets should be passed')
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        image_shapes = images.image_sizes
        
        proposals, rpn_losses = self.rpn(images, features, targets)
        if self.training: proposals = [target['boxes'] for target in targets] # List of tensors [num_gt_per_image, 4]
        detections, detector_losses = self.roi_heads(features, proposals, image_shapes, targets)
        detections = self.transform.postprocess(detections, image_shapes, original_image_sizes)
        
        losses = {}
        losses.update(rpn_losses)
        losses.update(detector_losses)
        
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn('RCNN always returns a (Losses, Detections) tuple in scripting')
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)