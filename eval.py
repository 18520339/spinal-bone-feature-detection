import torch
from torchvision.ops import box_iou
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix, roc_curve, auc
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from loader import ScoliosisDataset, get_loader, concat_batch
from trainers import FasterRCNNTrainer
plt.style.use('seaborn-v0_8')


def kfolds_evaluate(trainer_class, model, splits_paths, weights_paths, device):
    kfolds_preds, kfolds_targets, kfolds_results = {}, {}, {}
    is_faster_rcnn = issubclass(trainer_class, FasterRCNNTrainer)

    for fold_idx, (splits_path, weights_path) in enumerate(zip(splits_paths, weights_paths)):
        if is_faster_rcnn:
            temp_dataset = ScoliosisDataset(split='train', config_path='config.yaml', splits_path=splits_path, use_bg_class=is_faster_rcnn)
            dataset_mean, dataset_std = temp_dataset.means, temp_dataset.stds
            model.transform.image_mean = dataset_mean.tolist()
            model.transform.image_std = dataset_std.tolist()

        train_loader = get_loader('train', splits_path=splits_path, batch_size=32, use_bg_class=is_faster_rcnn)
        val_loader = get_loader('val', splits_path=splits_path, batch_size=32, use_bg_class=is_faster_rcnn)
        trainer = trainer_class(model, train_loader, val_loader)

        checkpoint = torch.load(weights_path)
        trainer.model.load_state_dict(checkpoint)
        mAP_preds, mAP_targets, results = trainer.evaluate()

        fold_name = f'Fold {fold_idx + 1}'
        kfolds_preds[fold_name], kfolds_targets[fold_name], kfolds_results[fold_name] = [], [], {}
        if is_faster_rcnn: # Mapping: 0 (Background) -> 2 (Background), 1 (Thoracic) -> 0 (Thoracic), 2 (Lumbar) -> 1 (Lumbar)
            mapping = torch.tensor([2, 0, 1], device=device)

            for pred_dict in mAP_preds:
                new_pred_dict = pred_dict.copy()
                new_pred_dict['labels'] = mapping[pred_dict['labels']]
                kfolds_preds[fold_name].append(new_pred_dict)

            for target_dict in mAP_targets:
                new_target_dict = target_dict.copy()
                new_target_dict['labels'] = mapping[target_dict['labels']]
                kfolds_targets[fold_name].append(new_target_dict)
        else:
            kfolds_preds[fold_name] = mAP_preds
            kfolds_targets[fold_name] = mAP_targets

        for metric, value in results.items():
            if metric not in ['map', 'map_50', 'map_75', 'map_per_class']: continue
            if torch.is_tensor(value):
                if value.ndim == 0: kfolds_results[fold_name][metric] = value.item() # scalar -> normal value
                else: # vector/matrix -> split into columns
                    for i, v in enumerate(value.tolist()):
                        class_name = metric.split('_')[0] + ('_lumbar' if i == (len(value) - 1) else '_thoracic')
                        kfolds_results[fold_name][class_name] = v
            else: kfolds_results[fold_name][metric] = value

    # Calculate mean and std for each metric accross folds in kfolds_results
    df = pd.DataFrame(kfolds_results).T
    kfolds_results['Avg.'] = (df.mean().round(4).astype(str) + ' ± ' + df.std().round(4).astype(str)).to_dict()
    kfolds_results['Best Fold'] = df.idxmax().to_dict()
    kfolds_results['Worst Fold'] = df.idxmin().to_dict()
    return kfolds_preds, kfolds_targets, kfolds_results


def compute_pr_curve(preds, targets, iou_threshold=0.5):
    all_scores, all_labels = [], [] # 1 for TP, 0 for FP (for detection PR curve)
    classification_gt_labels_for_report = [] 
    classification_pred_labels_for_report = []

    for pred, target in zip(preds, targets): # For each image
        pred_boxes = pred['boxes']   # [num_preds, 4]
        pred_labels = pred['labels'] # [num_preds]
        pred_scores = pred['scores'] # [num_preds]
        gt_boxes = target['boxes']   # [num_gts, 4]
        gt_labels = target['labels'] # [num_gts]

        # Filter ground truths to only foreground classes (0 or 1)
        fg_gt_indices = (gt_labels < 2).nonzero(as_tuple=True)[0]
        current_fg_gt_boxes = gt_boxes[fg_gt_indices]
        current_fg_gt_labels = gt_labels[fg_gt_indices]
        
        # Track which foreground GTs have been matched by a prediction.
        matched_gt_indices_detection = set() # This is for the detection PR curve (to prevent multiple predictions matching same GT)
        
        # Track for the classification report: stores index of pred that matched a GT, or -1 if unmatched
        gt_to_pred_match_map = [-1] * len(current_fg_gt_boxes) # Size is number of foreground GTs in the current image
        sorted_pred_indices = torch.argsort(pred_scores, descending=True) # Sort predictions by score for the PR curve calculation
        
        # Stage 1: Process Predictions for Detection PR Curve and prepare for Classification Report
        for i_pred in sorted_pred_indices:
            pred_box = pred_boxes[i_pred]
            pred_label = pred_labels[i_pred]
            pred_score = pred_scores[i_pred]

            if pred_label.item() >= 2: # Predictions with background label (2) are considered False Positives for foreground detection
                all_labels.append(0) # False Positive
                all_scores.append(pred_score.cpu().numpy())
                continue 

            if len(current_fg_gt_boxes) == 0: # Handle foreground predictions
                all_labels.append(0) # This foreground prediction is a False Positive (no foreground GTs to match)
                all_scores.append(pred_score.cpu().numpy())
                continue

            ious_with_fg_gts = box_iou(pred_box.unsqueeze(0), current_fg_gt_boxes)[0]
            if ious_with_fg_gts.numel() == 0:
                all_labels.append(0) # False Positive
                all_scores.append(pred_score.cpu().numpy())
                continue

            max_iou, fg_gt_idx_in_current_fg_gts = torch.max(ious_with_fg_gts, dim=0)
            if max_iou >= iou_threshold and fg_gt_idx_in_current_fg_gts.item() not in matched_gt_indices_detection:
                all_labels.append(1) # True Positive for detection
                all_scores.append(pred_score.cpu().numpy())
                matched_gt_indices_detection.add(fg_gt_idx_in_current_fg_gts.item()) # Mark GT as detected for PR curve
                gt_to_pred_match_map[fg_gt_idx_in_current_fg_gts.item()] = i_pred.item() # Store index of matching prediction
            else:
                all_labels.append(0) # False Positive for detection
                all_scores.append(pred_score.cpu().numpy())

        # Stage 2: Populate Classification Report Labels based on all Foreground GTs ---
        for gt_idx, fg_gt_label in enumerate(current_fg_gt_labels):
            classification_gt_labels_for_report.append(fg_gt_label.item()) # All foreground GTs

            # Check if this GT was successfully matched by a prediction
            pred_matched_idx = gt_to_pred_match_map[gt_idx]
            if pred_matched_idx != -1: # GT was matched by a prediction
                predicted_label_for_gt = pred_labels[pred_matched_idx].item()
                classification_pred_labels_for_report.append(predicted_label_for_gt)
            else: # This GT was a False Negative (not detected/matched by a prediction)
                # Assign it to a 'missed' category, using label 2 (background) as a proxy
                classification_pred_labels_for_report.append(2) 

    # Handle cases where all_labels or all_scores might be empty for PR curve
    if not all_labels: precision, recall = np.array([1.0]), np.array([0.0])
    else:
        all_labels_np = np.array(all_labels)
        all_scores_np = np.array(all_scores)
        if np.sum(all_labels_np) == 0: precision, recall = np.array([0.0]), np.array([0.0])
        else: precision, recall, _ = precision_recall_curve(all_labels_np, all_scores_np)

    # Return the collected labels for the classification report
    if not classification_gt_labels_for_report: return precision, recall, [], [] # If no foreground GTs were present at all
    return precision, recall, classification_gt_labels_for_report, classification_pred_labels_for_report


def plot_cls_metrics(preds, targets):
    label_names_full = ['Thoracic', 'Lumbar', 'Missed'] 
    precision, recall, all_matched_gt_labels, all_matched_pred_labels = compute_pr_curve(preds, targets)

    # Only include labels present in the ground truth for the report, but also include '2' if it's in predictions for FNs.
    unique_gt_labels = np.unique(all_matched_gt_labels).tolist()
    unique_pred_labels = np.unique(all_matched_pred_labels).tolist()
    
    # Determine the labels to include in the report for classification_report
    report_labels = sorted(list(set(unique_gt_labels + unique_pred_labels)))
    current_target_names = [label_names_full[i] for i in report_labels]
    print(classification_report(
        all_matched_gt_labels, all_matched_pred_labels, labels=report_labels, 
        target_names=current_target_names, digits=4, zero_division=0.0
    ))

    # Confusion matrix labels need to cover all possible predicted classes (0, 1, 2) that might appear
    cm_labels_extended = sorted(list(set([0, 1, 2]).union(set(all_matched_gt_labels)).union(set(all_matched_pred_labels))))
    cm_display_names = [label_names_full[i] for i in cm_labels_extended]
    cf_matrix = confusion_matrix(all_matched_gt_labels, all_matched_pred_labels, labels=cm_labels_extended)
    annot_labels = np.asarray([f'{count:,}' for count in cf_matrix.flatten()]).reshape(cf_matrix.shape)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    ax_cm = plt.subplot(1, 2, 2)
    sns.heatmap(
        cf_matrix, fmt='', annot=annot_labels, cmap='YlGnBu', square=True, 
        annot_kws={'size': 12}, xticklabels=cm_display_names, yticklabels=cm_display_names, ax=ax_cm
    )
    plt.title('Confusion Matrix') # Add title to CM

    # plt.subplot(1, 3, 3)
    # if len(np.unique(all_matched_gt_labels)) < 2:
    #     print('ROC curve cannot be plotted: Not enough unique classes in ground truth labels (need at least 2).')
    #     plt.text(
    #         0.5, 0.5, 'ROC Not Plotted\n(Single Class GT)', 
    #         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12
    #     )
    # else:
    #     # ROC curve calculation is generally for binary classification.
    #     # If 'Missed' (label 2) is present in predicted labels, or if the GTs themselves are not strictly binary (0 and 1),
    #     # a standard ROC curve for labels 0 and 1 might be misleading or incorrect.
    #     if 2 in unique_pred_labels or len(unique_gt_labels) != 2:
    #         print(f'ROC curve not plotted for multi-class/imbalanced scenario (GT unique classes: {unique_gt_labels}, Pred unique classes: {unique_pred_labels}).')
    #         plt.text(
    #             0.5, 0.5, 'ROC Not Plotted\n(Multi-class/Missed preds)', 
    #             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10
    #         )
    #     else: # If only foreground classes 0 and 1 are involved in both GT and Pred, proceed with binary ROC
    #         fpr, tpr, _ = roc_curve(all_matched_gt_labels, all_matched_pred_labels)
    #         plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.3f})')
    #         plt.plot([0, 1], [0, 1], 'm--')
    #         plt.legend(loc='lower right', frameon=True, shadow=True, borderpad=0.5)
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title('Receiver Operating Characteristic')
            
    plt.tight_layout()
    plt.show()
    
    
def display_validation_inference(
    model, images_dir, processed_height, processed_width, 
    best_fold_idx, splits_paths, weights_paths, get_predictions_func, device):
    def get_scale(image):
        orig_width, orig_height = image.size
        scale_x = orig_width / processed_width
        scale_y = orig_height / processed_height
        return scale_x, scale_y

    val_loader = get_loader('val', splits_path=splits_paths[best_fold_idx], batch_size=32)
    checkpoint = torch.load(weights_paths[best_fold_idx])
    model.load_state_dict(checkpoint)
    model.eval()

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 15))
    axes = axes.flatten()          # Flatten the grid for easy indexing
    for ax in axes: ax.grid(False) # Turn off gridlines
    grid_idx, pred_prid_idx = 0, 0 # Counter for the grid positions

    for names, images, targets in val_loader:
        images, targets, boxes_by_images, labels_by_images, all_boxes, all_labels = concat_batch(images, targets, device)
        pred_lengths, pred_boxes, pred_labels, pred_scores = get_predictions_func(images, boxes_by_images, all_boxes, all_labels) 
        
        # Reshape to match the number of predicted boxes per image
        pred_boxes_by_images = torch.split(pred_boxes, pred_lengths)
        pred_labels_by_images = torch.split(pred_labels, pred_lengths)
        pred_scores_by_images = torch.split(pred_scores, pred_lengths)

        # Plot each ground truth box
        for name, boxes, labels in zip(names, boxes_by_images, labels_by_images):
            if grid_idx >= len(axes): break # Stop if all subplots are filled
            image = Image.open(images_dir / f'{name}.bmp')
            scale_x, scale_y = get_scale(image) # Scale the boxes back to the original image dimensions
            orig_width, orig_height = image.size

            # Display image
            axes[grid_idx].imshow(image, cmap='gray')
            axes[grid_idx].axis('off') # Turn off axis labels and ticks
            axes[grid_idx].set_title(f'{orig_height} x {orig_width}\n({len(boxes)} boxes detected)')
            grid_idx += 1  # Move to the next subplot position

            for box, label in zip(boxes, labels):
                if label >= 2: continue
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                color = 'red' if label == 0 else 'green'
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
                axes[grid_idx - 1].add_patch(rect) # Add the rectangle to the plot

        # Plot each predicted box (Similar processing as for ground truth boxes)
        for name, boxes, labels, scores in zip(names, pred_boxes_by_images, pred_labels_by_images, pred_scores_by_images):
            if pred_prid_idx >= len(axes): break # Stop if all subplots are filled
            image = Image.open(images_dir / f'{name}.bmp')
            scale_x, scale_y = get_scale(image) # Scale the boxes back to the original image dimensions
            pred_prid_idx += 1  # Move to the next subplot position

            for box, label, score in zip(boxes, labels, scores):
                if label >= 2: continue
                x1, y1, x2, y2 = box.cpu().detach().numpy() # Detach the tensor before converting to NumPy
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
                color = 'cyan' if label == 0 else 'gold'
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor='none')
                axes[pred_prid_idx - 1].add_patch(rect) # Add the rectangle to the plot
                axes[pred_prid_idx - 1].text(
                    x1, y1, f'{score.cpu().detach().numpy():.4f}',
                    fontsize=9, fontweight='bold',
                    color=color, # bbox=dict(facecolor='white', alpha=0.2)
                )
        if grid_idx >= len(axes): break # Stop if all subplots are filled

    for ax in axes[grid_idx:]: ax.axis('off') # Hide any unused subplots
    plt.tight_layout()
    plt.show()