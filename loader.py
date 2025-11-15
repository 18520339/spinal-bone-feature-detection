import yaml
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class ScoliosisDataset(Dataset):
    def __init__(self, split='train', config_path='config.yaml', splits_path='splits.yaml'):
        # Initialize the dataset for a given split (train/val)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.split = split
        self.processed_images_dir = Path(self.config['data']['processed_images_dir'])
        self.processed_annotations_dir = Path(self.config['data']['processed_annotations_dir'])
        with open(splits_path, 'r') as f:
            splits = yaml.safe_load(f)

        self.image_names = splits[split]  # List of image names for the split
        self.means = np.array(splits['mean'], dtype=np.float32)  # Shape: (2,)
        self.stds = np.array(splits['std'], dtype=np.float32)    # Shape: (2,)
        self.transforms = self.get_transforms()


    def get_transforms(self):
        speckle_noise_std = self.config['augmentation']['speckle_noise_std']
        elastic_alpha = self.config['augmentation']['elastic_alpha']
        elastic_sigma = self.config['augmentation']['elastic_sigma']
        contrast_range = self.config['augmentation']['contrast_range']
        brightness_range = self.config['augmentation']['brightness_range']

        if self.split == 'train': # Apply augmentation if training
            return A.Compose([
                A.GaussNoise(std_range=(speckle_noise_std, speckle_noise_std), mean_range=(0.0, 0.0), per_channel=True, p=0.5),
                A.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, approximate=False, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=brightness_range, contrast_limit=contrast_range, p=0.5),
                A.Rotate(limit=10, p=0.5), # Small angles to avoid disrupting anatomical structure
                A.Normalize(mean=self.means, std=self.stds, max_pixel_value=1.0, p=1.0), # Ensure pixel values stay in [0, 1]
                ToTensorV2(p=1.0), # Convert to PyTorch tensor
            ], bbox_params=A.BboxParams(
                format='pascal_voc', # [x_min, y_min, x_max, y_max]
                label_fields=['labels'],  # Pass additional labels
                min_area=1,  # Drop boxes smaller than 1 pixels after augmentation
                min_visibility=0.3  # Discard boxes with less than 30% visibility after augmentation
            ))
        return A.Compose([
            A.Normalize(mean=self.means, std=self.stds, max_pixel_value=1.0, p=1.0), # Only normalization for validation
            ToTensorV2(p=1.0) # Convert to PyTorch tensor
        ])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = np.load(self.processed_images_dir / f'{image_name}.npy') # Shape: (H, W, 2)

        # Load annotations
        with open(self.processed_annotations_dir / f'{image_name}.txt', 'r') as f:
            annotations = [line.strip().split() for line in f.readlines()]
        annotations = np.array(annotations, dtype=np.float32) # Shape: (num_boxes, 5)

        # Extract boxes and labels
        boxes = annotations[:, 1:5] # x_min, y_min, x_max, y_max
        labels = annotations[:, 0].astype(np.int64) # 0: Thoracic, 1: Lumbar

        # Apply transformations
        transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
        image = transformed['image']
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)  # Shape: (num_boxes, 4)
        labels = torch.tensor(transformed['labels'], dtype=torch.int64)   # Shape: (num_boxes,)
        return image_name, image, {'boxes': boxes, 'labels': labels}
      
    
def collate_fn(batch):
    names, images, targets = zip(*batch)
    images = torch.stack(images, dim=0)  # Shape: (batch_size, 2, H, W)
    return list(names), images, list(targets)  # Targets as a list of dictionaries


def get_loader(split='train', config_path='config.yaml', splits_path='splits1.yaml', batch_size=32):
    # Create a data loader for a specific split (train/val)
    dataset = ScoliosisDataset(split=split, config_path=config_path, splits_path=splits_path)
    print(f'[{splits_path}] {split.capitalize()} dataset: {len(dataset)} samples')
    return DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True if split == 'train' else False, num_workers=2,
        pin_memory=True, collate_fn=collate_fn
    )
    
    
def concat_batch(images, targets, device): # Concat for batch processing
    images = images.to(device)
    boxes_by_images, labels_by_images, all_labels = [], [], []
    for target in targets:
        boxes_by_images.append(target['boxes'].to(device))
        labels_by_images.append(target['labels'].to(device))
        
    all_boxes = torch.cat(boxes_by_images, dim=0)
    all_labels = torch.cat(labels_by_images, dim=0)
    return images, targets, boxes_by_images, labels_by_images, all_boxes, all_labels
    
    
if __name__ == '__main__':
    train_loader = get_loader('train', splits_path='splits1.yaml', batch_size=32)
    val_loader = get_loader('val', splits_path='splits1.yaml', batch_size=32)
    for names, images, targets in train_loader:
        print(f'Batch images shape: {images.shape}')
        for names, target in zip(names, targets):
            print(f"{names}: boxes={target['boxes'].shape}, labels={target['labels'].shape}")
        break