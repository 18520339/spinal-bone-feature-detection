import cv2
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


class UltrasoundPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.image_size = tuple(self.config['preprocessing']['image_size']) # (height, width)
        self.despeckle_h = self.config['preprocessing']['despeckle_h']
        self.clahe_clip_limit = self.config['preprocessing']['clahe_clip_limit']
        self.clahe_tile_grid_size = tuple(self.config['preprocessing']['clahe_tile_grid_size'])
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)

        # Create necessary directories for processed data
        self.images_dir = Path(self.config['data']['images_dir'])
        self.annotations_dir = Path(self.config['data']['annotations_dir'])
        self.processed_images_dir = Path(self.config['data']['processed_images_dir'])
        self.processed_annotations_dir = Path(self.config['data']['processed_annotations_dir'])
        self.metadata_dir = Path(self.config['data']['metadata_dir'])

        # Create directories if they don't exist
        self.processed_images_dir.mkdir(parents=True, exist_ok=True)
        self.processed_annotations_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)


    def load_image(self, image_path): # Load an image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f'Failed to load image at {image_path}')
        return image


    def compute_edges(self, image): # Compute edge map using Sobel filter
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        return sobel


    def lee_filter(self, img, window_size=5): # Apply Lee filter to reduce speckle noise in ultrasound images
        # Compute local mean and variance
        mean = cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size))
        mean_sq = cv2.boxFilter(img.astype(np.float32) ** 2, -1, (window_size, window_size))
        variance = mean_sq - mean ** 2
        noise_variance = np.mean(variance) # Estimate noise variance
        
        # Lee filter: img = mean + (img - mean) * (local variance / (local variance + noise variance))
        img_float = img.astype(np.float32)
        weight = variance / (variance + noise_variance + 1e-6)  # Avoid division by zero
        filtered_img = mean + weight * (img_float - mean)
        return np.clip(filtered_img, 0, 255).astype(np.uint8)


    def preprocess(self, image): # Apply the full preprocessing pipeline
        # image = self.lee_filter(image, window_size=5) # Apply Lee filter to reduce speckle noise
        image = cv2.fastNlMeansDenoising( # Apply non-local means denoising to reduce speckle noise
            image, h=self.despeckle_h, 
            templateWindowSize=7, searchWindowSize=21
        )
        image = self.clahe.apply(image) # Apply CLAHE to enhance contrast
        edges = self.compute_edges(image)
        image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
        edges = cv2.resize(edges, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
        processed_image = np.stack([image, edges], axis=-1) # Stack image & edges as a 2-channel input. Shape: (H, W, 2)
        processed_image = processed_image.astype(np.float32) / 255.0 # Normalize to [0, 1]
        return processed_image


    def preprocess_and_save(self, image_path, annotation_path, output_image_path, output_annotation_path):
        # Preprocess an image, adjust annotations, and save both
        image = self.load_image(image_path)
        with open(annotation_path, 'r') as f: # Load annotations (format: class x_min y_min x_max y_max)
            annotations = [line.strip().split() for line in f.readlines()]
        annotations = np.array(annotations, dtype=np.float32)  # Shape: (num_boxes, 5)
        
        orig_h, orig_w = image.shape[:2] # Get original dimensions
        new_h, new_w = self.image_size
        processed_image = self.preprocess(image)
        
        if annotations.size > 0: # Adjust bounding box coordinates for resizing
            annotations[:, 1] *= (new_w / orig_w) # x_min
            annotations[:, 2] *= (new_h / orig_h) # y_min
            annotations[:, 3] *= (new_w / orig_w) # x_max
            annotations[:, 4] *= (new_h / orig_h) # y_max

        np.save(output_image_path, processed_image) # Save preprocessed image
        with open(output_annotation_path, 'w') as f:
            for annotation in annotations: # Save adjusted annotations
                f.write(f'{int(annotation[0])} {annotation[1]:.6f} {annotation[2]:.6f} {annotation[3]:.6f} {annotation[4]:.6f}\n')


    def preprocess_and_save_dataset(self): # Preprocess all images and annotations, and create train/val/test splits
        image_files = sorted([ # Get list of image files
            f for f in self.images_dir.glob('*') 
            if f.suffix in ['.jpg', 'jpeg', '.png']
        ])
        image_names = []

        for image_path in tqdm(image_files): # Process each image and its corresponding annotation file
            image_name = image_path.stem
            image_names.append(image_name)
            annotation_path = self.annotations_dir / f'{image_name}.txt'

            if not annotation_path.exists():
                raise FileNotFoundError(f'Annotation file {annotation_path} not found')
            
            self.preprocess_and_save(
                image_path = image_path,
                annotation_path = annotation_path,
                output_image_path = self.processed_images_dir / f'{image_name}.npy',
                output_annotation_path = self.processed_annotations_dir / f'{image_name}.txt'
            )
        
        # Create train/val splits and compute mean & std std for each channel across training images
        train_names, val_names = train_test_split(image_names, test_size=self.config['data']['val_split'], random_state=42)
        train_images = [np.load(self.processed_images_dir  / f'{image_name}.npy') for image_name in train_names] # List of (H, W, 2)
        train_images = np.stack(train_images, axis=0) # Shape: (N, H, W, 2)
        mean = np.mean(train_images, axis=(0, 1, 2))  # Shape: (2,)
        std = np.std(train_images, axis=(0, 1, 2))    # Shape: (2,)

        with open(self.metadata_dir / 'splits.yaml', 'w') as f: # Save splits to metadata
            yaml.dump({'train': train_names, 'val': val_names, 'mean': mean.tolist(), 'std': std.tolist()}, f)
        print(f'Dataset splits: Train={len(train_names)}, Val={len(val_names)}, mean={mean}, std={std}')


if __name__ == '__main__':
    preprocessor = UltrasoundPreprocessor(config_path='config.yaml')
    preprocessor.preprocess_and_save_dataset()