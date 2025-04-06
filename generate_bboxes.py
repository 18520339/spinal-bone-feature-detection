import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm

def extract_spine_yolo_boxes(mask_path, image_width=None, image_height=None, padding=0):
    '''
    Extract bounding boxes from a spine ultrasound segmentation mask, 
    with specialized filtering for vertebral structures.
    
    Args:
        mask_path (str): Path to the binary segmentation mask image
        image_width (int): Width of the original image (needed for YOLO format)
        image_height (int): Height of the original image (needed for YOLO format)
        padding (int): Padding to add around detected objects in pixels
    
    Returns:
        list: List of bounding boxes in the specified format
    '''
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read the mask
    if mask is None:
        raise ValueError(f"Failed to read mask image from {mask_path}")
    
    # Get image dimensions if not provided
    if image_height is None: image_height = mask.shape[0]
    if image_width is None: image_width = mask.shape[1]
    
    # Create a binary mask (ensure the mask is binary)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to improve segmentation
    # kernel = np.ones((3, 3), np.uint8)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    labeled_mask = measure.label(binary_mask) # Label connected components in the mask
    regions = measure.regionprops(labeled_mask) # Extract properties for each labeled region
    bounding_boxes = []
    
    # Sort regions by y-coordinate (highest y = bottom of image)
    sorted_regions = sorted(regions, key=lambda r: r.bbox[0], reverse=True)
    num_lumbar = min(6, len(sorted_regions)) # Determine how many regions should be lumbar (class 1)
    
    for i, region in enumerate(sorted_regions):
        y_min, x_min, y_max, x_max = region.bbox # Get bounding box coordinates
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(mask.shape[1], x_max + padding)
        y_max = min(mask.shape[0], y_max + padding)
        class_id = 1 if i < num_lumbar else 0 # Bottom 6 regions are lumbar (class 1), the rest are thoracic (class 0)
        
        # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
        x_center = (x_min + x_max) / (2 * image_width)
        y_center = (y_min + y_max) / (2 * image_height)
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        bounding_boxes.append([class_id, x_center, y_center, width, height])
    return bounding_boxes


def visualize_spine_segmentation_and_boxes(output_path, image_path, mask_path, boxes):
    '''
    Create a visualization showing the original image, mask, and detected bounding boxes.
    
    Args:
        image_path (str): Path to the original ultrasound image
        mask_path (str): Path to the segmentation mask
        output_path (str): Path to save the visualization image
        output_format (str): Format of the bounding boxes ('yolo' or 'voc')
    '''
    original = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or mask is None:
        raise ValueError(f"Failed to read image or mask from {image_path} or {mask_path}")
    
    # Extract bounding boxes
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) # Convert original to RGB (from BGR)
    image_with_boxes = original_rgb.copy() # Create a copy of the original for drawing boxes
    image_height, image_width = original.shape[:2]
    
    # Draw bounding boxes on the image and count classes
    for idx, (class_id, x_center, y_center, width, height) in enumerate(boxes):
        # Convert YOLO format to pixel coordinates
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        
        # color = (220, 20, 60) if class_id == 0 else (46, 204, 113)  # Red for Thoracic, Green for Lumbar
        color = (240, 128, 128) if class_id == 0 else (144, 238, 144) # Red for Thoracic, Green for Lumbar
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, 5) # Draw the bounding box
        
        # Calculate center position for the label and text position to center it
        label_x = x_min + (x_max - x_min) // 2
        label_y = y_min + (y_max - y_min) // 2
        font_scale, font_thickness = 2.0, 5
        
        (text_width, text_height), _ = cv2.getTextSize(f'{idx + 1}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = label_x - text_width // 2
        text_y = label_y + text_height // 2
        
        # Draw the label text in the center of the box
        cv2.putText(
            image_with_boxes, f'{idx + 1}', (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness
        )
    
    # Create a figure with 4 subplots
    plt.subplot(1, 3, 1) # Original image
    plt.imshow(original_rgb)
    plt.title('Original Ultrasound', fontsize=7)
    plt.axis('off')
    
    plt.subplot(1, 3, 2) # Mask
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask', fontsize=7)
    plt.axis('off')
    
    plt.subplot(1, 3, 3) # Image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('Obtained Boxes', fontsize=7)
    plt.axis('off')
    
    if output_path: # Save or show the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else: plt.show()
    

def batch_process_spine_data(mask_dir, original_dir, annotation_dir, viz_dir=None):
    '''
    Process all spine ultrasound images and masks in the given directories.
    
    Args:
        mask_dir (str): Directory containing segmentation masks
        original_dir (str): Directory containing original ultrasound images
        annotation_dir (str): Directory to save bounding box annotations
        viz_dir (str): Directory to save visualizations. If None, no visualizations are saved.
    '''
    # Create output directories if they don't exist and get all mask files
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True) if viz_dir else None
    mask_files = glob.glob(os.path.join(mask_dir, '*.png')) + glob.glob(os.path.join(mask_dir, '*.jpg'))
    
    for mask_path in tqdm(mask_files):
        # Get the filename without extension
        filename = os.path.basename(mask_path)
        name, ext = os.path.splitext(filename)
        
        # Find the corresponding original image
        original_path = os.path.join(original_dir, name + '.jpg')
        if original_path is None:
            print(f'Warning: No matching original image found for {filename}')
            continue
        
        # Extract bounding boxes
        original_img = cv2.imread(original_path)
        image_height, image_width = original_img.shape[:2]
        try:
            boxes = extract_spine_yolo_boxes(mask_path, image_width, image_height)
        except Exception as e:
            print(f'Error processing {mask_path}: {e}')
            continue
        
        # Save annotations
        annotation_path = os.path.join(annotation_dir, f'{name}.txt')
        with open(annotation_path, 'w') as f:
            for class_id, x_center, y_center, width, height in boxes:
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
        
        # Create visualization if requested
        if viz_dir is not None:
            visualization_path = os.path.join(viz_dir, f'{name}_visualization.png')
            visualize_spine_segmentation_and_boxes(visualization_path, original_path, mask_path, boxes)


if __name__ == '__main__':
    mask_dir = './dataset/mask'
    original_dir = './dataset/image'
    annotation_dir = './dataset/annotations'
    viz_dir = './dataset/visualizations'
    # batch_process_spine_data(mask_dir, original_dir, annotation_dir, viz_dir)

    # Read boxes from each file and visualize them
    for mask_path in tqdm(glob.glob(os.path.join(mask_dir, '*.png')) + glob.glob(os.path.join(mask_dir, '*.jpg'))):
        # Get the filename without extension
        filename = os.path.basename(mask_path)
        name, ext = os.path.splitext(filename)
        
        # Find the corresponding original image
        original_path = os.path.join(original_dir, name + '.jpg')
        if original_path is None:
            print(f'Warning: No matching original image found for {filename}')
            continue
        
        # Read the bounding boxes from the annotation file
        annotation_path = os.path.join(annotation_dir, f'{name}.txt')
        with open(annotation_path, 'r') as f:
            boxes = [list(map(float, line.strip().split())) for line in f.readlines()]
        
        # Create visualization
        visualization_path = os.path.join(viz_dir, f'{name}_visualization.png')
        visualize_spine_segmentation_and_boxes(visualization_path, original_path, mask_path, boxes)