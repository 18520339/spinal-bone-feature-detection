import cv2
import matplotlib.pyplot as plt
from skimage import measure
from pathlib import Path
from tqdm import tqdm


def extract_spine_boxes(mask_path, padding=0):
    '''
    Extract bounding boxes from a spine ultrasound segmentation mask, 
    with specialized filtering for vertebral structures.
    
    Args:
        mask_path (str): Path to the binary segmentation mask image
        padding (int): Padding to add around detected objects in pixels
    
    Returns:
        list: List of bounding boxes in the specified format
    '''
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read the mask
    if mask is None:
        raise ValueError(f"Failed to read mask image from {mask_path}")
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # Create a binary mask (ensure the mask is binary)
    
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
        bounding_boxes.append([class_id, x_min, y_min, x_max, y_max])
    return bounding_boxes


def visualize_spine_segmentation_and_boxes(output_path, image_path, mask_path, boxes):
    '''
    Create a visualization showing the original image, mask, and detected bounding boxes.
    
    Args:
        output_path (str): Path to save the visualization image
        image_path (str): Path to the original ultrasound image
        mask_path (str): Path to the segmentation mask
        boxes (list): List of bounding boxes to draw on the image
    '''
    original = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or mask is None:
        raise ValueError(f"Failed to read image or mask from {image_path} or {mask_path}")
    
    # Extract bounding boxes
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) # Convert original to RGB (from BGR)
    image_with_boxes = original_rgb.copy() # Create a copy of the original for drawing boxes
    
    # Draw bounding boxes on the image and count classes
    for idx, (class_id, x_min, y_min, x_max, y_max) in enumerate(boxes):
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
    mask_dir, original_dir, annotation_dir, viz_dir = Path(mask_dir), Path(original_dir), Path(annotation_dir), Path(viz_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    if viz_dir: viz_dir.mkdir(parents=True, exist_ok=True)
    
    for mask_path in tqdm(list(mask_dir.glob('*.png'))):
        filename = mask_path.stem
        original_path = original_dir / f'{filename}.jpg'
        
        # Extract bounding boxes
        try: boxes = extract_spine_boxes(mask_path)
        except Exception as e:
            print(f'Error processing {mask_path}: {e}')
            continue
        
        # Save annotations
        with open(annotation_dir / f'{filename}.txt', 'w') as f:
            for class_id, x_min, y_min, x_max, y_max in boxes:
                f.write(f'{class_id} {x_min} {y_min} {x_max} {y_max}\n')
        
        if viz_dir is not None: # Create visualization if requested
            visualize_spine_segmentation_and_boxes(viz_dir / f'{filename}_visualization.png', original_path, mask_path, boxes)


if __name__ == '__main__':
    mask_dir = './datasets/ultrasound/mask'
    original_dir = './datasets/ultrasound/image'
    annotation_dir = './datasets/ultrasound/annotations'
    viz_dir = './datasets/ultrasound/visualizations'
    # batch_process_spine_data(mask_dir, original_dir, annotation_dir, viz_dir)
    
    mask_dir, original_dir, annotation_dir, viz_dir = Path(mask_dir), Path(original_dir), Path(annotation_dir), Path(viz_dir)
    if viz_dir: # Create output directory for visualizations if it doesn't exist
        viz_dir.mkdir(parents=True, exist_ok=True) 

    for mask_path in tqdm(list(mask_dir.glob('*.png'))): # Read boxes from each file and visualize them
        filename = mask_path.stem
        with open(annotation_dir / f'{filename}.txt', 'r') as f: # Read the bounding boxes from the annotation file
            visualize_spine_segmentation_and_boxes(
                viz_dir / f'{filename}_visualization.png',
                original_dir / f'{filename}.jpg', 
                mask_path, boxes=[list(map(int, line.strip().split())) for line in f.readlines()]
            )