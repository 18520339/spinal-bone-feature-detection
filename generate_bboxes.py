import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import regionprops
from tqdm import tqdm

def extract_spine_bounding_boxes(
    mask_path, output_format='yolo', image_width=None, image_height=None, 
    min_area=50, min_height=10, min_width=10):
    '''
    Extract bounding boxes from a spine ultrasound segmentation mask, 
    with specialized filtering for vertebral structures.
    
    Args:
        mask_path (str): Path to the binary segmentation mask image
        output_format (str): Format of the output ('yolo' or 'voc')
        image_width (int): Width of the original image (needed for YOLO format)
        image_height (int): Height of the original image (needed for YOLO format)
        min_area (int): Minimum area of regions to consider
        min_height (int): Minimum height of bounding boxes
        min_width (int): Minimum width of bounding boxes
    
    Returns:
        list: List of bounding boxes in the specified format
    '''
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read the mask
    if mask is None:
        print(f'Error: Could not read mask at {mask_path}')
        return []
    
    # Get image dimensions if not provided
    if image_height is None: image_height = mask.shape[0]
    if image_width is None: image_width = mask.shape[1]
    
    # Create a binary mask (ensure the mask is binary)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Optional: Apply morphological operations to improve segmentation
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Label connected components in the mask
    labeled_mask = measure.label(binary_mask)
    
    # Extract properties for each labeled region
    regions = regionprops(labeled_mask)
    bounding_boxes = []
    
    for region in regions:
        # Get bounding box coordinates (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = region.bbox
        height, width = max_row - min_row, max_col - min_col # Calculate dimensions
        
        # Filter regions based on area and dimensions
        # These thresholds may need adjustment based on your specific dataset
        if region.area < min_area or height < min_height or width < min_width: continue
        
        # Optional: Filter based on aspect ratio for vertebral structures
        aspect_ratio = width / max(height, 1)  # Avoid division by zero
        if aspect_ratio > 3.0: continue # Vertebrae typically aren't too wide compared to height
        
        # Convert to the requested format
        if output_format.lower() == 'yolo': # YOLO format: [x_center, y_center, width, height] (normalized)
            x_center = ((min_col + max_col) / 2) / image_width
            y_center = ((min_row + max_row) / 2) / image_height
            width, height = width / image_width, height / image_height
            bounding_boxes.append([x_center, y_center, width, height])
        else: bounding_boxes.append([min_col, min_row, max_col, max_row]) # VOC format: [xmin, ymin, xmax, ymax]
    return bounding_boxes


def visualize_spine_segmentation_and_boxes(image_path, mask_path, output_path=None, output_format='yolo'):
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
        print('Error: Could not read one or both images')
        return
    
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) # Convert original to RGB (from BGR)
    image_with_boxes = original_rgb.copy() # Create a copy of the original for drawing boxes
    
    # Extract bounding boxes
    image_height, image_width = original.shape[:2]
    boxes = extract_spine_bounding_boxes(mask_path, output_format, image_width, image_height)
    
    # Draw bounding boxes on the image
    for box in boxes:
        if output_format.lower() == 'yolo':
            # Convert YOLO format to pixel coordinates
            x_center, y_center, width, height = box
            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height
            
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
        else: x_min, y_min, x_max, y_max = [int(coord) for coord in box] # VOC format
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
    
    
    # Create a figure with 3 subplots
    plt.subplot(1, 3, 1) # Original image
    plt.imshow(original_rgb)
    plt.title('Original Ultrasound')
    plt.axis('off')
    
    plt.subplot(1, 3, 2) # Mask
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3) # Image with bounding boxes
    plt.imshow(image_with_boxes)
    plt.title('Obtained Boxes')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path: # Save or show the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        # print(f'Visualization saved to {output_path}')
    else: plt.show()
    

def batch_process_spine_data(
    mask_dir, original_dir, output_dir, annotation_dir, 
    output_format='yolo', create_visualizations=True):
    '''
    Process all spine ultrasound images and masks in the given directories.
    
    Args:
        mask_dir (str): Directory containing segmentation masks
        original_dir (str): Directory containing original ultrasound images
        output_dir (str): Directory to save visualizations
        annotation_dir (str): Directory to save bounding box annotations
        output_format (str): Format of the output ('yolo' or 'voc')
        create_visualizations (bool): Whether to create visualization images
    '''
    # Create output directories if they don't exist and get all mask files
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    mask_files = glob.glob(os.path.join(mask_dir, '*.png')) + glob.glob(os.path.join(mask_dir, '*.jpg'))
    
    for mask_path in tqdm(mask_files):
        # Get the filename without extension
        filename = os.path.basename(mask_path)
        name, ext = os.path.splitext(filename)
        
        # Find the corresponding original image
        potential_exts = ['.png', '.jpg', '.jpeg']
        original_path = None
        
        for potential_ext in potential_exts:
            potential_path = os.path.join(original_dir, name + potential_ext)
            if os.path.exists(potential_path):
                original_path = potential_path
                break
        
        if original_path is None:
            print(f'Warning: No matching original image found for {filename}')
            continue
        
        # Extract bounding boxes
        original_img = cv2.imread(original_path)
        image_height, image_width = original_img.shape[:2]
        boxes = extract_spine_bounding_boxes(mask_path, output_format, image_width, image_height)
        
        # Save annotations
        annotation_path = os.path.join(annotation_dir, f'{name}.txt')
        with open(annotation_path, 'w') as f:
            for box in boxes:
                if output_format.lower() == 'yolo':
                    line = f'0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n' # YOLO format with class ID 0
                else: line = f'0,{box[0]},{box[1]},{box[2]},{box[3]}\n' # VOC format
                f.write(line)
        # print(f'Saved annotations for {filename} to {annotation_path}')
        
        # Create visualization if requested
        if create_visualizations:
            visualization_path = os.path.join(output_dir, f'{name}_visualization.png')
            visualize_spine_segmentation_and_boxes(original_path, mask_path, visualization_path, output_format)

if __name__ == '__main__':
    # Process all images in batch
    mask_dir = './dataset/mask'
    original_dir = './dataset/image'
    output_dir = './dataset/visualizations'
    annotation_dir = './dataset/annotations'
    batch_process_spine_data(mask_dir, original_dir, output_dir, annotation_dir, output_format='yolo')

    # Process a single pair
    # single_mask_path = './dataset/masks/example_mask.png'
    # single_image_path = './dataset/images/example_image.png'
    # single_output_path = './dataset/visualizations/example_visualization.png'
    # visualize_spine_segmentation_and_boxes(single_image_path, single_mask_path, single_output_path, 'yolo')