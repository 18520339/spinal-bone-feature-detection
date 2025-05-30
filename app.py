import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
from io import BytesIO
from preprocessor import UltrasoundPreprocessor
from model import MultiTaskModel
import seaborn as sns

# Set page configuration for a professional look
st.set_page_config(
    page_title="Scoliosis Detection Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful and modern design
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 36px;
        margin-bottom: 10px;
    }
    h2 {
        color: #34495e;
        font-size: 24px;
        margin-top: 20px;
    }
    .stFileUploader {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        background-color: #ecf0f1;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stSlider label {
        color: #34495e;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def plot_cf_matrix(cf_matrix, label_names, ax=None):
    labels = np.asarray([f'{name}\n{count:,}\n{percent:.2%}' for name, count, percent in zip(
        ['True Neg', 'False Pos', 'False Neg', 'True Pos'], # Group names
        cf_matrix.flatten(), # Group counts
        cf_matrix.flatten() / np.sum(cf_matrix) # Group percentages
    )]).reshape(2, 2)
    sns.heatmap(
        cf_matrix, fmt='', annot=labels,
        cmap='YlGnBu', square=True, annot_kws={'size': 12},
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )

# Sidebar for project information and controls
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrH9S3ypULnvn37eQrbmrJ2AMTJD8pv_z5vA&s", caption="Scoliosis Detection")
    st.header("About This Demo")
    st.markdown("""
    This demo showcases an AI model for detecting Thoracic and Lumbar vertebrae in ultrasound images of the spine, designed to assist in scoliosis assessment. Upload an ultrasound image to see the model in action!
    
    **Steps:**
    1. Upload a grayscale ultrasound image.
    2. (Optional) Upload a ground truth annotation file for comparison.
    3. Click "Run Detection" to process the image.
    4. Adjust visualization settings and explore results.
    """)
    st.header("Model Details")
    st.markdown("""
    - **Model**: CNN + GNN (MultiTaskModel)
    - **Classes**: Thoracic (Red), Lumbar (Green)
    - **Input Size**: 448x224 pixels
    - **Trained on**: 50 ultrasound images
    """)
    st.header("Visualization Controls")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Filter predictions below this confidence score.")
    nms_iou_threshold = st.slider("NMS IoU Threshold", 0.0, 1.0, 0.3, 0.05, help="IoU threshold for Non-Maximum Suppression to reduce overlapping boxes.")
    show_gt = st.checkbox("Show Ground Truth Boxes", value=True)
    show_pred = st.checkbox("Show Predicted Boxes", value=True)
    show_cobb = st.checkbox("Show Cobb Angle Curve", value=True)
    gt_scale_preprocessed = st.checkbox("Ground Truth Boxes are in Preprocessed Scale (448x224)", value=False, help="Check if the ground truth boxes are in the preprocessed image scale (448x224). Uncheck if they are in the original image scale.")

# Main title and description
st.title("🩺 Scoliosis Detection in Ultrasound Images")
st.markdown("Upload an ultrasound image to detect Thoracic and Lumbar vertebrae using our GNN-based model, and explore scoliosis assessment features.")

# File uploader for image and optional annotations
uploaded_file = st.file_uploader("Choose an ultrasound image", type=["jpg", "jpeg", "png"], help="Upload a grayscale ultrasound image.")
uploaded_annotation = st.file_uploader("Choose a ground truth annotation file (optional)", type=["txt"], help="Upload a .txt file with annotations in format: class x_min y_min x_max y_max")

# Placeholder for the image and results
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image_np = np.array(image)  # Convert to numpy array for preprocessing
    orig_h, orig_w = image_np.shape[:2]  # Store original dimensions
    with col1:
        st.subheader("Original Image")
        st.image(image_np, caption=f"Uploaded Ultrasound Image ({orig_h}x{orig_w})", use_container_width=True)

    # Load ground truth annotations if provided
    gt_boxes = None
    gt_labels = None
    gt_boxes_input = None
    if uploaded_annotation is not None:
        with uploaded_annotation:
            annotations = [line.strip().split() for line in uploaded_annotation.readlines()]
            annotations = np.array(annotations, dtype=np.float32)
            gt_boxes = annotations[:, 1:5]  # x_min, y_min, x_max, y_max
            gt_labels = annotations[:, 0].astype(np.int64)  # 0: Thoracic, 1: Lumbar

            # Scale ground truth boxes for visualization and model input
            height, width = 448, 224  # Preprocessed image size
            if gt_scale_preprocessed:
                # Ground truth boxes are in preprocessed scale (448x224)
                scale_x = orig_w / width
                scale_y = orig_h / height
                gt_boxes[:, [0, 2]] *= scale_x
                gt_boxes[:, [1, 3]] *= scale_y
                gt_boxes_input = annotations[:, 1:5].copy()
                gt_boxes_input[:, [0, 2]] = np.clip(gt_boxes_input[:, [0, 2]], 0, width)
                gt_boxes_input[:, [1, 3]] = np.clip(gt_boxes_input[:, [1, 3]], 0, height)
            else:
                # Ground truth boxes are in original image scale
                gt_boxes = annotations[:, 1:5].copy()
                gt_boxes[:, [0, 2]] = np.clip(gt_boxes[:, [0, 2]], 0, orig_w)
                gt_boxes[:, [1, 3]] = np.clip(gt_boxes[:, [1, 3]], 0, orig_h)
                scale_x = width / orig_w
                scale_y = height / orig_h
                gt_boxes_input = annotations[:, 1:5].copy()
                gt_boxes_input[:, [0, 2]] *= scale_x
                gt_boxes_input[:, [1, 3]] *= scale_y

            # Check if coordinates are within expected ranges
            if gt_scale_preprocessed:
                if (gt_boxes_input[:, [0, 2]] > width).any() or (gt_boxes_input[:, [1, 3]] > height).any():
                    st.warning("Ground truth box coordinates exceed preprocessed image dimensions (448x224). Clamped to bounds.")
            else:
                if (gt_boxes[:, [0, 2]] > orig_w).any() or (gt_boxes[:, [1, 3]] > orig_h).any():
                    st.warning("Ground truth box coordinates exceed original image dimensions. Clamped to bounds.")

            st.write("Ground Truth Boxes (Original Scale):")
            st.write(gt_boxes[:3])
            st.write("Ground Truth Boxes (Preprocessed Scale for Model Input):")
            st.write(gt_boxes_input[:3])

    # Button to run detection
    if st.button("Run Detection"):
        # Initialize the preprocessor
        preprocessor = UltrasoundPreprocessor(config_path="config.yaml")

        # Preprocess the image and store intermediate results
        with st.spinner("Preprocessing image..."):
            image_denoised = cv2.fastNlMeansDenoising(image_np, h=preprocessor.despeckle_h, templateWindowSize=7, searchWindowSize=21)
            image_clahe = preprocessor.clahe.apply(image_denoised)
            edges = preprocessor.compute_edges(image_clahe)
            processed_image = preprocessor.preprocess(image_np)
            processed_image_tensor = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        # Display preprocessing steps
        with st.expander("View Preprocessing Steps"):
            col_pre1, col_pre2, col_pre3 = st.columns(3)
            with col_pre1:
                st.image(image_denoised, caption="After Denoising", use_container_width=True)
            with col_pre2:
                st.image(image_clahe, caption="After CLAHE", use_container_width=True)
            with col_pre3:
                st.image(edges, caption="Edge Map", use_container_width=True)

        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MultiTaskModel(num_classes=2).to(device)
        model.load_state_dict(torch.load("graph-scoliosis/weights/best_mAP50.pth", map_location=device))
        model.eval()

        # Use ground truth boxes if provided; otherwise, generate proposals
        height, width = 448, 224  # Preprocessed image size
        if gt_boxes_input is not None:
            boxes = torch.tensor(gt_boxes_input, dtype=torch.float32).to(device)
            boxes_by_images = [boxes]
            st.write("Using ground truth boxes for inference:")
            st.write(f"Number of boxes: {len(boxes)}")
            st.write(f"Sample boxes (preprocessed scale): {boxes[:3].cpu().numpy()}")
        else:
            # Generate bounding box proposals based on spinal structure
            boxes = []
            center_x = width // 2
            box_width = 40
            box_height = 30
            step_size = 20
            for y in range(0, height - box_height, step_size):
                x_min = center_x - box_width // 2
                x_max = center_x + box_width // 2
                y_min = y
                y_max = y + box_height
                boxes.append([x_min, y_min, x_max, y_max])
            boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
            boxes_by_images = [boxes]
            st.write("Using generated proposals for inference:")
            st.write(f"Number of boxes: {len(boxes)}")
            st.write(f"Sample boxes (preprocessed scale): {boxes[:3].cpu().numpy()}")

        # Run inference
        with st.spinner("Running detection..."):
            with torch.no_grad():
                pred_boxes, pred_labels, edge_sim = model(processed_image_tensor.to(device), boxes, boxes_by_images)
                pred_scores = F.softmax(pred_labels, dim=-1).max(dim=-1)[0].cpu().numpy()
                pred_labels = pred_labels.argmax(dim=-1).cpu().numpy()
                pred_boxes = pred_boxes.cpu().numpy()

        # Store original predictions for dynamic filtering
        orig_pred_boxes = pred_boxes.copy()
        orig_pred_labels = pred_labels.copy()
        orig_pred_scores = pred_scores.copy()

        # Filter predictions based on confidence threshold
        mask = orig_pred_scores >= confidence_threshold
        pred_boxes = orig_pred_boxes[mask]
        pred_labels = orig_pred_labels[mask]
        pred_scores = orig_pred_scores[mask]

        # Apply Non-Maximum Suppression (NMS)
        if len(pred_boxes) > 0:
            boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(pred_scores, dtype=torch.float32)
            nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold=nms_iou_threshold)
            pred_boxes = pred_boxes[nms_indices.numpy()]
            pred_labels = pred_labels[nms_indices.numpy()]
            pred_scores = pred_scores[nms_indices.numpy()]

        # Scale the predicted boxes back to the original image dimensions
        scale_x = orig_w / width
        scale_y = orig_h / height
        pred_boxes_scaled = pred_boxes.copy()
        pred_boxes_scaled[:, [0, 2]] *= scale_x
        pred_boxes_scaled[:, [1, 3]] *= scale_y

        # Compute Cobb angle if requested
        cobb_angle = None
        curve_points = None
        if show_cobb:
            if len(pred_boxes_scaled) > 2:  # Need at least 3 points for a curve
                # Compute box centers
                centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in pred_boxes_scaled])
                centers = centers[np.argsort(centers[:, 1])]  # Sort by y-coordinate (top to bottom)
                x, y = centers[:, 0], centers[:, 1]

                # Fit a cubic spline
                cs = CubicSpline(y, x)
                y_fit = np.linspace(y.min(), y.max(), 100)
                x_fit = cs(y_fit)

                # Compute tangents at the top and bottom points
                dy = 1.0  # Small increment for derivative
                tangent_top = (cs(y_fit[0] + dy) - cs(y_fit[0])) / dy
                tangent_bottom = (cs(y_fit[-1] + dy) - cs(y_fit[-1])) / dy

                # Compute Cobb angle (angle between tangents)
                angle_rad = np.abs(np.arctan(tangent_top) - np.arctan(tangent_bottom))
                cobb_angle = np.degrees(angle_rad)

                # Store curve points for visualization
                curve_points = np.vstack((x_fit, y_fit)).T
            else:
                st.warning("Not enough detected boxes (minimum 3 required) to compute the Cobb angle.")

        # Detection statistics
        st.header("Detection Statistics")
        num_thoracic = np.sum(pred_labels == 0) if len(pred_labels) > 0 else 0
        num_lumbar = np.sum(pred_labels == 1) if len(pred_labels) > 0 else 0
        avg_confidence = np.mean(pred_scores) if len(pred_scores) > 0 else 0.0
        st.write(f"Number of Thoracic Vertebrae Detected: {num_thoracic}")
        st.write(f"Number of Lumbar Vertebrae Detected: {num_lumbar}")
        st.write(f"Average Confidence Score: {avg_confidence:.2f}")

        # Confusion matrix if ground truth is available
        if gt_labels is not None and len(pred_boxes) > 0:
            pred_labels_matched = []
            for box in pred_boxes:
                # Find the closest ground truth box (in preprocessed scale)
                distances = np.linalg.norm(gt_boxes_input[:, :2] - box[:2], axis=1)
                if len(distances) > 0:
                    closest_idx = np.argmin(distances)
                    pred_labels_matched.append(gt_labels[closest_idx])
                else:
                    pred_labels_matched.append(-1)  # No match
            pred_labels_matched = np.array(pred_labels_matched)
            # Compute confusion matrix
            cm = np.zeros((2, 2), dtype=np.int32)  # 2x2 matrix for Thoracic (0) and Lumbar (1)
            for pred, gt in zip(pred_labels, pred_labels_matched):
                if gt != -1:  # Only count matched boxes
                    cm[pred, gt] += 1
            fig, ax = plt.subplots(figsize=(5, 5))
            # im = ax.imshow(cm, cmap="Blues")
            # ax.set_xticks([0, 1])
            # ax.set_yticks([0, 1])
            # ax.set_xticklabels(["Thoracic", "Lumbar"])
            # ax.set_yticklabels(["Thoracic", "Lumbar"])
            # ax.set_xlabel("Ground Truth")
            # ax.set_ylabel("Predicted")
            # plt.colorbar(im, label="Count")
            # for i in range(2):
            #     for j in range(2):
            #         ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plot_cf_matrix(cm, ["Thoracic", "Lumbar"], ax=ax)
            st.pyplot(fig)

        # Visualize the results
        image_with_boxes = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        if show_gt and gt_boxes is not None and gt_labels is not None:
            for box, label in zip(gt_boxes, gt_labels):
                x_min, y_min, x_max, y_max = box.astype(int)
                color = (0, 0, 255)  # Blue for ground truth
                cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, 2)
                label_text = "GT: " + ("Thoracic" if label == 0 else "Lumbar")
                cv2.putText(image_with_boxes, label_text, (x_min, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if show_pred and len(pred_boxes_scaled) > 0:
            for box, label, score in zip(pred_boxes_scaled, pred_labels, pred_scores):
                x_min, y_min, x_max, y_max = box.astype(int)
                color = (255, 0, 0) if label == 0 else (0, 255, 0)
                cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), color, 3)
                label_text = f"{'Thoracic' if label == 0 else 'Lumbar'} ({score:.2f})"
                cv2.putText(image_with_boxes, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if show_cobb and curve_points is not None:
            for i in range(len(curve_points) - 1):
                pt1 = (int(curve_points[i, 0]), int(curve_points[i, 1]))
                pt2 = (int(curve_points[i + 1, 0]), int(curve_points[i + 1, 1]))
                cv2.line(image_with_boxes, pt1, pt2, (0, 255, 255), 2)  # Yellow curve
            st.write(f"Estimated Cobb Angle: {cobb_angle:.2f} degrees")

        with col2:
            st.subheader("Detection Results")
            caption = "Detected Vertebrae (Red: Thoracic, Green: Lumbar, Blue: Ground Truth, Yellow: Cobb Angle Curve)"
            st.image(image_with_boxes, caption=caption, use_container_width=True)

        # Download options
        st.header("Download Results")
        # Download annotated image
        _, img_buffer = cv2.imencode(".png", image_with_boxes)
        img_bytes = img_buffer.tobytes()
        st.download_button(
            label="Download Annotated Image",
            data=img_bytes,
            file_name="annotated_image.png",
            mime="image/png"
        )
        # Download predictions as text file
        pred_text = "class x_min y_min x_max y_max confidence\n"
        for box, label, score in zip(pred_boxes_scaled, pred_labels, pred_scores):
            pred_text += f"{label} {box[0]} {box[1]} {box[2]} {box[3]} {score}\n"
        st.download_button(
            label="Download Predictions",
            data=pred_text,
            file_name="predictions.txt",
            mime="text/plain"
        )

        # Display edge similarities
        with st.expander("View Edge Similarities (GNN Output)"):
            edge_sim = edge_sim.cpu().numpy()
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(edge_sim, cmap="viridis")
            plt.colorbar(im, label="Edge Similarity")
            plt.title("GNN Edge Similarity Matrix")
            st.pyplot(fig)

# Footer
st.markdown(
    """
    <hr style='border: 1px solid #3498db;'>
    <p style='text-align: center; color: #7f8c8d;'>
    Developed by Group 15 | 49275 Neural Networks and Fuzzy Logic 
    </p>
    """,
    unsafe_allow_html=True
)