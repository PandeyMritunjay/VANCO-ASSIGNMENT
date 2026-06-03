"""
ASL Detection Streamlit App
Live webcam demo for American Sign Language detection using YOLOv8
No cv2 dependency for Streamlit Cloud compatibility
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from ultralytics import YOLO


# Page configuration
st.set_page_config(
    page_title="ASL Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_model():
    """Load YOLO model"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def process_image(image, model):
    """Process image and run inference"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run inference (YOLO accepts RGB numpy arrays)
    results = model(img_array, conf=0.5, verbose=False)
    
    # Create a copy for drawing
    img_pil = image.copy()
    draw = ImageDraw.Draw(img_pil)
    
    # Draw detections
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            # Try to use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw label background
            draw.rectangle([(x1, y1 - text_height - 10), (x1 + text_width, y1)], 
                         fill=(0, 255, 0))
            
            # Draw label text
            draw.text((x1, y1 - text_height - 8), label, fill=(255, 255, 255), font=font)
    
    return img_pil, results


def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model = load_model()
    if not model:
        st.error("Failed to load model. Please ensure yolov8n.pt is in the directory.")
        return
    
    st.success(f"Model loaded: {len(model.names)} classes")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Detection")
        
        # Input method selection
        input_method = st.radio(
            "Choose Input Method",
            ["Upload Image", "Use Webcam"],
            help="Select how you want to provide input for detection"
        )
        
        if input_method == "Upload Image":
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image showing an ASL sign"
            )
            
            if uploaded_file:
                # Read image
                image = Image.open(uploaded_file)
                
                # Display original
                st.subheader("Original Image")
                st.image(image, width=640)
                
                # Process
                with st.spinner("Processing..."):
                    start_time = time.time()
                    processed_img, results = process_image(image, model)
                    inference_time = (time.time() - start_time) * 1000
                
                # Display processed
                st.subheader("Detection Result")
                st.image(processed_img, width=640)
                
                # Display results
                st.subheader("Detection Details")
                for result in results:
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = model.names[class_id]
                            st.success(f"Detected: {class_name} (Confidence: {confidence:.2f})")
                    else:
                        st.warning("No ASL sign detected")
                
                # Metrics
                st.subheader("Performance Metrics")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Inference Time", f"{inference_time:.2f} ms")
                with col_m2:
                    st.metric("FPS", f"{1000/inference_time:.1f}")
        
        else:  # Webcam
            st.subheader("Webcam Detection")
            st.warning("⚠️ Webcam is not available on Streamlit Cloud. Please use 'Upload Image' option instead.")
            st.info("For webcam detection, run this app locally with: streamlit run app.py")
    
    with col2:
        st.header("About")
        st.markdown("""
        <div class="info-box">
        <h3>ASL Detection System</h3>
        <p>This system uses YOLOv8 to detect and classify American Sign Language alphabet signs in real-time.</p>
        
        <h4>Features:</h4>
        <ul>
        <li>Real-time detection</li>
        <li>Bounding box visualization</li>
        <li>Confidence scores</li>
        <li>Multiple ASL classes</li>
        </ul>
        
        <h4>Model Info:</h4>
        <p>Architecture: YOLOv8-nano</p>
        <p>Input Size: 640x640</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Model Classes")
        if model:
            for i, class_name in enumerate(model.names):
                st.write(f"• {class_name}")


if __name__ == "__main__":
    main()
