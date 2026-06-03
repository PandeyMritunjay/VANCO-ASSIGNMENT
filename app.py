"""
ASL Detection Streamlit App
Live webcam demo for American Sign Language detection using YOLOv8
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run inference
    results = model(img_bgr, conf=0.5, verbose=False)
    
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
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_bgr, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_bgr, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb, results


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
                st.image(image, use_column_width=True)
                
                # Process
                with st.spinner("Processing..."):
                    start_time = time.time()
                    processed_img, results = process_image(image, model)
                    inference_time = (time.time() - start_time) * 1000
                
                # Display processed
                st.subheader("Detection Result")
                st.image(processed_img, use_column_width=True)
                
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
            st.info("Click 'Start Webcam' to begin live detection")
            
            if st.button("Start Webcam"):
                # Initialize webcam
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Could not open webcam. Please check your camera.")
                    return
                
                st_frame = st.empty()
                stop_button = st.button("Stop Webcam")
                
                fps_counter = 0
                fps_start_time = time.time()
                
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame")
                        break
                    
                    # Convert to PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # Process
                    start_time = time.time()
                    processed_img, results = process_image(image, model)
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Calculate FPS
                    fps_counter += 1
                    if time.time() - fps_start_time >= 1.0:
                        fps = fps_counter
                        fps_counter = 0
                        fps_start_time = time.time()
                    else:
                        fps = fps_counter / (time.time() - fps_start_time + 0.001)
                    
                    # Display
                    st_frame.image(processed_img, channels="RGB", use_column_width=True)
                    
                    # Display info
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("FPS", f"{fps:.1f}")
                    with col_info2:
                        st.metric("Latency", f"{inference_time:.1f} ms")
                    with col_info3:
                        if len(results[0].boxes) > 0:
                            class_id = int(results[0].boxes[0].cls[0])
                            confidence = float(results[0].boxes[0].conf[0])
                            class_name = model.names[class_id]
                            st.metric("Detected", f"{class_name} ({confidence:.2f})")
                        else:
                            st.metric("Detected", "None")
                    
                    # Check for stop
                    if stop_button:
                        cap.release()
                        st.success("Webcam stopped")
                        break
                
                cap.release()
    
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
