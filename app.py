import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from ultralytics import YOLO

st.set_page_config(page_title="ASL Detection", page_icon="??", layout="centered")

st.markdown('<h1 style="text-align:center;color:#1f77b4;font-size:3rem;font-weight:bold;">?? ASL Detection System</h1>', unsafe_allow_html=True)
st.markdown("---")

def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(image, model):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model(img_bgr, conf=0.5, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_bgr, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, results

model = load_model()
if not model:
    st.error("Failed to load model. Please ensure yolov8n.pt is in the directory.")
else:
    st.success(f"Model loaded: {len(model.names)} classes")

input_method = st.radio("Choose Input Method", ["Upload Image", "Use Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        with st.spinner("Processing..."):
            start_time = time.time()
            processed_img, results = process_image(image, model)
            inference_time = (time.time() - start_time) * 1000
        
        st.subheader("Detection Result")
        st.image(processed_img, use_column_width=True)
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    st.success(f"Detected: {class_name} (Confidence: {confidence:.2f})")
            else:
                st.warning("No ASL sign detected")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Inference Time", f"{inference_time:.2f} ms")
        with col_m2:
            st.metric("FPS", f"{1000/inference_time:.1f}")

else:
    st.subheader("Webcam Detection")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera.")
        else:
            st_frame = st.empty()
            stop_button = st.button("Stop Webcam")
            fps_counter = 0
            fps_start_time = time.time()
            
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                start_time = time.time()
                processed_img, results = process_image(image, model)
                inference_time = (time.time() - start_time) * 1000
                
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                else:
                    fps = fps_counter / (time.time() - fps_start_time + 0.001)
                
                st_frame.image(processed_img, channels="RGB", use_column_width=True)
                
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
                
                if stop_button:
                    cap.release()
                    st.success("Webcam stopped")
                    break
            
            cap.release()

