"""
ASL Detection Streamlit App
Simple image upload demo - object detection requires local deployment
"""

import streamlit as st
from PIL import Image
import numpy as np


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


def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 ASL Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Info box
    st.markdown("""
    <div class="info-box">
    <h3>⚠️ Object Detection on Streamlit Cloud</h3>
    <p>Object detection requires OpenCV (cv2) which is not available on Streamlit Cloud.</p>
    
    <h4>Current Status:</h4>
    <ul>
    <li>✅ Image upload working</li>
    <li>❌ Object detection disabled (requires local deployment)</li>
    </ul>
    
    <h4>To Run Full Detection:</h4>
    <ol>
    <li>Clone this repository locally</li>
    <li>Install dependencies: pip install -r requirements.txt</li>
    <li>Run locally: streamlit run app_ultralytics.py</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Image upload demo
    st.header("Image Upload Demo")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to test the upload functionality"
    )
    
    if uploaded_file:
        # Read image
        image = Image.open(uploaded_file)
        
        # Display image
        st.subheader("Uploaded Image")
        st.image(image, width=640)
        
        # Get image info
        img_array = np.array(image)
        st.subheader("Image Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Width", img_array.shape[1])
        with col2:
            st.metric("Height", img_array.shape[0])
        
        st.info("Object detection will be available when running locally with OpenCV support.")


if __name__ == "__main__":
    main()
