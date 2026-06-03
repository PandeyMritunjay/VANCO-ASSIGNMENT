# ASL Detection - Streamlit Deployment

Live ASL detection web application deployed on Streamlit.

## Features

- **Real-time Detection**: Upload images or use webcam for live detection
- **YOLOv8 Model**: State-of-the-art object detection
- **Interactive UI**: Clean and intuitive interface
- **Performance Metrics**: Real-time FPS and latency display
- **Confidence Control**: Adjustable detection threshold

## Quick Start

### Local Deployment

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Upload your trained model:**
   - Place your trained `best.pt` file in this directory
   - Or upload it through the web interface

3. **Run the app:**
```bash
streamlit run app.py
```

4. **Open browser:**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy to Streamlit:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select this directory
   - Deploy

## Usage

### Upload Model
- Click "Browse files" in the sidebar
- Select your trained `.pt` model file
- The app will automatically load the model

### Image Detection
1. Select "Upload Image" option
2. Upload an image showing an ASL sign
3. View detection results with bounding boxes
4. Check confidence scores and performance metrics

### Webcam Detection
1. Select "Use Webcam" option
2. Click "Start Webcam"
3. Position your hand showing an ASL sign
4. View real-time detection
5. Click "Stop Webcam" when done

### Adjust Settings
- **Confidence Threshold**: Slider to adjust minimum confidence (0.1-0.9)
- Lower threshold = more detections (may include false positives)
- Higher threshold = fewer detections (may miss some signs)

## Model Requirements

- **Format**: YOLOv8 `.pt` file
- **Classes**: Trained on ASL alphabet signs
- **Input Size**: 640x640 (recommended)
- **Model Size**: YOLOv8-nano recommended for web deployment

## Performance

- **Inference Time**: 30-100ms (depending on model and hardware)
- **FPS**: 10-30 FPS (webcam mode)
- **Latency**: < 200ms for real-time detection

## Troubleshooting

**Model not loading:**
- Ensure the model file is in `.pt` format
- Check if the model is a valid YOLOv8 model
- Verify the file size (should be 6-110 MB)

**Webcam not working:**
- Check browser permissions for camera access
- Ensure no other application is using the webcam
- Try refreshing the page

**Slow performance:**
- Use YOLOv8-nano model (smallest)
- Reduce image resolution if possible
- Close other browser tabs

**No detections:**
- Lower the confidence threshold
- Ensure the image clearly shows the ASL sign
- Check if the model was trained on similar data

## File Structure

```
streamlit_deployment/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── best.pt            # Trained model (upload or place here)
```

## Notes

- The app requires a trained model to function
- For best results, use a model trained on diverse ASL data
- Webcam performance depends on internet connection and device hardware
- The app runs entirely in the browser after model loading
