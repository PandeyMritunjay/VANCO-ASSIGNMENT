#!/bin/bash

echo "Setting up ASL Detection Streamlit Deployment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "Setup complete!"
echo "Run the app with: streamlit run app.py"
