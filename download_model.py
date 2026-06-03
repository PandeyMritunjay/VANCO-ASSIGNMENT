"""
Script to pre-download the SentenceTransformer model to avoid download issues during app startup.
Run this once before running the Streamlit app.
"""

import os
from sentence_transformers import SentenceTransformer

# Set cache directory
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR

print("Downloading SentenceTransformer model...")
print(f"Cache directory: {HF_CACHE_DIR}")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=HF_CACHE_DIR)
    print("Model downloaded successfully!")
    print(f"Model saved to: {HF_CACHE_DIR}")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please check your internet connection and try again.")
