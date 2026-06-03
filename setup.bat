@echo off
echo Setting up ASL Detection Streamlit Deployment...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

echo Setup complete!
echo Run the app with: streamlit run app.py
