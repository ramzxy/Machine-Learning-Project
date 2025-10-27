@echo off
cd /d "%~dp0"
echo Starting Diabetes Prediction App...
echo.
streamlit run app.py --server.port 8502


