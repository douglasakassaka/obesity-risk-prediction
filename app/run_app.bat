@echo off
cd /d "G:\Meu Drive\GitHub"
call venv\Scripts\activate.bat
cd obesity-risk-prediction
python app/train.py
python -m streamlit run app/app.py
pause