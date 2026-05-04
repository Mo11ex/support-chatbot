@echo off
cd /d C:\chat-bot-test\support-chatbot
backend\venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///ml/logs/mlflow.db --default-artifact-root ml/logs/mlruns