@echo off
cd /d "%~dp0"
echo Dang khoi dong JetRacer Local ONNX Track Simulator...
echo Mo trinh duyet tai: http://127.0.0.1:8766
start "" http://127.0.0.1:8766
".venv_cuda\Scripts\python.exe" tools\sim_app.py --port 8766
pause
