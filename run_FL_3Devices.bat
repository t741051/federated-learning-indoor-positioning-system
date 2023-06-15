@echo off
start cmd /K "python server_3Devices.py"
timeout /t 5 >nul
start cmd /K "python client_3Devices.py --partition 0"
start cmd /K "python client_3Devices.py --partition 1"
start cmd /K "python client_3Devices.py --partition 2"