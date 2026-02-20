@echo off
echo Deleting all the files in the 'logs' folder...
del /q /f "logs\*.*"
echo Deleting pycache files...
del /q /f "__pycache__\*.*"
del /q /f "scripts\__pycache__\*.*"
del /q /f "model\__pycache__\*.*"
del /q /f "data\__pycache__\*.*"
echo Done.