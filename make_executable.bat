call ".\.venv\Scripts\activate.bat"
pip install pyinstaller
pyinstaller --onefile program.py 
xcopy /E /I "./config" "./dist/config"
xcopy /E /I "./scenes" "./dist/scenes"