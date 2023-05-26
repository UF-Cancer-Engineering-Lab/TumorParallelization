source ./.venv/bin/activate
pip install pyinstaller
pyinstaller --onefile program.py 
cp -r ./config ./dist/
cp -r ./scenes ./dist/