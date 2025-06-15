@echo off
echo Building Doctor31 Medical Validator for Windows...
echo.

pip install --upgrade pyinstaller && pyinstaller --clean --onefile main.py --name Doctor31_Medical_Validator --add-data "src/templates:src/templates" --add-data "src/static:src/static" --add-data "src/validation.py:src" --add-data "src/web_gui.py:src" --add-data "src/log_config.py:src" --add-data "src/__init__.py:src" --hidden-import flask --hidden-import pandas --hidden-import numpy --hidden-import sklearn --hidden-import sklearn.ensemble --hidden-import sklearn.preprocessing --hidden-import openpyxl --hidden-import werkzeug --hidden-import jinja2 --hidden-import click --hidden-import itsdangerous --hidden-import markupsafe --hidden-import joblib --hidden-import scipy --hidden-import threadpoolctl --console --noconfirm

echo.
echo Build completed! Check the 'dist' folder for the executable.
pause
