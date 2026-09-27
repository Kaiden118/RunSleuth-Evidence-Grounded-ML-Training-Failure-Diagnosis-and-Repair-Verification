@echo off
python "%~dp0check_camelyon.py" %*
exit /b %errorlevel%
