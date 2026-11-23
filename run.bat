@echo off
rem Voice-Enabled Data Analyst: one-click setup and start for Windows.
rem Double-click this file, or run "run.bat" in a terminal. "run.bat setup" only sets up.
rem The first run needs internet (to download packages) and takes a few minutes.
setlocal EnableExtensions
cd /d "%~dp0"
title Voice Data Analyst
set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto :have_venv
echo.
echo === First-time setup: creating a private Python environment in .venv ===
call :find_python
if not defined PYTHON goto :no_python
echo Using %PYTHON%
%PYTHON% -m venv .venv
if errorlevel 1 goto :fail

:have_venv
if exist ".venv\installed.txt" goto :have_packages
echo.
echo === Installing packages (a few minutes, only the first time) ===
"%VENV_PY%" -m pip install --disable-pip-version-check --quiet --upgrade pip
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :fail
echo installed> ".venv\installed.txt"

:have_packages
if not exist ".env" copy /y ".env.example" ".env" >nul
"%VENV_PY%" scripts\initialize_database.py
if errorlevel 1 goto :fail
if exist "data\.setup-complete" goto :start

echo.
echo === Create the Super Admin account (the platform owner who signs in first) ===
if defined SUPER_ADMIN_PASSWORD goto :admin_from_env
"%VENV_PY%" scripts\create_super_admin.py --if-missing
goto :admin_done
:admin_from_env
"%VENV_PY%" scripts\create_super_admin.py --if-missing --from-env
:admin_done
if errorlevel 1 goto :fail

echo.
set "DEMO=Y"
set /p "DEMO=Add a demo company with sample data so you can try the app right away? [Y/n] "
if /i "%DEMO%"=="n" goto :demo_done
if /i "%DEMO%"=="no" goto :demo_done
"%VENV_PY%" scripts\seed_demo_data.py
echo Write down the two demo passwords above. They are shown only once.
:demo_done
echo done> "data\.setup-complete"

:start
if /i "%~1"=="setup" goto :end
echo.
echo === Starting the app at http://localhost:8501 ===
echo Your browser opens in a few seconds. Keep this window open; close it or press Ctrl+C to stop.
start "" /min cmd /c "timeout /t 6 /nobreak >nul && start http://localhost:8501"
"%VENV_PY%" -m streamlit run app.py --server.port 8501
goto :end

:find_python
set "PYTHON="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PYTHON (
    py -%%V -c "import sys" >nul 2>nul && set "PYTHON=py -%%V"
  )
)
if defined PYTHON exit /b 0
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul && set "PYTHON=python"
exit /b 0

:no_python
echo.
echo Python 3.10, 3.11, 3.12 or 3.13 was not found (3.14 is not supported yet).
echo Install Python 3.13 from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" in the installer. Then run this file again.
goto :fail_pause

:fail
echo.
echo Setup stopped because of the error above. Fix it and run this file again.
echo (To start over completely, delete the .venv folder.)

:fail_pause
if not defined VDAS_NO_PAUSE pause
exit /b 1

:end
endlocal
