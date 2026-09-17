@echo off
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY="

REM Priorizar el entorno Conda usado durante el desarrollo.
if /I "%CONDA_DEFAULT_ENV%"=="tesis" (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
)
if not defined PY if exist "%USERPROFILE%\.conda\envs\tesis\python.exe" set "PY=%USERPROFILE%\.conda\envs\tesis\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\tesis\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\tesis\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\tesis\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\tesis\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\tesis\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\envs\tesis\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\envs\tesis\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\envs\tesis\python.exe"

REM Si no existe ese entorno, intentar Python 3.11 y luego el Python del PATH.
if not defined PY for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"

if not defined PY (
    echo.
    echo ============================================================
    echo NO SE ENCONTRO PYTHON
    echo ============================================================
    echo Instala Python 3.11 o crea/activa el entorno Conda "tesis".
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo SISTEMA INTEGRADO DE CONSTRUCCION 3D
echo ============================================================
echo Python: %PY%
echo.
echo Verificando dependencias y recursos del sistema...
echo.

REM El verificador muestra el estado e instala automaticamente lo que falte.
"%PY%" "%~dp0herramientas\00_verificar_dependencias.py"
if errorlevel 1 (
    echo.
    echo ============================================================
    echo NO FUE POSIBLE PREPARAR EL ENTORNO
    echo ============================================================
    echo Ejecuta VERIFICAR_SISTEMA.bat para revisar el detalle.
    pause
    exit /b 1
)

echo.
echo Verificacion completada correctamente.
echo Iniciando aplicacion...
echo.

"%PY%" "%~dp0Sistema_3D.py"
if errorlevel 1 pause
endlocal
