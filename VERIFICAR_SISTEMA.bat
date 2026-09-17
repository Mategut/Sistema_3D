@echo off
REM Comprueba el entorno y la calibracion estereo activa.
setlocal
REM Resolver las rutas desde la carpeta que contiene este lanzador.
cd /d "%~dp0"
REM Mantener codificacion UTF-8 en Python y su salida de consola.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY="

REM Prioriza el entorno Conda usado durante el desarrollo.
if /I "%CONDA_DEFAULT_ENV%"=="tesis" (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
)
if not defined PY if exist "%USERPROFILE%\.conda\envs\tesis\python.exe" set "PY=%USERPROFILE%\.conda\envs\tesis\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\tesis\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\tesis\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\tesis\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\tesis\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\envs\tesis\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\envs\tesis\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\envs\tesis\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\envs\tesis\python.exe"

REM Si no existe ese entorno, intenta Python 3.11 o el Python disponible en PATH.
if not defined PY for /f "delims=" %%P in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"

if not defined PY (
    echo No se encontro una instalacion de Python.
    echo Instala Python 3.11 y vuelve a ejecutar este archivo.
    pause
    exit /b 1
)

echo Python: %PY%
echo.
"%PY%" "%~dp0herramientas\00_verificar_dependencias.py"
if errorlevel 1 (
    echo.
    echo No fue posible completar la configuracion automaticamente.
    pause
    exit /b 1
)

echo.
echo Verificando calibracion estereo activa...
"%PY%" "%~dp0herramientas\02_verificar_calibracion_estereo.py"
REM Conservar el resultado de la comprobacion antes de pausar la consola.
set RC=%ERRORLEVEL%
pause
exit /b %RC%
