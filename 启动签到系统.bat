@echo off
chcp 65001 > nul
title 人脸签到系统

echo ====================================
echo   人脸签到系统启动器
echo ====================================
echo.

REM 检查Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 启动 Streamlit
echo [1/2] 启动 Streamlit 服务...
start "Streamlit" python -X utf8 -m streamlit run scripts/app.py --server.headless true --server.port 8502

REM 等待Streamlit启动
timeout /t 5 /nobreak > nul

REM 启动 ngrok（如果已配置）
if exist "ngrok.exe" (
    echo [2/2] 启动 ngrok 隧道...
    start "ngrok" ngrok http 8502
    timeout /t 5 /nobreak > nul
    echo.
    echo ====================================
    echo   服务已启动！
    echo ====================================
    echo.
    echo 访问本地: http://localhost:8502
    echo 查看隧道: http://localhost:4040
) else (
    echo.
    echo ====================================
    echo   服务已启动！
    echo ====================================
    echo.
    echo 访问本地: http://localhost:8502
    echo.
    echo [提示] 如需外网访问，请下载 ngrok:
    echo   https://ngrok.com/download
    echo   然后运行: ngrok http 8502
)

echo.
pause
