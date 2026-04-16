@echo off
chcp 65001 > nul
echo ====================================
echo   外网访问设置
echo ====================================
echo.

REM 检查ngrok
where ngrok > nul 2>&1
if %errorlevel%==0 (
    echo [OK] ngrok 已安装
) else (
    echo [下载] 正在下载 ngrok...
    powershell -Command "Invoke-WebRequest -Uri 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip' -OutFile 'ngrok.zip'"
    powershell -Command "Expand-Archive -Path 'ngrok.zip' -DestinationPath '.' -Force"
    del ngrok.zip
    echo [OK] ngrok 已下载
)

echo.
echo 请选择隧道服务:
echo   1. ngrok (需要注册 https://ngrok.com)
echo   2. Cloudflare Tunnel (免费，无需注册)
echo.

set /p choice=请选择 (1 或 2):

if "%choice%"=="1" (
    echo.
    echo 请到 https://ngrok.com 注册并获取 authtoken
    echo 然后运行: ngrok config add-authtoken YOUR_TOKEN
    echo.
    echo 配置完成后，运行:
    echo   ngrok http 8502
    echo.
    pause
) else if "%choice%"=="2" (
    echo.
    echo 正在下载 cloudflared...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
    echo.
    echo [OK] cloudflared 已下载
    echo.
    echo 运行以下命令启动隧道:
    echo   cloudflared.exe tunnel --url http://localhost:8502
    echo.
    echo 或使用 --url flag 直接启动:
    cloudflared.exe tunnel --url http://localhost:8502
)

pause
