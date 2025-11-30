# Docker 启动脚本
# 使用方法: .\docker-start.ps1 [选项]

param(
    [string]$Mode = "watch",  # watch, up, build
    [string]$App = "streamlit_app.py"  # streamlit_app.py 或 arg_app.py
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Agent Service Toolkit - Docker 启动" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 设置环境变量
$env:STREAMLIT_APP = $App
Write-Host "使用应用: $App" -ForegroundColor Yellow
Write-Host ""

switch ($Mode.ToLower()) {
    "watch" {
        Write-Host "启动模式: 热重载模式 (推荐用于开发)" -ForegroundColor Green
        Write-Host "代码修改会自动触发服务重启" -ForegroundColor Green
        Write-Host ""
        Write-Host "按 Ctrl+C 停止服务" -ForegroundColor Yellow
        Write-Host ""
        docker compose watch
    }
    "up" {
        Write-Host "启动模式: 普通模式" -ForegroundColor Green
        Write-Host "代码修改需要手动重启服务" -ForegroundColor Yellow
        Write-Host ""
        docker compose up
    }
    "build" {
        Write-Host "启动模式: 重新构建并启动" -ForegroundColor Green
        Write-Host "这会重新构建所有 Docker 镜像" -ForegroundColor Yellow
        Write-Host ""
        docker compose up --build
    }
    default {
        Write-Host "未知模式: $Mode" -ForegroundColor Red
        Write-Host "可用模式: watch, up, build" -ForegroundColor Yellow
        exit 1
    }
}

