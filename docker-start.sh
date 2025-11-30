#!/bin/bash
# Docker 启动脚本 (Linux/Mac)
# 使用方法: ./docker-start.sh [模式] [应用]

MODE=${1:-watch}  # watch, up, build
APP=${2:-streamlit_app.py}  # streamlit_app.py 或 arg_app.py

echo "========================================"
echo "  Agent Service Toolkit - Docker 启动"
echo "========================================"
echo ""

# 设置环境变量
export STREAMLIT_APP=$APP
echo "使用应用: $APP"
echo ""

case $MODE in
    watch)
        echo "启动模式: 热重载模式 (推荐用于开发)"
        echo "代码修改会自动触发服务重启"
        echo ""
        echo "按 Ctrl+C 停止服务"
        echo ""
        docker compose watch
        ;;
    up)
        echo "启动模式: 普通模式"
        echo "代码修改需要手动重启服务"
        echo ""
        docker compose up
        ;;
    build)
        echo "启动模式: 重新构建并启动"
        echo "这会重新构建所有 Docker 镜像"
        echo ""
        docker compose up --build
        ;;
    *)
        echo "未知模式: $MODE"
        echo "可用模式: watch, up, build"
        exit 1
        ;;
esac

