# Docker 服务诊断脚本
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Docker 服务诊断" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查服务状态
Write-Host "1. 检查服务状态..." -ForegroundColor Yellow
docker compose ps
Write-Host ""

# 2. 检查 agent_service 日志
Write-Host "2. 检查 agent_service 最近日志..." -ForegroundColor Yellow
docker compose logs agent_service --tail 20
Write-Host ""

# 3. 测试 agent_service 连接
Write-Host "3. 测试 agent_service 连接..." -ForegroundColor Yellow
docker compose exec agent_service python -c "import httpx; r = httpx.get('http://localhost:8080/info', timeout=5); print(f'Status: {r.status_code}'); print(f'Response: {r.text[:200]}')" 2>&1
Write-Host ""

# 4. 测试从 streamlit_app 连接到 agent_service
Write-Host "4. 测试从 streamlit_app 连接到 agent_service..." -ForegroundColor Yellow
docker compose exec streamlit_app python -c "import httpx; import time; max_retries=3; for i in range(max_retries): try: r = httpx.get('http://agent_service:8080/info', timeout=5); print(f'Success: Status {r.status_code}'); break; except Exception as e: print(f'Attempt {i+1}/{max_retries}: {e}'); time.sleep(2) if i < max_retries-1 else None" 2>&1
Write-Host ""

# 5. 检查网络连接
Write-Host "5. 检查网络连接..." -ForegroundColor Yellow
docker compose exec streamlit_app ping -c 2 agent_service 2>&1 | Select-String -Pattern "timeout|unreachable|packets" -Context 0
Write-Host ""

# 6. 检查环境变量
Write-Host "6. 检查 streamlit_app 环境变量..." -ForegroundColor Yellow
docker compose exec streamlit_app env | Select-String -Pattern "AGENT_URL|STREAMLIT_APP" -Context 0
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  诊断完成" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

