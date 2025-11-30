# Docker 启动指南

## 快速启动

### 方式 1: 使用热重载模式（推荐，开发时使用）

```bash
# Windows PowerShell
docker compose watch

# 或使用启动脚本
.\docker-start.ps1 watch
```

### 方式 2: 普通启动模式

```bash
# Windows PowerShell
docker compose up

# 或使用启动脚本
.\docker-start.ps1 up
```

### 方式 3: 重新构建并启动

```bash
# Windows PowerShell
docker compose up --build

# 或使用启动脚本
.\docker-start.ps1 build
```

## 选择要运行的应用

### 使用默认的 streamlit_app.py

```bash
docker compose watch
```

### 使用 arg_app.py（带文件上传功能）

```bash
# 方式 1: 设置环境变量
$env:STREAMLIT_APP="arg_app.py"
docker compose watch

# 方式 2: 使用启动脚本
.\docker-start.ps1 watch arg_app.py
```

## 服务访问地址

启动成功后，可以通过以下地址访问：

- **Streamlit 应用**: http://localhost:8501
- **Agent Service API**: http://localhost:8080
- **API 文档 (ReDoc)**: http://localhost:8080/redoc
- **API 文档 (Swagger)**: http://localhost:8080/docs

## 常用命令

### 查看服务状态

```bash
docker compose ps
```

### 查看日志

```bash
# 查看所有服务日志
docker compose logs

# 查看特定服务日志
docker compose logs agent_service
docker compose logs streamlit_app

# 实时查看日志
docker compose logs -f streamlit_app
```

### 停止服务

```bash
docker compose down
```

### 停止并删除数据卷

```bash
docker compose down -v
```

### 重启特定服务

```bash
docker compose restart agent_service
docker compose restart streamlit_app
```

## 热重载说明

使用 `docker compose watch` 时，以下文件变化会自动触发服务重启：

### agent_service 自动重载
- `src/agents/` 目录下的文件
- `src/schema/` 目录下的文件
- `src/service/` 目录下的文件
- `src/core/` 目录下的文件
- `src/memory/` 目录下的文件

### streamlit_app 自动重载
- `src/client/` 目录下的文件
- `src/schema/` 目录下的文件
- `src/streamlit_app.py`
- `src/arg_app.py`

## 故障排除

### 服务无法启动

1. 检查端口是否被占用：
```bash
# Windows
netstat -ano | findstr :8080
netstat -ano | findstr :8501
```

2. 检查 Docker 是否运行：
```bash
docker ps
```

3. 查看详细错误日志：
```bash
docker compose logs
```

### 健康检查失败

如果服务显示 "unhealthy"，等待几秒钟让服务完全启动。如果持续失败：

1. 检查服务日志：
```bash
docker compose logs agent_service
docker compose logs streamlit_app
```

2. 检查环境变量配置（`.env` 文件）

3. 重新构建镜像：
```bash
docker compose up --build
```

### 修改依赖后需要重建

如果修改了 `pyproject.toml` 或 `uv.lock`，需要重新构建：

```bash
docker compose up --build
```

## 环境变量配置

在 `.env` 文件中配置以下变量：

```env
# LLM API Keys (至少需要一个)
OPENAI_API_KEY=your_openai_api_key
# GROQ_API_KEY=your_groq_api_key  # 可选，用于 LlamaGuard

# 数据库配置
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=agent_service

# 向量数据库配置
VECTOR_DB_TYPE=chroma  # 或 qdrant
CHROMA_DB_PATH=./chroma_db_mixed
USE_LOCAL_MODEL=False  # 是否使用本地 embedding 模型
LOCAL_MODEL_NAME=BAAI/bge-small-en-v1.5

# Streamlit 应用选择
STREAMLIT_APP=streamlit_app.py  # 或 arg_app.py
```

## 开发建议

1. **开发时使用 `watch` 模式**：代码修改会自动重载
2. **生产环境使用 `up` 模式**：更稳定，不会自动重载
3. **定期清理未使用的镜像**：`docker system prune -a`
4. **查看资源使用情况**：`docker stats`

