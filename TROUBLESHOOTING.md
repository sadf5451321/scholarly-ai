# 故障排除指南

## 常见错误及解决方案

### 1. Internal Server Error - API 密钥问题

**错误信息**：
```
Error code: 400 - Access denied, please make sure your account is in good standing
```

**原因**：
- API 密钥无效或过期
- 账户欠费或状态异常
- API 密钥配置错误

**解决方案**：

1. **检查 API 密钥配置**：
   - 确保 `.env` 文件中配置了至少一个有效的 LLM API 密钥
   - 支持的提供商：
     - `OPENAI_API_KEY` - OpenAI
     - `DEEPSEEK_API_KEY` - DeepSeek
     - `ANTHROPIC_API_KEY` - Anthropic (Claude)
     - `GOOGLE_API_KEY` - Google Gemini
     - `GROQ_API_KEY` - Groq
     - `COMPATIBLE_API_KEY` + `COMPATIBLE_BASE_URL` - OpenAI 兼容 API（如阿里云）

2. **检查账户状态**：
   - 登录对应服务商的控制台
   - 确认账户余额充足
   - 检查 API 使用限制

3. **验证 API 密钥**：
   ```bash
   # 测试 OpenAI API
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   
   # 测试其他 API（根据你的配置）
   ```

4. **使用其他模型提供商**：
   如果当前 API 有问题，可以切换到其他提供商：
   ```env
   # 在 .env 文件中添加其他 API 密钥
   OPENAI_API_KEY=your_openai_key
   # 或
   DEEPSEEK_API_KEY=your_deepseek_key
   ```

### 2. ValidationError - run_id 为 None

**错误信息**：
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Feedback
run_id
  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
```

**原因**：
- 消息没有 `run_id` 字段
- 反馈功能尝试使用 `None` 作为 `run_id`

**解决方案**：
- ✅ 已修复：代码现在会检查 `run_id` 是否存在，如果为 `None` 则跳过反馈功能
- 如果问题仍然存在，刷新页面或重启服务

### 3. Connection Refused - 服务连接问题

**错误信息**：
```
Error connecting to agent service at http://agent_service:8080: Connection refused
```

**解决方案**：

1. **检查服务状态**：
   ```bash
   docker compose ps
   ```

2. **检查服务日志**：
   ```bash
   docker compose logs agent_service
   docker compose logs streamlit_app
   ```

3. **重启服务**：
   ```bash
   docker compose restart agent_service
   docker compose restart streamlit_app
   ```

4. **检查环境变量**：
   - 确保 `HOST=0.0.0.0` 在 `compose.yaml` 中配置
   - 确保 `AGENT_URL=http://agent_service:8080` 在 streamlit_app 中配置

### 4. 健康检查失败

**症状**：
- 服务显示 `(unhealthy)` 状态

**解决方案**：

1. **等待服务启动**：
   - 服务启动需要时间，等待 30-60 秒

2. **检查健康检查配置**：
   - 健康检查使用 Python 的 `httpx`，确保服务可以响应

3. **手动测试**：
   ```bash
   # 测试 agent_service
   docker compose exec agent_service python -c "import httpx; httpx.get('http://localhost:8080/info', timeout=5).raise_for_status()"
   
   # 测试 streamlit_app
   docker compose exec streamlit_app python -c "import httpx; httpx.get('http://localhost:8501/healthz', timeout=5).raise_for_status()"
   ```

### 5. 数据库连接问题

**错误信息**：
- PostgreSQL 连接失败
- 数据库初始化错误

**解决方案**：

1. **检查 PostgreSQL 状态**：
   ```bash
   docker compose ps postgres
   docker compose logs postgres
   ```

2. **检查环境变量**：
   ```env
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_DB=agent_service
   ```

3. **重启数据库**：
   ```bash
   docker compose restart postgres
   ```

### 6. 向量数据库问题

**错误信息**：
- "无法访问数据库"
- "No documents found"

**解决方案**：

1. **检查数据库路径**：
   ```env
   CHROMA_DB_PATH=./chroma_db_mixed
   VECTOR_DB_TYPE=chroma
   ```

2. **创建数据库**：
   ```bash
   python scripts/create_chroma_db.py
   # 或使用本地模型
   python scripts/create_chroma_db_local.py
   ```

3. **检查文件权限**：
   - 确保数据库目录可读写

## 诊断工具

### 使用诊断脚本

```powershell
# Windows
.\docker-diagnose.ps1

# Linux/Mac
./docker-diagnose.sh
```

### 手动诊断步骤

1. **检查服务状态**：
   ```bash
   docker compose ps
   ```

2. **查看日志**：
   ```bash
   docker compose logs --tail 50
   ```

3. **测试连接**：
   ```bash
   # 测试 agent_service
   curl http://localhost:8080/info
   
   # 测试 streamlit_app
   curl http://localhost:8501/healthz
   ```

4. **检查环境变量**：
   ```bash
   docker compose exec agent_service env | grep -E "API_KEY|HOST|PORT"
   ```

## 获取帮助

如果问题仍然存在：

1. **查看完整日志**：
   ```bash
   docker compose logs > logs.txt
   ```

2. **检查配置文件**：
   - `compose.yaml`
   - `.env`
   - `src/core/settings.py`

3. **重新构建服务**：
   ```bash
   docker compose down
   docker compose up --build
   ```

## 预防措施

1. **定期检查 API 密钥**：
   - 确保 API 密钥有效
   - 监控 API 使用量和余额

2. **使用健康检查**：
   - 服务会自动检查健康状态
   - 关注服务状态变化

3. **备份配置**：
   - 保存 `.env` 文件备份
   - 记录重要的配置更改

4. **监控日志**：
   - 定期查看服务日志
   - 及时发现潜在问题

