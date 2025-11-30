# 使用本地开源 Embedding 模型

本项目支持使用本地开源 embedding 模型替代 OpenAI Embeddings，无需 API 密钥，完全本地运行。

## 支持的本地模型类型

### 1. HuggingFace Embeddings（推荐）

使用 HuggingFace 上的开源 embedding 模型。

**推荐模型：**
- `BAAI/bge-small-en-v1.5` - 英文，小模型，速度快
- `BAAI/bge-base-en-v1.5` - 英文，基础模型
- `BAAI/bge-large-en-v1.5` - 英文，大模型，效果好
- `BAAI/bge-small-zh-v1.5` - 中文，小模型
- `sentence-transformers/all-MiniLM-L6-v2` - 通用，轻量级

**安装依赖：**
```bash
pip install langchain-community sentence-transformers
```

### 2. Ollama Embeddings

如果本地运行 Ollama，可以使用 Ollama 的 embedding 模型。

**推荐模型：**
- `nomic-embed-text` - 通用文本嵌入
- `all-minilm` - 轻量级模型

**安装依赖：**
```bash
pip install langchain-community
```

**启动 Ollama：**
```bash
ollama serve
ollama pull nomic-embed-text
```

### 3. Sentence Transformers

直接使用 sentence-transformers 库的模型（实际上也是 HuggingFace）。

## 使用方法

### 方法 1: 使用本地脚本创建数据库

```bash
# 使用 HuggingFace 模型
python scripts/create_chroma_db_local.py \
    --model huggingface \
    --model-name BAAI/bge-small-en-v1.5 \
    --folder ./data

# 使用 Ollama 模型
python scripts/create_chroma_db_local.py \
    --model ollama \
    --model-name nomic-embed-text \
    --folder ./data

# 使用 SentenceTransformer 模型
python scripts/create_chroma_db_local.py \
    --model sentence-transformers \
    --model-name all-MiniLM-L6-v2 \
    --folder ./data
```

### 方法 2: 在代码中使用本地模型

#### 修改 `scripts/create_chroma_db.py`

```python
# 原来的代码
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

# 改为使用本地模型
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},  # 或 "cuda" 如果有 GPU
    encode_kwargs={"normalize_embeddings": True},
)
```

#### 修改 `src/agents/tools.py`

```python
# 原来的代码
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 改为使用本地模型
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

### 方法 3: 使用环境变量配置

创建 `.env` 文件：

```bash
# 使用本地模型
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_TYPE=huggingface
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# 如果使用 Ollama
# USE_LOCAL_EMBEDDINGS=true
# LOCAL_EMBEDDING_TYPE=ollama
# LOCAL_EMBEDDING_MODEL=nomic-embed-text
# OLLAMA_BASE_URL=http://localhost:11434
```

然后在代码中使用 `tools_local.py`：

```python
from agents.tools_local import database_search_local
```

## 模型对比

| 模型 | 大小 | 速度 | 质量 | 语言支持 |
|------|------|------|------|----------|
| `BAAI/bge-small-en-v1.5` | ~130MB | 快 | 好 | 英文 |
| `BAAI/bge-base-en-v1.5` | ~400MB | 中等 | 很好 | 英文 |
| `BAAI/bge-large-en-v1.5` | ~1.3GB | 慢 | 优秀 | 英文 |
| `all-MiniLM-L6-v2` | ~80MB | 很快 | 中等 | 英文 |
| `nomic-embed-text` (Ollama) | ~274MB | 中等 | 好 | 多语言 |

## 性能优化

### 使用 GPU（如果可用）

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},  # 使用 GPU
    encode_kwargs={"normalize_embeddings": True},
)
```

### 批量处理

本地模型支持批量处理，可以一次性处理多个文本：

```python
texts = ["text1", "text2", "text3"]
embeddings = embeddings.embed_documents(texts)  # 批量处理
```

## 注意事项

1. **首次使用**：模型会自动从 HuggingFace 下载，需要网络连接
2. **存储空间**：模型文件会保存在本地缓存目录（通常 `~/.cache/huggingface/`）
3. **内存要求**：小模型约需 1-2GB RAM，大模型可能需要 4GB+ RAM
4. **兼容性**：使用不同 embedding 模型创建的向量数据库不兼容，需要重新创建

## 完整示例

查看 `scripts/create_chroma_db_local.py` 获取完整的使用示例。

## 故障排除

### 问题：模型下载失败

**解决方案：**
- 检查网络连接
- 使用镜像站点：设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`
- 手动下载模型到本地

### 问题：内存不足

**解决方案：**
- 使用更小的模型（如 `bge-small`）
- 减少 batch_size
- 使用 CPU 而不是 GPU

### 问题：速度慢

**解决方案：**
- 使用 GPU（如果可用）
- 使用更小的模型
- 增加 batch_size 进行批量处理

