import json
import math
import re
import time
import httpx
import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings
import os

# 统一的向量数据库文件夹
VECTOR_DB_BASE_DIR = "./vector_databases"


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_embeddings():
    """Get the embeddings for the local model or OpenAi model."""
    global _embeddings_cache
    import logging
    logger = logging.getLogger(__name__)
    
    if _embeddings_cache is None:
        logger.debug("Embeddings cache is empty, creating new embeddings")
        with _embeddings_lock:
            # 双重检查锁定
            if _embeddings_cache is None:
                use_local_model = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"
                if use_local_model:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    catche_folder = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "embedding.model",
                    )
                    model_name = os.getenv("LOCAL_MODEL_NAME", "BAAI/bge-small-en-v1.5")
                    
                    # 设置离线模式环境变量
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    
                    try:
                        _embeddings_cache = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=catche_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                        logger.info(f"Embeddings initialized successfully: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
                        # 如果离线模式失败，尝试在线模式
                        logger.warning("Retrying without offline mode...")
                        os.environ.pop("HF_HUB_OFFLINE", None)
                        _embeddings_cache = HuggingFaceEmbeddings(
                            model_name=model_name,
                            cache_folder=catche_folder,
                            model_kwargs={"device": "cpu"},
                            encode_kwargs={"normalize_embeddings": True},
                        )
                else:
                    try:
                        _embeddings_cache = OpenAIEmbeddings()
                        logger.info("OpenAI embeddings initialized successfully")
                    except Exception as e:
                        raise RuntimeError(
                            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
                        ) from e
    else:
        logger.debug("Using cached embeddings")
    
    return _embeddings_cache
import threading
_vector_db_retriever = None
_vector_db_lock = threading.Lock()
_embeddings_cache = None  # 添加这行
_embeddings_lock = threading.Lock()  # 添加这行


import logging
def clear_retriever_cache():
    """
    清除向量数据库 retriever 缓存，并尝试关闭数据库连接
    """
    logger = logging.getLogger(__name__)
    global _vector_db_retriever
    with _vector_db_lock:
        # 尝试关闭数据库连接（如果存在）
        if _vector_db_retriever is not None:
            try:
                # 对于 Chroma，尝试关闭连接
                if hasattr(_vector_db_retriever, 'vectorstore'):
                    vectorstore = _vector_db_retriever.vectorstore
                    if hasattr(vectorstore, '_client'):
                        # Chroma 客户端
                        if hasattr(vectorstore._client, 'close'):
                            vectorstore._client.close()
                    elif hasattr(vectorstore, '_collection'):
                        # Qdrant 客户端
                        if hasattr(vectorstore._collection, '_client'):
                            client = vectorstore._collection._client
                            if hasattr(client, 'close'):
                                client.close()
            except Exception as e:
                logger.warning(f"关闭数据库连接时出错（可忽略）: {e}")
        
        # 清除缓存
        _vector_db_retriever = None
        logger.info("Vector database retriever cache cleared")
    


def _get_retriever():
    """
    获取缓存的 retriever，如果不存在则创建。
    使用双重检查锁定模式确保线程安全。
    """
    global _vector_db_retriever
    
    logger = logging.getLogger(__name__)
    
    if _vector_db_retriever is None:
        logger.debug("Retriever cache is empty, creating new retriever")
        with _vector_db_lock:
            # 双重检查：再次检查是否已被其他线程创建
            if _vector_db_retriever is None:
                try:
                    logger.info("Loading vector database...")
                    _vector_db_retriever = load_vector_db()
                    logger.info("Vector database loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load vector database: {e}", exc_info=True)
                    raise
    else:
        logger.debug("Using cached retriever")
    
    return _vector_db_retriever


def load_vector_db():
    """
    加载向量数据库（支持 Chroma 和 Qdrant）
    通过环境变量 VECTOR_DB_TYPE 选择数据库类型
    默认路径统一使用 vector_databases 文件夹
    """
    db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()  # 默认使用 Chroma
    embeddings = get_embeddings()
    
    # 确保统一文件夹存在
    os.makedirs(VECTOR_DB_BASE_DIR, exist_ok=True)
    
    if db_type == "qdrant":
        # 使用 Qdrant 本地嵌入式模式
        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "需要安装 qdrant-client 和 langchain-qdrant: "
                "pip install qdrant-client langchain-qdrant"
            )
        
        # Qdrant 本地嵌入式模式
        # 如果环境变量未设置，使用统一文件夹下的默认路径
        default_qdrant_path = os.path.join(VECTOR_DB_BASE_DIR, "qdrant_db")
        qdrant_path = os.getenv("QDRANT_PATH", default_qdrant_path)
        collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        
        # 创建本地 Qdrant 客户端
        client = QdrantClient(path=qdrant_path)  # 本地嵌入式模式
        
        # 获取 embedding 维度
        embedding_dim = len(embeddings.embed_query("test"))
        
        # 确保集合存在
        try:
            client.get_collection(collection_name)
        except Exception:
            # 集合不存在，创建它
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
        
        # 创建 Qdrant vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    
    else:
        # 默认使用 ChromaDB
        # 如果环境变量未设置，使用统一文件夹下的默认路径
        default_chroma_path = os.path.join(VECTOR_DB_BASE_DIR, "chroma_db")
        db_path = os.getenv("CHROMA_DB_PATH", default_chroma_path)
        chroma_db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
        return retriever





def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"Starting database search for query: {query}")
        
        # 使用缓存的 retriever，避免每次都重新创建数据库连接
        retriever = _get_retriever()
        logger.debug("Retriever obtained successfully")
        
        # Search the database for relevant documents
        documents = retriever.invoke(query)
        logger.debug(f"Search completed, found {len(documents) if documents else 0} documents")
        
        if not documents:
            # 返回明确信息，告诉 Agent 搜索已完成但没有找到相关内容
            logger.warning(f"No documents found for query: {query}")
            return "No relevant documents found in the database for this query. The database search completed successfully, but no matching content was retrieved."
        
        # Format the documents into a string
        context_str = format_contexts(documents)
        logger.debug(f"Formatted context length: {len(context_str)} characters")
        
        return context_str
    except Exception as e:
        # 记录详细错误信息
        logger.error(f"Database search error for query '{query}': {e}", exc_info=True)
        # 返回错误信息，让 Agent 知道发生了什么
        error_msg = f"Database search error: {str(e)}"
        logger.error(f"Returning error message to agent: {error_msg}")
        return error_msg


database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"  # Update name with the purpose of your database


def openreview_search_func(
    venue: str = "ICML 2025 oral",
    domain: str = "ICML.cc/2025/Conference",
    invitation: str = "ICML.cc/2025/Conference/-/Submission",
    max_papers: int = 500,
    limit_per_page: int = 25,
) -> str:
    """Searches OpenReview API for academic papers.

    Useful for when you need to find papers from conferences like ICML, NeurIPS, ICLR, etc.
    This tool fetches papers from OpenReview API with pagination support.

    Args:
        venue (str): The venue filter (e.g., "ICML 2025 oral", "NeurIPS 2024"). Default: "ICML 2025 oral"
        domain (str): The conference domain. Default: "ICML.cc/2025/Conference"
        invitation (str): The submission invitation path. Default: "ICML.cc/2025/Conference/-/Submission"
        max_papers (int): Maximum number of papers to fetch. Default: 500
        limit_per_page (int): Number of papers per page. Default: 25

    Returns:
        str: JSON string containing the list of papers with their details.
    """
    base_url = "https://api2.openreview.net/notes"
    all_notes = []

    try:
        for offset in range(0, max_papers, limit_per_page):
            params = {
                "content.venue": venue,
                "details": "replyCount,presentation,writable",
                "domain": domain,
                "invitation": invitation,
                "limit": limit_per_page,
                "offset": offset,
            }

            response = httpx.get(base_url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            if not data.get("notes"):
                break

            all_notes.extend(data["notes"])

            # Rate limiting: sleep between requests
            if offset + limit_per_page < max_papers:
                time.sleep(1)

        # Format the results
        result = {
            "total_papers": len(all_notes),
            "papers": [
                {
                    "id": note.get("id"),
                    "title": note.get("content", {}).get("title", "N/A"),
                    "abstract": note.get("content", {}).get("abstract", "N/A"),
                    "authors": note.get("content", {}).get("authors", []),
                    "venue": note.get("content", {}).get("venue", "N/A"),
                    "keywords": note.get("content", {}).get("keywords", []),
                }
                for note in all_notes
            ],
        }

        return json.dumps(result, indent=2, ensure_ascii=False)

    except httpx.HTTPError as e:
        raise ValueError(f"Failed to fetch papers from OpenReview: {e}")
    except Exception as e:
        raise ValueError(f"Error processing OpenReview data: {e}")


openreview_search: BaseTool = tool(openreview_search_func)
openreview_search.name = "OpenReview_Search"


