"""
使用本地开源 Embedding 模型创建 Qdrant 向量数据库

支持本地嵌入式 Qdrant，无需单独运行 Qdrant 服务。
"""
import argparse
import os
import shutil
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def get_local_embeddings(model_type: str, model_name: str) -> Any:
    """获取本地 embedding 模型"""
    cache_folder = os.path.join(os.getcwd(), "embedding.model")
    
    if model_type == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            print(f"使用 HuggingFace 模型: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            raise ImportError(
                "需要安装 langchain-community: pip install langchain-community"
            )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def create_qdrant_db(
    folder_path: str,
    db_path: str = "./qdrant_db",
    collection_name: str = "documents",
    delete_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
    model_type: str = "huggingface",
    model_name: str = "BAAI/bge-small-en-v1.5",
):
    """
    使用本地 embedding 模型创建 Qdrant 数据库
    """
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        raise ImportError(
            "需要安装 qdrant-client 和 langchain-qdrant: "
            "pip install qdrant-client langchain-qdrant"
        )

    # 获取本地 embedding 模型
    embeddings = get_local_embeddings(model_type, model_name)

    # 创建本地 Qdrant 客户端（嵌入式模式）
    if delete_db and os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"已删除现有数据库: {db_path}")

    client = QdrantClient(path=db_path)  # 本地嵌入式模式

    # 获取 embedding 维度
    test_embedding = embeddings.embed_query("test")
    embedding_dim = len(test_embedding)

    # 创建集合
    try:
        client.get_collection(collection_name)
        print(f"集合 '{collection_name}' 已存在，将添加新文档")
    except Exception:
        print(f"创建新集合: {collection_name}")
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

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    # 导入文档加载器
    try:
        from langchain_community.document_loaders import (
            Docx2txtLoader,
            PyPDFLoader,
            TextLoader,
        )
    except ImportError:
        raise ImportError("需要安装 langchain-community: pip install langchain-community")

    # 处理文档
    total_chunks = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 根据文件扩展名加载文档
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"跳过不支持的文件类型: {filename}")
            continue

        # 加载并分割文档
        print(f"\n处理文件: {filename}")
        document = loader.load()
        chunks = text_splitter.split_documents(document)
        print(f"  分割为 {len(chunks)} 个文本块")

        # 添加到 Qdrant
        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        print(f"  ✅ 文档 {filename} 已添加到数据库")

    print(f"\n✅ Qdrant 向量数据库创建完成!")
    print(f"  位置: {db_path}")
    print(f"  集合名: {collection_name}")
    print(f"  总文本块数: {total_chunks}")
    print(f"  使用的模型: {model_name} ({model_type})")
    
    return vector_store


def main():
    parser = argparse.ArgumentParser(description="使用本地 embedding 模型创建 Qdrant 数据库")
    parser.add_argument(
        "--folder",
        type=str,
        default="./data",
        help="包含文档的文件夹路径 (默认: ./data)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./qdrant_db",
        help="数据库路径 (默认: ./qdrant_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="集合名称 (默认: documents)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["huggingface"],
        default="huggingface",
        help="模型类型 (默认: huggingface)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="模型名称 (默认: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="文本块大小 (默认: 2000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=500,
        help="文本块重叠大小 (默认: 500)",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="保留已存在的数据库（不删除）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("使用本地 Embedding 模型创建 Qdrant 数据库")
    print("=" * 80)
    print(f"文档文件夹: {args.folder}")
    print(f"数据库路径: {args.db_path}")
    print(f"集合名称: {args.collection}")
    print(f"模型类型: {args.model}")
    print(f"模型名称: {args.model_name}")
    print("=" * 80)

    try:
        vector_store = create_qdrant_db(
            folder_path=args.folder,
            db_path=args.db_path,
            collection_name=args.collection,
            delete_db=not args.keep_existing,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_type=args.model,
            model_name=args.model_name,
        )

        # 测试检索
        print("\n" + "=" * 80)
        print("测试检索功能")
        print("=" * 80)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        query = "What's the company's mission"
        print(f"查询: {query}")
        similar_docs = retriever.invoke(query)

        print(f"\n找到 {len(similar_docs)} 个相关文档:")
        for i, doc in enumerate(similar_docs, start=1):
            print(f"\n结果 {i}:")
            print(f"  内容: {doc.page_content[:200]}...")
            if hasattr(doc, "metadata"):
                print(f"  元数据: {doc.metadata}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
