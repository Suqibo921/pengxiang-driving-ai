"""
向量数据库管理模块
===================
使用 ChromaDB 存储和检索文档向量。
Embedding 使用 sklearn 的 TF-IDF（离线可用，无需下载模型）。
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer


class VectorStore:
    """向量数据库管理器，封装 ChromaDB 的增删查操作。"""

    def __init__(self, persist_dir: str, model_name: str = "tfidf"):
        """
        初始化向量数据库。

        参数:
            persist_dir: 向量数据库持久化目录
            model_name: 模型名称（保留参数，兼容性）
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(path=persist_dir)

        # 初始化 TF-IDF 向量化器（完全离线，无需下载）
        # 使用字符级别的 n-gram，适合中文场景
        print("🧠 初始化 TF-IDF 向量化器（离线，中文 n-gram）")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True,
        )
        self._is_fitted = False
        self._dimension = 5000

        # 尝试加载已训练的向量化器
        vectorizer_path = os.path.join(persist_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            try:
                with open(vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self._is_fitted = True
                self._dimension = len(self.vectorizer.get_feature_names_out())
                print(f"✅ 加载已训练的向量化器，维度: {self._dimension}")
            except Exception:
                print("⚠️ 向量化器加载失败，将重新训练")

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="pengxiang_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def _fit_vectorizer(self, texts: List[str]):
        """训练向量化器。"""
        print("🔧 训练 TF-IDF 向量化器...")
        self.vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self.vectorizer.get_feature_names_out())
        print(f"✅ 向量化器训练完成，维度: {self._dimension}")

        # 保存向量化器
        vectorizer_path = os.path.join(self.persist_dir, "tfidf_vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"💾 向量化器已保存至: {vectorizer_path}")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示。"""
        if not self._is_fitted:
            raise RuntimeError("向量化器尚未训练，请先添加文档")
        vec = self.vectorizer.transform([text])
        # L2 归一化
        vec_norm = vec.toarray()[0]
        norm = np.linalg.norm(vec_norm)
        if norm > 0:
            vec_norm = vec_norm / norm
        return vec_norm.tolist()

    def add_documents(self, chunks: List[Dict[str, str]]):
        """
        将文档切片添加到向量数据库。

        参数:
            chunks: 文档切片列表，每项包含 text, source, chunk_id
        """
        if not chunks:
            print("⚠️ 没有文档需要添加")
            return

        texts = [c["text"] for c in chunks]
        sources = [c["source"] for c in chunks]
        ids = [f"{c['source']}_{c['chunk_id']}" for c in chunks]

        # 训练向量化器
        self._fit_vectorizer(texts)

        # 生成 embeddings
        print(f"🔢 生成 {len(texts)} 个文本的向量...")
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            embeddings.append(emb)

        # 添加到 ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": s} for s in sources],
            ids=ids
        )
        print(f"✅ 已添加 {len(chunks)} 个文档块到向量数据库")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        搜索最相关的文档片段。

        参数:
            query: 搜索查询
            top_k: 返回结果数量

        返回:
            相关文档片段列表，每项包含 text, source, score
        """
        query_embedding = self.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                documents.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"] if results["metadatas"] else "",
                    "score": results["distances"][0][i] if results["distances"] else 0
                })

        return documents

    def count(self) -> int:
        """返回向量数据库中的文档块数量。"""
        return self.collection.count()

    def clear(self):
        """清空向量数据库。"""
        self.client.delete_collection("pengxiang_knowledge")
        self.collection = self.client.get_or_create_collection(
            name="pengxiang_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self._is_fitted = False
        print("🗑️ 已清空向量数据库")