"""
向量数据库管理模块
===================
使用 TF-IDF + 文件存储实现向量检索，纯 Python 依赖，离线可用。
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """向量数据库管理器，基于 TF-IDF + 文件存储。"""

    def __init__(self, persist_dir: str):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # 文件路径
        self.vectorizer_path = os.path.join(persist_dir, "tfidf_vectorizer.pkl")
        self.matrix_path = os.path.join(persist_dir, "tfidf_matrix.npy")
        self.docs_path = os.path.join(persist_dir, "documents.json")

        # 初始化 TF-IDF 向量化器
        print("🧠 初始化 TF-IDF 向量化器（离线，中文 n-gram）")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=5000,
            sublinear_tf=True,
        )
        self._is_fitted = False
        self._doc_matrix = None
        self._documents = []

        # 尝试加载已有数据
        self._load()

    def _load(self):
        """加载已保存的向量数据。"""
        if os.path.exists(self.vectorizer_path) and os.path.exists(self.matrix_path) and os.path.exists(self.docs_path):
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self._doc_matrix = np.load(self.matrix_path)
                with open(self.docs_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
                self._is_fitted = True
                print(f"✅ 加载已有向量数据，共 {len(self._documents)} 个文档块")
            except Exception as e:
                print(f"⚠️ 加载失败，将重新构建: {e}")

    def _save(self):
        """保存向量数据到文件。"""
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        np.save(self.matrix_path, self._doc_matrix)
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False)
        print(f"💾 已保存 {len(self._documents)} 个文档块")

    def add_documents(self, chunks: List[Dict[str, str]]):
        """添加文档块到向量存储。"""
        if not chunks:
            print("⚠️ 没有文档需要添加")
            return

        texts = [c["text"] for c in chunks]

        # 训练并生成向量
        print(f"🔧 训练 TF-IDF 向量化器，{len(texts)} 个文本...")
        self._doc_matrix = self.vectorizer.fit_transform(texts)
        self._is_fitted = True
        self._documents = chunks

        self._save()
        print(f"✅ 已添加 {len(chunks)} 个文档块")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索最相关的文档片段。"""
        if not self._is_fitted or self._doc_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._doc_matrix).flatten()

        top_k = min(top_k, len(self._documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = self._documents[idx]
                results.append({
                    "text": doc["text"],
                    "source": doc.get("source", ""),
                    "score": float(similarities[idx])
                })

        return results

    def count(self) -> int:
        """返回文档块数量。"""
        return len(self._documents)

    def clear(self):
        """清空向量数据。"""
        self._doc_matrix = None
        self._documents = []
        self._is_fitted = False
        for path in [self.vectorizer_path, self.matrix_path, self.docs_path]:
            if os.path.exists(path):
                os.remove(path)
        print("🗑️ 已清空向量数据库")