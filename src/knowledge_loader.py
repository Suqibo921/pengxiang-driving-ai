"""
知识库加载与文本切片模块
=========================
负责读取知识库 Markdown 文件，进行文本切片。
"""

import os
import glob
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_knowledge_files(knowledge_base_dir: str) -> List[Dict[str, str]]:
    """
    加载知识库目录下的所有 Markdown 文件。

    参数:
        knowledge_base_dir: 知识库目录路径

    返回:
        包含文件名和内容的字典列表
    """
    documents = []
    md_files = glob.glob(os.path.join(knowledge_base_dir, "*.md"))

    if not md_files:
        print(f"⚠️ 未找到知识库文件，请确认目录: {knowledge_base_dir}")
        return documents

    # 按文件名排序，保证加载顺序一致
    md_files.sort()

    for file_path in md_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                documents.append({
                    "filename": filename,
                    "content": content,
                    "filepath": file_path
                })
                print(f"  ✅ 加载: {filename}")
            else:
                print(f"  ⚠️ 跳过空文件: {filename}")
        except Exception as e:
            print(f"  ❌ 读取失败: {filename} - {e}")

    print(f"\n📚 共加载 {len(documents)} 个知识库文件")
    return documents


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, str]]:
    """
    将文档切分为小块。

    参数:
        documents: 文档列表（含 filename 和 content）
        chunk_size: 每个切片的最大字符数
        chunk_overlap: 切片之间的重叠字符数

    返回:
        包含文本内容和来源文件名的切片列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "\n## ", "\n# ", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(doc_chunks):
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": doc["filename"],
                    "chunk_id": i
                })

    print(f"📝 共切分为 {len(chunks)} 个文本块")
    return chunks