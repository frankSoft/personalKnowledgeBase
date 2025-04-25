'''
向量数据库获取和管理模块
提供向量数据库的创建、加载和初始化功能
支持基于文件路径初始化向量数据库
自动处理新建和已有向量库的不同情况
'''
import sys  # 用于操作 Python 运行环境
# sys.path.append("../embedding")  # 可选：添加 embedding 目录到模块搜索路径
# sys.path.append("../database")   # 可选：添加 database 目录到模块搜索路径

from langchain_community.embeddings import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import os  # 用于文件和目录操作
from embedding.zhipuai_embedding import ZhipuAIEmbeddings   # 导入智谱AI的 Embeddings 模型
from database.create_db import create_db, load_knowledge_db # 导入创建和加载向量数据库的函数
from embedding.call_embedding import get_embedding          # 导入 embedding 获取函数

def get_vectordb(file_path: str = None, persist_path: str = None, embedding="openai", embedding_key: str = None):
    """
    返回向量数据库对象
    输入参数：
    file_path: 知识库原始文件路径
    persist_path: 向量数据库持久化目录
    embedding: 指定使用的 embedding 模型（如 openai、zhipuai、m3e 等），默认 openai
    embedding_key: embedding 模型的 API key（如有需要）
    返回：
    vectordb: 向量数据库对象，可用于检索
    """
    try:
        # 确保持久化路径存在
        if persist_path:
            os.makedirs(persist_path, exist_ok=True)
        
        # 获取 embedding 对象（根据 embedding 类型和 key）
        print("get_vectordb 获取 embedding 对象（根据 embedding 类型和 key）：", embedding)
        embedding_model = get_embedding(embedding=embedding, embedding_key=embedding_key)
        
        if os.path.exists(persist_path):  # 如果持久化目录存在
            contents = os.listdir(persist_path)  # 获取目录下所有文件
            if len(contents) == 0:  # 如果目录为空，说明还未建立向量库
                # print("目录为空")
                if file_path:
                    vectordb = create_db(file_path, persist_path, embedding_model)  # 创建新的向量数据库
                    if isinstance(vectordb, str):  # 如果返回的是错误信息字符串
                        raise Exception(vectordb)
                    # presit_knowledge_db(vectordb)  # 可选：持久化数据库
                    vectordb = load_knowledge_db(persist_path, embedding_model)  # 加载刚创建的数据库
                else:
                    raise Exception("文件路径为空，无法创建向量数据库")
            else:
                # print("目录不为空")
                vectordb = load_knowledge_db(persist_path, embedding_model)  # 直接加载已有数据库
        else:  # 如果目录不存在，从头开始创建向量数据库
            if file_path:
                vectordb = create_db(file_path, persist_path, embedding_model)  # 创建新的向量数据库
                if isinstance(vectordb, str):  # 如果返回的是错误信息字符串
                    raise Exception(vectordb) 
                # presit_knowledge_db(vectordb)  # 可选：持久化数据库
                vectordb = load_knowledge_db(persist_path, embedding_model)  # 加载刚创建的数据库
            else:
                raise Exception("文件路径为空，无法创建向量数据库")

        # print("get_vectordb 获取向量数据库成功")
        return vectordb  # 返回向量数据库对象
    except Exception as e:
        print(f"获取向量数据库失败: {str(e)}")
        raise e  # 重新抛出异常，让上层处理