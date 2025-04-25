'''
Embedding模型调用统一接口模块
===============================
本模块旨在提供一个统一的接口，用于调用和管理多种不同的文本嵌入（Embedding）模型。
目前支持的模型包括：
- OpenAI 的 text-embedding-ada-002 模型
- Hugging Face 上的 moka-ai/m3e-base 模型
- 智谱AI (ZhipuAI) 的 Embedding 模型
主要功能：
- 封装不同 Embedding 模型的初始化和调用逻辑。
- 提供 `get_embedding` 函数作为统一入口，根据指定的模型名称返回相应的 Embedding 对象实例。
- 自动处理 API 密钥的获取（通过参数传递或从环境变量/配置文件加载）。
'''
import os  # 导入os模块，用于路径处理
import sys  # 导入sys模块，用于操作解释器相关功能
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 将项目根目录加入模块搜索路径，便于导入自定义模块

from embedding.zhipuai_embedding import ZhipuAIEmbeddings  # 导入智谱AI的Embedding实现
# 这些导入可能需要被替换，因为需要自定义实现
# from langchain_community.embeddings import HuggingFaceEmbeddings  # 导入HuggingFace的Embedding实现
# from langchain_community.embeddings import OpenAIEmbeddings  # 导入OpenAI的Embedding实现
from llm.call_llm import parse_llm_api_key

# 需要自定义实现的OpenAIEmbeddings类
class OpenAIEmbeddings:
    """
    OpenAI Embeddings 模型的简单封装
    """
    def __init__(self, openai_api_key=None):
        import openai
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        
    def embed_documents(self, texts):
        """批量文本向量化"""
        import openai
        # 调用OpenAI的embeddings接口
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return [data["embedding"] for data in response["data"]]
        
    def embed_query(self, text):
        """单个查询文本向量化"""
        return self.embed_documents([text])[0]

# 需要自定义实现的HuggingFaceEmbeddings类
class HuggingFaceEmbeddings:
    """
    HuggingFace Embeddings 模型的简单封装
    """
    def __init__(self, model_name="moka-ai/m3e-base"):
        self.model_name = model_name
        try:
            print("尝试导入sentence_transformers")  
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"导入sentence_transformers成功")
        except ImportError:
            raise ValueError(
                "sentence_transformers package not found, please install it with "
                "`pip install sentence-transformers`"
            )
        
    def embed_documents(self, texts):
        """批量文本向量化"""
        # 调用sentence_transformers进行向量化
        print("批量文本向量化.调用 model.encode 进行向量化")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    def embed_query(self, text):
        """单个查询文本向量化"""
        print("单个查询文本向量化.调用 embed_documents 进行向量化")
        return self.embed_documents([text])[0]

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    """
    统一获取各种Embedding模型的接口函数
    参数:
        embedding (str): Embedding模型名称，支持'm3e'(HuggingFace), 'openai', 'zhipuai'
        embedding_key (str, optional): 模型API密钥，如果为None则自动从环境变量或配置文件获取
        env_file (str, optional): 环境变量文件路径，用于加载API密钥
    返回:
        object: 对应的Embedding模型对象，可直接用于文本向量化
    异常:
        ValueError: 当指定的embedding模型不支持时抛出
    """
    if embedding == 'm3e':
        # 返回 HuggingFace 的 m3e-base Embedding 模型对象
        print(f"返回 HuggingFace 的 moka-ai/m3e-base Embedding 模型对象")
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    if embedding_key == None:
        # 如果未传入 embedding_key，则自动解析获取
        embedding_key = parse_llm_api_key(embedding)
    if embedding == "openai":
        # 返回 OpenAI Embedding 模型对象
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    elif embedding == "zhipuai":
        # 返回智谱AI Embedding 模型对象
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        # 不支持的embedding类型抛出异常
        raise ValueError(f"embedding {embedding} not support ")