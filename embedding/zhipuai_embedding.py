'''
智谱AI文本向量化模型封装
基于LangChain的Embeddings接口实现，提供文本向量化能力
支持单文本和批量文本的向量化处理
用于知识库文档的向量化存储和检索
'''
from __future__ import annotations  # 兼容未来的注解语法

import logging  # 导入日志模块
import os       # 导入操作系统模块
import sys      # 导入系统模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 将上级目录加入sys.path，便于导入模块
from typing import Any, Dict, List, Optional  # 导入类型注解
from pydantic import BaseModel, model_validator  # 导入pydantic用于数据校验和模型定义
from llm.call_llm import get_from_dict_or_env  # 导入自定义的环境变量获取工具

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 定义Embeddings基类接口（简化版，仅保留必要方法）
class Embeddings:
    """接口提供了获取文本向量的方法"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为向量"""
        raise NotImplementedError  # 子类需实现该方法
        
    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为向量"""
        raise NotImplementedError  # 子类需实现该方法

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models.
    封装智谱AI的文本向量化接口，继承pydantic的BaseModel和自定义Embeddings接口
    """
    zhipuai_api_key: Optional[str] = None  # 智谱AI API Key，可从环境变量或参数获取

    model: str = "text_embedding"  # 模型名称，智谱AI当前只有一个embedding模型
    
    request_id: Optional[str] = None  # 对应智谱AI的请求ID，可选
    client: Any = None  # 智谱AI客户端对象

    @model_validator(mode='after')
    def validate_environment(self) -> 'ZhipuAIEmbeddings':
        """
        验证智谱AI API Key是否在环境变量或配置中可用。
        若未找到zhipuai包则抛出异常。
        """
        # 获取API Key，优先使用传入参数，否则从环境变量读取
        self.zhipuai_api_key = get_from_dict_or_env(
            {"zhipuai_api_key": self.zhipuai_api_key},
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            import zhipuai  # 动态导入zhipuai包
            zhipuai.api_key = self.zhipuai_api_key  # 设置API Key
            self.client = zhipuai.model_api  # 获取模型API客户端

        except ImportError:
            # 未安装zhipuai包时抛出异常
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return self

    def _embed(self, texts: str) -> List[float]:
        """
        调用智谱AI的文本向量化接口，获取文本的向量表示
        参数:
            texts (str): 需要向量化的文本
        返回:
            List[float]: 文本的向量表示，一个浮点数列表
        异常:
            ValueError: 当API调用失败或返回非200状态码时抛出
        """
        try:
            # 调用智谱AI模型API进行文本向量化
            resp = self.client.invoke(
                model="text_embedding",  # 指定模型名称
                prompt=texts             # 输入文本
            )
        except Exception as e:
            # 捕获API调用异常并抛出
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if resp["code"] != 200:
            # 若API返回非200状态码，抛出异常
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (resp["code"], resp["msg"])
            )
        embeddings = resp["data"]["embedding"]  # 获取返回的embedding向量
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本向量化
        参数:
            text (str): 需要向量化的查询文本
        返回:
            List[float]: 查询文本的向量表示，一个浮点数列表
        """
        resp = self.embed_documents([text])  # 调用批量接口处理单文本
        return resp[0]  # 返回第一个（也是唯一一个）向量

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量将多个文档文本向量化
        参数:
            texts (List[str]): 需要向量化的文档文本列表
        返回:
            List[List[float]]: 每个文档的向量表示列表，每个向量是一个浮点数列表
        """
        # 遍历每个文本，分别调用_embed方法获取向量
        return [self._embed(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        异步批量向量化文档文本（当前未实现）
        参数:
            texts (List[str]): 需要向量化的文档文本列表
        异常:
            NotImplementedError: 当前智谱API不支持异步请求，抛出未实现错误
        """
        # 智谱官方API暂不支持异步请求，直接抛出异常
        raise NotImplementedError(
            "请使用`embed_documents`。智谱官方API暂不支持异步请求")

    async def aembed_query(self, text: str) -> List[float]:
        """
        异步向量化单个查询文本（当前未实现）
        参数:
            text (str): 需要向量化的查询文本
        返回:
            List[float]: 查询文本的向量表示，一个浮点数列表
        异常:
            NotImplementedError: 当前智谱API不支持异步请求，抛出未实现错误
        """
        # 智谱官方API暂不支持异步请求，直接抛出异常
        raise NotImplementedError(
            "请使用`aembed_query`。智谱官方API暂不支持异步请求")
