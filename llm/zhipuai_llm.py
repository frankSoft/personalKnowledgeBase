'''
基于智谱 AI 大模型自定义 LLM 类
提供智谱API的封装，支持ChatGLM系列模型的调用
支持同步/异步调用、流式输出等功能
继承自Self_LLM基类，实现智谱AI特有的调用逻辑
'''
from __future__ import annotations  # 启用Python未来版本注解特性

import logging  # 导入日志模块，用于记录运行日志
from typing import (  # 导入类型标注相关模块
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
)

from langchain.callbacks.manager import (  # 导入回调管理器
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM  # 导入LangChain的LLM基类
from pydantic import Field, model_validator  # 导入Pydantic验证器
from langchain.schema.output import GenerationChunk  # 导入生成结果块类型
from langchain.utils import get_from_dict_or_env  # 导入工具函数，用于获取环境变量
from llm.self_llm import Self_LLM  # 导入自定义的Self_LLM基类

logger = logging.getLogger(__name__)  # 初始化日志记录器

class ZhipuAILLM(Self_LLM):
    """
    智谱AI大模型LLM封装类
    使用方法：
    需安装zhipuai Python包，并设置环境变量zhipuai_api_key
    支持模型：chatglm_pro, chatglm_std, chatglm_lite
    示例:
        .. code-block:: python
            from langchain.llms import ZhipuAILLM
            zhipuai_model = ZhipuAILLM(model="chatglm_std", temperature=temperature)
    """
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)  # 模型额外参数

    client: Any  # 智谱API客户端

    model: str = "chatglm_std"  # 默认模型为chatglm_std
    """Model name in chatglm_pro, chatglm_std, chatglm_lite. """

    zhipuai_api_key: Optional[str] = None  # 智谱API密钥

    incremental: Optional[bool] = True  # 是否增量返回结果
    """Whether to incremental the results or not."""

    streaming: Optional[bool] = False  # 是否流式返回结果
    """Whether to streaming the results or not."""
    # streaming = -incremental

    request_timeout: Optional[int] = 60  # 请求超时时间
    """request timeout for chat http requests"""

    top_p: Optional[float] = 0.8  # 采样参数top_p
    temperature: Optional[float] = 0.95  # 温度系数
    request_id: Optional[float] = None  # 请求ID

    @model_validator(mode='after')
    def validate_enviroment(self) -> 'ZhipuAILLM':
        """
        验证环境配置并初始化智谱AI客户端
        返回: ZhipuAILLM: 更新后的模型实例
        异常: ValueError: 当未找到zhipuai包时抛出
        """
        self.zhipuai_api_key = get_from_dict_or_env(
            {"zhipuai_api_key": self.zhipuai_api_key},
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            import zhipuai

            zhipuai.api_key = self.zhipuai_api_key
            self.client = zhipuai.model_api
        except ImportError:
            raise ValueError(
                "zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return self

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        获取模型标识参数
        返回: Dict[str, Any]: 包含模型名称和基础参数的字典
        """
        return {
            **{"model": self.model},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        """
        返回LLM类型标识
        返回: str: 当前LLM类型名称，用于LangChain内部标识
        """
        return "zhipuai"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        获取智谱AI API的默认调用参数
        返回: Dict[str, Any]: 默认参数字典
        """
        normal_params = {
            "streaming": self.streaming,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "request_id": self.request_id,
        }
        return {**normal_params, **self.model_kwargs}

    def _convert_prompt_msg_params(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        """
        转换提示和参数为智谱AI API所需格式
        参数:
            prompt: 提示文本
            **kwargs: 其他参数 
        返回: dict: 转换后的参数字典
        """
        return {
            **{"prompt": prompt, "model": self.model},
            **self._default_params,
            **kwargs,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        调用智谱AI模型生成回答
        参数:
            prompt: 输入到模型的提示文本
            stop: 停止词列表，目前未使用
            run_manager: 运行回调管理器
            **kwargs: 其他参数
        返回: str: 模型生成的文本
        说明:
            当流式返回时，将逐个返回生成的文本块
            否则，一次性返回完整的生成结果
        示例:
            .. code-block:: python
                response = zhipuai_model("Tell me a joke.")
        """
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion
        params = self._convert_prompt_msg_params(prompt, **kwargs)

        response_payload = self.client.invoke(**params)
        return response_payload["data"]["choices"][-1]["content"].strip('"').strip(" ")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        异步调用智谱AI模型生成回答
        参数:
            prompt: 输入到模型的提示文本
            stop: 停止词列表，目前未使用 
            run_manager: 异步运行回调管理器
            **kwargs: 其他参数
        返回: str: 模型生成的文本
        """
        if self.streaming:
            completion = ""
            async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion

        params = self._convert_prompt_msg_params(prompt, **kwargs)

        response = await self.client.async_invoke(**params)

        return response_payload

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        流式调用智谱AI模型，同步迭代器方式
        参数:
            prompt: 输入到模型的提示文本
            stop: 停止词列表，目前未使用
            run_manager: 运行回调管理器
            **kwargs: 其他参数
        返回: Iterator[GenerationChunk]: 生成结果块的迭代器
        """
        params = self._convert_prompt_msg_params(prompt, **kwargs)

        for res in self.client.invoke(**params):
            if res:
                chunk = GenerationChunk(text=res)
                yield chunk
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """
        流式调用智谱AI模型，异步迭代器方式
        参数:
            prompt: 输入到模型的提示文本
            stop: 停止词列表，目前未使用
            run_manager: 异步运行回调管理器
            **kwargs: 其他参数
        返回: AsyncIterator[GenerationChunk]: 生成结果块的异步迭代器
        """
        params = self._convert_prompt_msg_params(prompt, **kwargs)

        async for res in await self.client.ado(**params):
            if res:
                chunk = GenerationChunk(text=res["data"]["choices"]["content"])

                yield chunk
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text)
