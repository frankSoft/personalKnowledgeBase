'''
在 LangChain LLM 基础上封装的项目类，统一了 GPT、文心、讯飞、智谱多种 API 调用
'''
from langchain.llms.base import LLM
from typing import Dict, Any, Mapping
from pydantic import Field

class Self_LLM(LLM):
    # 自定义 LLM, 继承自 langchain.llms.base.LLM
    # 原生接口地址
    url : str =  None
    # 默认选用 deepseek 模型
    model_name: str = "deepseek-chat"
    # 访问时延上限
    request_timeout: float = None
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None
    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # 定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """
        获取调用大模型的默认参数。
        返回:
            Dict[str, Any]: 包含默认参数的字典，包括温度系数和请求超时时间等
            这些参数将在调用大模型API时使用
        """
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        # print(type(self.model_kwargs))
        return {**normal_params}
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        获取模型的标识参数，用于在日志或调试中识别当前模型的配置。
        返回: Mapping[str, Any]: 包含模型名称和默认参数的字典，用于唯一标识模型配置
        """
        return {**{"model_name": self.model_name}, **self._default_params}