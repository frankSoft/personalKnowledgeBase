'''
基于百度文心大模型自定义 LLM 类
提供文心大模型API的访问封装
支持获取access_token和调用模型生成内容
继承自Self_LLM基类，实现百度文心特有的调用逻辑
'''
from langchain.llms.base import LLM  # 导入LangChain的LLM基类，用于实现自定义LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple  # 导入类型提示模块，用于类型标注
from pydantic import Field  # 导入Pydantic的Field，用于模型字段定义
from llm.self_llm import Self_LLM  # 导入自定义的Self_LLM基类
import json  # 导入json模块，用于处理JSON数据
import requests  # 导入requests模块，用于发送HTTP请求
from langchain.callbacks.manager import CallbackManagerForLLMRun  # 导入回调管理器，用于LLM运行时回调

# 调用文心 API 的工具函数
def get_access_token(api_key : str, secret_key : str):
    """
    使用 API Key，Secret Key 获取百度文心的access_token
    参数:
        api_key (str): 百度文心应用的API Key
        secret_key (str): 百度文心应用的Secret Key
    返回: str: 获取到的access_token，用于后续API调用鉴权
    """
    # 指定网址
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 设置 POST 访问
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # 通过 POST 访问获取账户对应的 access_token
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

class Wenxin_LLM(Self_LLM):
    """
    百度文心大模型的自定义 LLM 类
    封装文心API调用逻辑，继承自Self_LLM
    """
    # URL
    url : str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}"
    # Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None

    def init_access_token(self):
        """
        初始化获取文心API的access_token
        当api_key和secret_key均不为空时，尝试获取access_token
        """
        if self.api_key != None and self.secret_key != None:
            # 两个 Key 均非空才可以获取 access_token
            try:
                self.access_token = get_access_token(self.api_key, self.secret_key)
            except Exception as e:
                print(e)
                print("获取 access_token 失败，请检查 Key")
        else:
            print("API_Key 或 Secret_Key 为空，请检查 Key")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        """
        调用文心大模型API的核心方法
        参数:
            prompt (str): 用户输入的提示文本
            stop (Optional[List[str]]): 停止词列表，目前未使用
            run_manager (Optional[CallbackManagerForLLMRun]): 运行回调管理器，目前未使用
            **kwargs (Any): 其他参数
        返回: str: 模型生成的回答文本
        """
        # 如果 access_token 为空，初始化 access_token
        if self.access_token == None:
            self.init_access_token()
        # API 调用 url
        url = self.url.format(self.access_token)
        # 配置 POST 参数
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",# user prompt
                    "content": "{}".format(prompt)# 输入的 prompt
                }
            ],
            'temperature' : self.temperature
        })
        headers = {
            'Content-Type': 'application/json'
        }
        # 发起请求
        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)
        if response.status_code == 200:
            # 返回的是一个 Json 字符串
            js = json.loads(response.text)
            # print(js)
            return js["result"]
        else:
            return "请求失败"
        
    @property
    def _llm_type(self) -> str:
        """
        返回LLM类型标识
        返回: str: 当前LLM类型名称，用于LangChain内部标识
        """
        return "Wenxin"
