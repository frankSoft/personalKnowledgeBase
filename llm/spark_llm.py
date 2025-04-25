'''
基于讯飞星火大模型自定义 LLM 类
'''
from langchain.llms.base import LLM  # 导入 LangChain 的 LLM 基类
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple  # 类型注解
from pydantic import Field  # Pydantic 字段类型
from llm.self_llm import Self_LLM  # 导入自定义 LLM 基类
import json  # 用于处理 JSON 数据
import requests  # 用于发送 HTTP 请求
from langchain.callbacks.manager import CallbackManagerForLLMRun  # LangChain 回调管理
import _thread as thread  # 用于多线程操作
import base64  # 用于 base64 编解码
import datetime  # 用于处理时间
import hashlib  # 用于哈希加密
import hmac  # 用于 HMAC 加密
import json  # 冗余导入，可省略
from urllib.parse import urlparse  # 用于解析 URL
import ssl  # 用于处理 SSL 证书
from datetime import datetime  # 用于获取当前时间
from time import mktime  # 用于时间戳转换
from urllib.parse import urlencode  # 用于 URL 编码
from wsgiref.handlers import format_date_time  # 用于格式化 HTTP 时间
import websocket  # 导入 websocket_client 库
import queue  # 用于线程安全队列

class Spark_LLM(Self_LLM):
    """
    讯飞星火大模型的自定义 LLM 类，继承自 Self_LLM。
    支持通过 websocket 协议与星火大模型服务交互。
    """
    url : str = "ws://spark-api.xf-yun.com/v1.1/chat"  # 讯飞星火大模型的 websocket 服务地址
    appid : str = None  # APPID
    api_secret : str = None  # APISecret
    domain :str = "general"  # 领域参数
    max_tokens : int = 4096  # 最大 token 数

    def getText(self, role, content, text = []):
        """
        构造对话消息格式，role 是指定角色，content 是 prompt 内容
        参数:
            role: 角色（如"user"）
            content: 消息内容
            text: 消息列表
        返回:
            更新后的消息列表
        """
        jsoncon = {}  # 新建消息字典
        jsoncon["role"] = role  # 设置角色
        jsoncon["content"] = content  # 设置内容
        text.append(jsoncon)  # 添加到消息列表
        return text  # 返回消息列表

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        """
        LLM 主调用方法，负责将 prompt 发送到星火服务并返回结果。
        参数:
            prompt: 用户输入的 prompt
            stop: 停止词
            run_manager: 回调管理器
            kwargs: 其他参数
        返回:
            星火大模型的回复内容
        """
        if self.api_key == None or self.appid == None or self.api_secret == None:
            # 三个 Key 均存在才可以正常调用
            print("请填入 Key")
            raise ValueError("Key 不存在")
        # 将 Prompt 填充到星火格式
        question = self.getText("user", prompt)
        # 发起请求
        try:
            response = spark_main(self.appid, self.api_key, self.api_secret, self.url, self.domain, question, self.temperature, self.max_tokens)
            return response
        except Exception as e:
            print(e)
            print("请求失败")
            return "请求失败"
        
    @property
    def _llm_type(self) -> str:
        """
        返回 LLM 类型标识
        """
        return "Spark"

answer = ""  # 全局变量，用于 websocket 消息拼接

class Ws_Param(object):
    """
    讯飞星火 websocket 鉴权参数生成类。
    用于生成带鉴权的 websocket 连接 URL。
    """
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID  # APPID
        self.APIKey = APIKey  # APIKey
        self.APISecret = APISecret  # APISecret
        self.host = urlparse(Spark_url).netloc  # 解析 host
        self.path = urlparse(Spark_url).path  # 解析 path
        self.Spark_url = Spark_url  # websocket 服务地址
        self.temperature = 0  # 默认温度
        self.max_tokens = 2048  # 默认最大 token 数

    def create_url(self):
        """
        生成带鉴权参数的 websocket 连接 URL。
        返回:
            带鉴权参数的 websocket 连接 URL
        """
        now = datetime.now()  # 获取当前时间
        date = format_date_time(mktime(now.timetuple()))  # 格式化为 HTTP 时间

        # 拼接签名字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        return url

# WebSocket 回调函数
def on_error(ws, error):
    """
    websocket 错误回调
    """
    print("### error:", error)

def on_close(ws,one,two):
    """
    websocket 关闭回调
    """
    print(" ")

def on_open(ws):
    """
    websocket 连接建立回调，启动新线程发送数据
    """
    thread.start_new_thread(run, (ws,))

def run(ws, *args):
    """
    websocket 发送数据线程
    """
    data = json.dumps(gen_params(appid=ws.appid, domain=ws.domain, question=ws.question, temperature=ws.temperature, max_tokens=ws.max_tokens))
    ws.send(data)

def on_message(ws, message):
    """
    websocket 消息接收回调（全局 answer 版本，已被 spark_main 局部队列替代）
    """
    data = json.loads(message)  # 解析消息
    code = data['header']['code']  # 获取返回码
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]  # 获取回复内容
        status = choices["status"]  # 获取状态
        content = choices["text"][0]["content"]  # 获取文本内容
        print(content, end="")  # 打印内容
        global answer
        answer += content  # 拼接内容
        if status == 2:  # 结束标志
            ws.close()

def gen_params(appid, domain, question, temperature, max_tokens):
    """
    通过appid和用户的提问来生成请求参数
    参数:
        appid: 应用ID
        domain: 领域
        question: 用户问题
        temperature: 生成温度
        max_tokens: 最大 token 数
    返回:
        dict: websocket 请求参数
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": 0.5,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data

def spark_main(appid, api_key, api_secret, Spark_url, domain, question, temperature, max_tokens):
    """
    讯飞星火 websocket 主流程，负责连接、发送、接收并返回最终回复内容。
    参数:
        appid: 应用ID
        api_key: API Key
        api_secret: API Secret
        Spark_url: websocket 服务地址
        domain: 领域
        question: 用户问题
        temperature: 生成温度
        max_tokens: 最大 token 数
    返回:
        str: 模型回复内容
    """
    output_queue = queue.Queue()  # 创建线程安全队列
    def on_message(ws, message):
        data = json.loads(message)  # 解析消息
        code = data['header']['code']  # 获取返回码
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]  # 获取回复内容
            status = choices["status"]  # 获取状态
            content = choices["text"][0]["content"]  # 获取文本内容
            # 将输出值放入队列
            output_queue.put(content)
            if status == 2:  # 结束标志
                ws.close()

    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)  # 创建鉴权参数对象
    websocket.enableTrace(False)  # 关闭调试
    wsUrl = wsParam.create_url()  # 生成带鉴权的 websocket url
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid  # 绑定 appid
    ws.question = question  # 绑定问题
    ws.domain = domain  # 绑定领域
    ws.temperature = temperature  # 绑定温度
    ws.max_tokens = max_tokens  # 绑定最大 token 数
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 启动 websocket 连接
    return ''.join([output_queue.get() for _ in range(output_queue.qsize())])  # 拼接所有回复内容并返回





