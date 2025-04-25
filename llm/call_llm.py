'''
将各个大模型的原生接口封装在一个接口
'''
import openai  # OpenAI 官方 Python SDK
import json    # 用于处理 JSON 数据
import requests  # 用于发送 HTTP 请求
import _thread as thread  # 用于多线程操作
import base64   # 用于 base64 编解码
import datetime # 用于处理时间
from dotenv import load_dotenv, find_dotenv  # 用于加载 .env 环境变量
import hashlib  # 用于加密
import hmac     # 用于 HMAC 加密
import os       # 用于操作系统相关功能
import queue    # 用于线程安全的队列
from urllib.parse import urlparse  # 用于解析 URL
import ssl      # 用于处理 SSL 证书
from datetime import datetime      # 用于时间戳
from time import mktime            # 用于时间戳转换
from urllib.parse import urlencode # 用于 URL 编码
from wsgiref.handlers import format_date_time  # 用于格式化 HTTP 时间
import zhipuai  # 智谱AI官方SDK

import websocket  # 使用 websocket_client 库

# DeepSeek API 端点
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1" # 新增 DeepSeek API 端点常量

def get_completion(prompt :str, model :str, temperature=0.1,api_key=None, secret_key=None, access_token=None, appid=None, api_secret=None, max_tokens=2048):
    """
    通用大模型调用入口，根据模型类型自动分发到不同的原生接口。
    支持 OpenAI、百度文心、讯飞星火、智谱、DeepSeek 等主流大模型。 # 更新文档字符串
    参数说明见下方各模型专用函数。
    返回：模型回复字符串
    """
    # 调用大模型获取回复，支持上述三种模型+gpt
    # arguments:
    # prompt: 输入提示
    # model：模型名
    # temperature: 温度系数
    # api_key：如名
    # secret_key, access_token：调用文心系列模型需要
    # appid, api_secret: 调用星火系列模型需要
    # max_tokens : 返回最长序列
    # return: 模型返回，字符串
    # 调用 GPT
    if model in ["gpt-3.5", "gpt-4.1"]: # 保持原有 GPT 模型判断
        return get_completion_gpt(prompt, model, temperature, api_key, max_tokens)
    elif model in ["ERNIE 4.5", "ERNIE Tiny"]:
        return get_completion_wenxin(prompt, model, temperature, api_key, secret_key)
    elif model in ["Spark X1", "Spark Lite"]:
        return get_completion_spark(prompt, model, temperature, api_key, appid, api_secret, max_tokens)
    elif model in ["glm-4-plus", "glm-4-flash-250414"]:
        return get_completion_glm(prompt, model, temperature, api_key, max_tokens)
    # 新增 DeepSeek 模型判断
    elif model in ["deepseek-chat", "deepseek-reasoner"]:
        return get_completion_deepseek(prompt, model, temperature, api_key, max_tokens)
    else:
        # return "不正确的模型" # 可以保留或修改为抛出异常
        raise ValueError(f"不支持的模型: {model}") # 修改为抛出异常更明确

def get_completion_gpt(prompt : str, model : str, temperature : float, api_key:str, max_tokens:int):
    """
    调用 OpenAI GPT 系列模型的原生接口，返回模型回复内容。
    """
    if api_key == None:
        # 尝试从环境变量或 .env 文件获取 OpenAI API Key
        api_key = parse_llm_api_key("openai")
    # 设置 OpenAI API Key
    openai.api_key = api_key
    # 确保使用 OpenAI 官方 API 端点 (如果之前被修改过)
    openai.api_base = "https://api.openai.com/v1" # 确保 API Base 正确
    # 准备发送给模型的消息体
    messages = [{"role": "user", "content": prompt}]
    # 调用 OpenAI 的 ChatCompletion 接口
    response = openai.ChatCompletion.create(
        model=model,             # 指定模型名称
        messages=messages,       # 传入消息列表
        temperature=temperature, # 模型输出的温度系数，控制输出的随机程度
        max_tokens = max_tokens, # 回复最大长度
    )
    # 提取并返回模型生成的回复内容
    return response.choices[0].message["content"]

def get_access_token(api_key, secret_key):
    """
    使用 API Key 和 Secret Key 获取百度文心 access_token。
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

def get_completion_wenxin(prompt : str, model : str, temperature : float, api_key:str, secret_key : str):
    """
    调用百度文心大模型原生接口，返回模型回复内容。
    """
    if api_key == None or secret_key == None:
        api_key, secret_key = parse_llm_api_key("wenxin")
    # 获取access_token
    access_token = get_access_token(api_key, secret_key)
    # 调用接口
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}"
    # 配置 POST 参数
    payload = json.dumps({
        "messages": [
            {
                "role": "user",# user prompt
                "content": "{}".format(prompt)# 输入的 prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    # 发起请求
    response = requests.request("POST", url, headers=headers, data=payload)
    # 返回的是一个 Json 字符串
    js = json.loads(response.text)
    return js["result"]

def get_completion_spark(prompt : str, model : str, temperature : float, api_key:str, appid : str, api_secret : str, max_tokens : int):
    """
    调用讯飞星火大模型原生接口，返回模型回复内容。
    """
    if api_key == None or appid == None and api_secret == None:
        api_key, appid, api_secret = parse_llm_api_key("spark")
    
    # 配置 1.5 和 2 的不同环境
    if model == "Spark-1.5":
        domain = "general"  
        Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
    else:
        domain = "generalv2"    # v2.0版本
        Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址

    question = [{"role":"user", "content":prompt}]
    response = spark_main(appid,api_key,api_secret,Spark_url,domain,question,temperature, max_tokens)
    return response

def get_completion_glm(prompt : str, model : str, temperature : float, api_key:str, max_tokens : int):
    """
    调用智谱 GLM 系列模型原生接口，返回模型回复内容。
    """
    if api_key == None:
        # 尝试从环境变量或 .env 文件获取智谱 API Key
        api_key = parse_llm_api_key("zhipuai")
    # 设置智谱 API Key
    zhipuai.api_key = api_key

    # 调用智谱 Model API
    response = zhipuai.model_api.invoke(
        model=model,                         # 指定模型名称
        prompt=[{"role":"user", "content":prompt}], # 传入用户提示
        temperature = temperature,           # 设置温度系数
        max_tokens=max_tokens                # 设置最大 token 数
        )
    # 提取并清理返回的文本内容
    return response["data"]["choices"][0]["content"].strip('"').strip(" ")

# --- 新增 DeepSeek 调用函数 ---
def get_completion_deepseek(prompt : str, model : str, temperature : float, api_key:str, max_tokens:int):
    """
    调用 DeepSeek 系列模型的原生接口 (兼容 OpenAI)，返回模型回复内容。
    参数:
        prompt (str): 输入的提示文本。
        model (str): 要使用的 DeepSeek 模型名称 (例如 "deepseek-chat", "deepseek-coder")。
        temperature (float): 控制生成文本随机性的温度系数。
        api_key (str): DeepSeek API 密钥。如果为 None，则尝试从环境变量加载。
        max_tokens (int): 生成回复的最大 token 数量。
    返回:
        str: 模型生成的回复内容。
    """
    # 如果未直接提供 API Key，则尝试从环境变量或 .env 文件获取
    if api_key is None:
        api_key = parse_llm_api_key("deepseek") # 使用 parse_llm_api_key 获取

    # 设置 DeepSeek API Key
    openai.api_key = api_key
    # **关键：设置 API 端点为 DeepSeek 的地址**
    openai.api_base = DEEPSEEK_API_BASE

    # 准备发送给模型的消息体
    messages = [{"role": "user", "content": prompt}]
    try:
        # 调用 OpenAI 兼容的 ChatCompletion 接口
        response = openai.ChatCompletion.create(
            model=model,             # 指定 DeepSeek 模型名称
            messages=messages,       # 传入消息列表
            temperature=temperature, # 模型输出的温度系数
            max_tokens=max_tokens,   # 回复最大长度
        )
        # 提取并返回模型生成的回复内容
        return response.choices[0].message["content"]
    except Exception as e:
        # 捕获并打印调用过程中的错误
        print(f"调用 DeepSeek API 时出错: {e}")
        # 可以选择返回错误信息或重新抛出异常
        return f"调用 DeepSeek API 失败: {e}"
    finally:
        # 可选：调用后恢复 OpenAI 的默认 API Base，以防影响后续 OpenAI 调用
        openai.api_base = "https://api.openai.com/v1"

# def getText(role, content, text = []):
#     # role 是指定角色，content 是 prompt 内容
#     jsoncon = {}
#     jsoncon["role"] = role
#     jsoncon["content"] = content
#     text.append(jsoncon)
#     return text

# 星火 API 调用使用的全局变量
answer = ""

class Ws_Param(object):
    """
    讯飞星火 WebSocket 参数生成类。
    用于生成带鉴权的 WebSocket 连接 URL。
    """
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url
        # 自定义
        self.temperature = 0
        self.max_tokens = 2048

    def create_url(self):
        """
        生成带鉴权参数的 WebSocket 连接 URL。
        """
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
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
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 收到websocket错误的处理
def on_error(ws, error):
    """
    WebSocket 错误回调
    """
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,one,two):
    """
    WebSocket 关闭回调
    """
    print(" ")


# 收到websocket连接建立的处理
def on_open(ws):
    """
    WebSocket 连接建立回调，启动新线程发送数据
    """
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    """
    WebSocket 发送数据线程
    """
    data = json.dumps(gen_params(appid=ws.appid, domain= ws.domain,question=ws.question, temperature = ws.temperature, max_tokens = ws.max_tokens))
    ws.send(data)


# 收到websocket消息的处理
def on_message(ws, message):
    """
    WebSocket 消息接收回调函数
    处理从星火大模型服务器返回的消息，并更新全局答案变量
    
    参数:
        ws: WebSocket连接对象
        message: 服务器返回的消息内容，JSON格式字符串
    """
    # print(message)
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content,end ="")
        global answer
        answer += content
        # print(1)
        if status == 2:
            ws.close()


def gen_params(appid, domain, question, temperature, max_tokens):
    """
    生成星火大模型请求参数
    
    参数:
        appid: 星火API的应用ID
        domain: 对话领域，如"general"或"generalv2"
        question: 用户问题或对话历史
        temperature: 生成温度，控制输出的随机性
        max_tokens: 生成的最大token数
        
    返回:
        dict: 结构化的请求参数字典
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
                "temperature" : temperature,
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
    星火大模型主调用函数，通过WebSocket与服务器通信
    
    参数:
        appid: 星火API的应用ID
        api_key: API密钥
        api_secret: API密钥对应的secret
        Spark_url: 星火API的WebSocket URL
        domain: 对话领域
        question: 用户问题或对话历史
        temperature: 生成温度
        max_tokens: 生成的最大token数
        
    返回:
        str: 星火大模型的回答内容
    """
    # print("星火:")
    output_queue = queue.Queue()
    def on_message(ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            # print(content, end='')
            # 将输出值放入队列
            output_queue.put(content)
            if status == 2:
                ws.close()

    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.temperature = temperature
    ws.max_tokens = max_tokens
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return ''.join([output_queue.get() for _ in range(output_queue.qsize())])

def parse_llm_api_key(model:str, env_file:dict()=None):
    """
    通过模型类型和环境变量解析平台参数（API Key、Secret等）。
    支持 openai、wenxin、spark、zhipuai、deepseek。 # 更新文档字符串
    """
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "deepseek":
        return env_file["deepseek_api_key"]
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "wenxin":
        return env_file["wenxin_api_key"], env_file["wenxin_secret_key"]
    elif model == "spark":
        return env_file["spark_api_key"], env_file["spark_appid"], env_file["spark_api_secret"]
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
        # return env_file["ZHIPUAI_API_KEY"]
    else:
        raise ValueError(f"model{model} not support!!!")

# 增加自定义的get_from_dict_or_env函数实现
def get_from_dict_or_env(data_dict, key, env_key):
    """
    从字典或环境变量中获取指定键的值。
    首先尝试从字典中获取，如果不存在则从环境变量中获取。
    参数:
        data_dict (dict): 要搜索的字典
        key (str): 字典中的键名
        env_key (str): 环境变量名
    返回:
        str: 找到的值
    异常:
        ValueError: 当在字典和环境变量中都找不到指定的键时抛出
    """
    if key in data_dict and data_dict[key]:
        return data_dict[key]

    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]

    raise ValueError(
        f"无法找到变量 {key} 或环境变量 {env_key}，请确保在字典中或环境变量中设置了此值。"
    )

# --- 添加 main 测试块 ---
if __name__ == "__main__":
    # 加载 .env 文件中的环境变量 (如果存在)
    _ = load_dotenv(find_dotenv())

    # 定义测试用例：模型名称和对应的 prompt
    test_cases = [
        # {"model": "gpt-3.5", "prompt": "你好，GPT-3.5！请介绍一下你自己。"}, # 取消注释以测试 GPT-3.5
        # {"model": "ERNIE 4.5", "prompt": "你好，文心一言！请介绍一下你自己。"}, # 取消注释以测试文心
        # {"model": "Spark Lite", "prompt": "你好，讯飞星火！请介绍一下你自己。"}, # 取消注释以测试星火
        # {"model": "glm-4-flash-250414", "prompt": "你好，智谱 GLM！请介绍一下你自己。"}, # 取消注释以测试智谱
        {"model": "deepseek-chat", "prompt": "你好，DeepSeek！请介绍一下你自己。"}, # 测试 DeepSeek
    ]

    print("--- 开始测试 get_completion 函数 ---")

    for case in test_cases:
        model_name = case["model"]
        prompt_text = case["prompt"]

        print(f"\n--- 测试模型: {model_name} ---")
        print(f"Prompt: {prompt_text}")

        try:
            # 调用通用的 get_completion 函数
            # 函数内部会根据 model_name 自动调用相应的实现并处理 API Key
            response = get_completion(prompt=prompt_text, model=model_name)
            print("\n模型回复:")
            print(response)
        except ValueError as ve:
            print(f"配置或模型错误: {ve}")
        except KeyError as ke:
            print(f"错误：缺少必要的环境变量或 API 密钥配置 for {model_name}。错误详情: {ke}")
        except Exception as e:
            print(f"调用模型 {model_name} 时发生未知错误: {e}")
            # 可以取消注释下一行以打印详细堆栈信息
            # import traceback
            # traceback.print_exc()

        print(f"--- 模型 {model_name} 测试结束 ---")

    print("\n--- 所有测试完成 ---")