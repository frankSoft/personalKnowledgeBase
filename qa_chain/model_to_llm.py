'''
模型到LLM实例转换模块
将模型名称和参数转换为对应平台的LLM实例
支持OpenAI、文心、星火、智谱、DeepSeek等多种大模型 # 更新文档：增加 DeepSeek
提供统一的模型初始化接口
'''
import sys  # 导入 sys 模块，用于操作 Python 运行环境
sys.path.append("../llm")  # 添加 llm 目录到模块搜索路径，便于导入自定义大模型相关模块
from llm.wenxin_llm import Wenxin_LLM  # 导入百度文心 LLM 封装类
from llm.spark_llm import Spark_LLM    # 导入讯飞星火 LLM 封装类
from llm.zhipuai_llm import ZhipuAILLM # 导入智谱 AI LLM 封装类
from langchain.chat_models import ChatOpenAI  # 导入 OpenAI 聊天模型
from llm.call_llm import parse_llm_api_key    # 导入 API key 解析工具函数

# 新增 DeepSeek API 端点常量
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

def model_to_llm(model: str = None, temperature: float = 0.0, appid: str = None, api_key: str = None, Spark_api_secret: str = None, Wenxin_secret_key: str = None):
    """
    根据模型名称和参数，返回对应的大模型 LLM 实例。
    支持 OpenAI、百度文心、讯飞星火、智谱、DeepSeek 等主流大模型平台。 # 更新文档：增加 DeepSeek
    参数说明：
    - model: 模型名称（如deepseek-chat 等） # 更新文档：增加 DeepSeek 示例
    - temperature: 生成温度参数，控制输出多样性
    - appid: 星火专用 APPID
    - api_key: 各平台 API Key
    - Spark_api_secret: 星火专用 Secret
    - Wenxin_secret_key: 文心专用 Secret
    返回：
    - llm: 对应平台的 LLM 实例，可直接用于推理
    """
    # OpenAI 系列模型
    if model in ["gpt-3.5", "gpt-4.1"]:
        if api_key is None:
            api_key = parse_llm_api_key("openai")  # 自动获取 OpenAI API Key
        # 确保使用 OpenAI 官方 API 端点
        llm = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key, openai_api_base="https://api.openai.com/v1")
    # 百度文心系列模型
    elif model in ["ERNIE 4.5", "ERNIE Tiny"]:
        if api_key is None or Wenxin_secret_key is None:
            api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")  # 自动获取文心 API Key 和 Secret
        llm = Wenxin_LLM(model=model, temperature=temperature, api_key=api_key, secret_key=Wenxin_secret_key)
    # 讯飞星火系列模型
    elif model in ["Spark X1", "Spark Lite"]:
        if api_key is None or appid is None or Spark_api_secret is None:
            api_key, appid, Spark_api_secret = parse_llm_api_key("spark")  # 自动获取星火 API Key、APPID、Secret
        llm = Spark_LLM(model=model, temperature=temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
    # 智谱系列模型
    elif model in ["glm-4-plus", "glm-4-flash-250414"]:
        if api_key is None:
            api_key = parse_llm_api_key("zhipuai")  # 自动获取智谱 API Key
        llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature=temperature)
    # 新增 DeepSeek 系列模型
    elif model in ["deepseek-chat", "deepseek-coder"]:
        if api_key is None:
            api_key = parse_llm_api_key("deepseek") # 自动获取 DeepSeek API Key
        # 使用 ChatOpenAI 类，但指定 DeepSeek 的 API 端点
        llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=DEEPSEEK_API_BASE # 指定 DeepSeek API 端点
        )
    else:
        # 不支持的模型类型，抛出异常
        raise ValueError(f"model {model} not support!!!") # 修正变量引用格式
    return llm  # 返回大模型实例