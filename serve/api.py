'''
FastAPI服务接口
提供知识库问答的HTTP API接口
支持多种大模型和Embedding模型的选择
可通过参数配置温度、top-k等生成参数
'''
from fastapi import FastAPI  # 导入 FastAPI，用于创建 Web API 服务
from pydantic import BaseModel  # 导入 BaseModel，用于数据校验和序列化
import os  # 导入 os 模块，用于操作系统相关功能
import sys  # 导入 sys 模块，用于操作 Python 运行环境

# 导入功能模块目录
sys.path.append("../")  # 将上级目录加入模块搜索路径，便于导入自定义模块
from qa_chain.QA_chain_self import QA_chain_self  # 导入自定义的问答链类

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"  # 设置 HTTP 代理（如需科学上网可取消注释）
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890" # 设置 HTTPS 代理

app = FastAPI()  # 创建 FastAPI 应用对象

# 定义默认的 prompt 模板，用于生成最终的提示词
template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。最多使用三句话。
尽量使答案简明扼要。总是在回答的最后说"谢谢你的提问！"。
{context}
问题: {question}
有用的回答:"""

# 定义一个数据模型，用于接收POST请求中的数据
class Item(BaseModel):
    """
    用于描述前端请求体的数据结构，包含模型选择、参数配置、API密钥、数据库路径等信息。
    """
    prompt : str # 用户输入的提问内容
    model : str = "deepseek-chat" # 使用的模型名称，默认deepseek
    temperature : float = 0.1 # 生成回答的温度系数，影响输出多样性
    if_history : bool = False # 是否使用历史对话功能，默认不使用
    api_key: str = None # API_Key，用于访问大模型API
    secret_key : str = None # Secret_Key，部分模型API需要
    access_token: str = None # access_token，部分API需要
    appid : str = None # APPID，部分API需要
    spark_api_secret : str = None # 讯飞星火API Secret
    wenxin_secret_key : str = None # 文心一言 Secret Key
    db_path : str = "../vector_db/chroma" # 向量数据库路径
    file_path : str = "../knowledge_db" # 知识库源文件路径
    prompt_template : str = template # prompt 模板
    input_variables : list = ["context","question"] # prompt 模板变量
    embedding : str = "m3e" # 使用的 embedding 模型
    top_k : int = 5 # 检索返回的 top k 条结果
    embedding_key : str = None # embedding API key

@app.post("/")  # 定义 POST 路由，处理根路径的 POST 请求
async def get_response(item: Item):
    """
    接收前端请求，调用问答链进行知识库问答，返回模型生成的答案。
    """
    # 首先确定需要调用的链
    if not item.if_history:  # 如果不使用历史对话链
        # 调用 Chat 链
        # return item.embedding_key
        if item.embedding_key == None:  # 如果 embedding_key 未设置，则使用 api_key
            item.embedding_key = item.api_key
        # 创建 QA_chain_self 实例，传入所有参数
        chain = QA_chain_self(
            model=item.model,
            temperature=item.temperature,
            top_k=item.top_k,
            file_path=item.file_path,
            persist_path=item.db_path,
            appid=item.appid,
            api_key=item.api_key,
            embedding=item.embedding,
            template=template,
            spark_api_secret=item.spark_api_secret,
            wenxin_secret_key=item.wenxin_secret_key,
            embedding_key=item.embedding_key
        )
        # 调用 answer 方法，传入用户问题，获取回答
        response = chain.answer(question = item.prompt)
        return response  # 返回回答内容
    else:
        # 由于 API 存在即时性问题，不能支持历史链
        return "API 不支持历史链"