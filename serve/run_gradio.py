'''
    基于Gradio构建的Web界面服务程序
    提供知识库问答、大模型对话、多模型切换等功能
    支持带历史对话和不带历史对话两种模式
    可以上传和处理自定义知识库文件
'''
# 导入必要的库
import sys  # 用于操作 Python 运行环境和模块路径
import os   # 用于操作系统相关的操作，例如读取环境变量

# 将项目根目录加入模块搜索路径，便于导入自定义模块
# os.path.dirname(__file__) 获取当前文件所在目录
# os.path.dirname(os.path.dirname(__file__)) 获取上级目录，即项目根目录
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import IPython.display   # 用于在 IPython 环境中显示数据，例如图片（在此脚本中可能未直接使用，但保留导入）
import io                # 用于处理内存中的二进制流（例如文件流）
import gradio as gr      # 导入 Gradio 库，用于快速构建 Web UI
from dotenv import load_dotenv, find_dotenv  # 导入用于加载 .env 文件环境变量的函数
from llm.call_llm import get_completion      # 导入自定义的调用大语言模型 (LLM) 的函数
from database.create_db import create_db_info  # 导入创建或更新知识库向量数据库的函数
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self  # 导入带聊天历史的问答链类
from qa_chain.QA_chain_self import QA_chain_self            # 导入不带聊天历史的问答链类
import re  # 导入正则表达式模块，用于文本处理
import traceback # 导入 traceback 模块以打印详细错误堆栈
from typing import List, Tuple # 导入类型提示

# 加载 .env 文件中的环境变量
# find_dotenv() 会自动查找 .env 文件
# load_dotenv() 会将 .env 文件中的键值对加载到环境变量中
_ = load_dotenv(find_dotenv())

# 定义支持的大模型平台及其对应的模型名称列表
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5", "gpt-4.1"], # OpenAI 模型
    "wenxin": ["ERNIE 4.5", "ERNIE Tiny"], # 文心一言模型
    "xinhuo": ["Spark X1", "Spark Lite"], # 讯飞星火模型
    "zhipuai": ["glm-4-plus", "glm-4-flash-250414"], # 智谱AI模型
    "deepseek": ["deepseek-chat", "deepseek-reasoner"], # Deepseek 模型
}

# 将所有模型名称合并到一个列表中，用于 Gradio 下拉菜单选项
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = "deepseek-chat"  # 设置默认使用的大语言模型
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']  # 定义支持的文本嵌入 (Embedding) 模型列表
INIT_EMBEDDING_MODEL = "m3e"  # 设置默认使用的 Embedding 模型
DEFAULT_DB_PATH = "./knowledge_db"  # 默认的原始知识库文件存放路径
DEFAULT_PERSIST_PATH = "./vector_db/chroma"  # 默认的向量数据库持久化存储路径

# 定义界面中使用的头像和 Logo 图片路径
AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"  # AI 机器人头像
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"  # Datawhale 用户头像
AIGC_LOGO_PATH = "./figures/aigc_logo.png"  # AI Logo (代码中注释掉了，未使用)
DATAWHALE_LOGO_PATH = "./figures/datawhale_logo.png"  # Datawhale Logo (代码中注释掉了，未使用)

def get_model_by_platform(platform):
    """
    根据指定的平台名称，返回该平台支持的大模型列表。
    如果平台名称无效，返回空字符串。
    参数: platform (str): 大模型平台名称 (例如 "openai", "wenxin")
    返回: list or str: 对应平台的模型列表，如果平台不存在则返回空字符串
    """
    return LLM_MODEL_DICT.get(platform, "") # 使用字典的 get 方法安全地获取值

class Model_center():
    """
    问答链管理中心类。
    用于缓存和管理不同模型和 Embedding 组合的问答链实例，避免重复创建。
    包含两种类型的问答链缓存：
    - chat_qa_chain_self: 缓存带聊天历史的问答链实例 (Chat_QA_chain_self)
    - qa_chain_self: 缓存不带聊天历史的问答链实例 (QA_chain_self)
    """
    def __init__(self):
        """
        初始化 Model_center 类。
        创建两个空字典用于缓存问答链实例。
        """
        self.chat_qa_chain_self = {}  # 初始化带历史问答链的缓存字典
        self.qa_chain_self = {}       # 初始化不带历史问答链的缓存字典

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = None, model: str = "deepseek-chat", embedding: str = "m3e", temperature: float = 0.0, top_k: int = 4, history_len: int = 3, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH) -> tuple:
        """
        调用带历史记录的问答链进行回答。
        如果对应的模型和 Embedding 组合的问答链实例已存在于缓存中，则直接使用；否则，创建一个新的实例并缓存。
        参数:
            question (str): 用户提出的问题。
            chat_history (list, optional): 聊天历史记录列表，每个元素是一个 (用户问题, AI回答) 的元组。默认为 None，会初始化为空列表。
            model (str, optional): 使用的大语言模型名称。默认为 "deepseek-chat"。
            embedding (str, optional): 使用的文本嵌入模型名称。默认为 "m3e"。
            temperature (float, optional): LLM 生成文本时的温度参数，控制随机性。默认为 0.0。
            top_k (int, optional): 从向量数据库检索时返回的最相似文档数量。默认为 4。
            history_len (int, optional): 控制问答链内部保留的对话轮数。默认为 3。
            file_path (str, optional): 知识库源文件所在的目录路径。默认为 DEFAULT_DB_PATH。
            persist_path (str, optional): 向量数据库持久化存储的路径。默认为 DEFAULT_PERSIST_PATH。
        返回:
            tuple: 一个元组，包含两个元素：
                   - str: 错误信息（如果发生异常）或空字符串（如果成功）。
                   - list: 更新后的聊天历史记录列表。
        """
        if chat_history is None: # 如果未传入 chat_history，则初始化为空列表
            chat_history = []
        if question is None or len(question) < 1:  # 检查问题是否有效
            return "", chat_history # 无效问题则直接返回空回复和原历史
        try:
            # 使用 (模型, embedding) 作为缓存的键
            cache_key = (model, embedding)
            # 检查缓存中是否已存在对应的问答链实例
            if cache_key not in self.chat_qa_chain_self:
                # 如果不存在，则创建一个新的 Chat_QA_chain_self 实例并存入缓存
                print(f"Creating new Chat_QA_chain_self for {cache_key}") # 打印日志，方便调试
                self.chat_qa_chain_self[cache_key] = Chat_QA_chain_self(
                    model=model,                      # LLM 模型名称
                    temperature=temperature,          # 温度参数
                    top_k=top_k,                      # 检索 Top K
                    chat_history=chat_history,        # 初始聊天历史
                    file_path=file_path,              # 知识库路径
                    persist_path=persist_path,        # 向量库路径
                    embedding=embedding               # Embedding 模型名称
                )
            # 从缓存中获取问答链实例
            chain = self.chat_qa_chain_self[cache_key]
            # 调用问答链实例的 answer 方法获取回答，该方法内部会处理历史记录
            # 注意：这里的 temperature 和 top_k 再次传入，允许动态调整，但通常建议在创建实例时固定
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            # 捕获执行过程中可能出现的任何异常
            error_msg = f"Error in chat_qa_chain_self_answer: {e}" # 格式化错误信息
            print(error_msg) # 打印错误信息到控制台
            traceback.print_exc()  # 打印完整的错误堆栈
            return error_msg, chat_history # 返回错误信息和原始聊天历史

    def qa_chain_self_answer(self, question: str, chat_history: list = None, model: str = "deepseek-chat", embedding="m3e", temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH) -> tuple:
        """
        调用不带历史记录的问答链进行回答。
        与带历史记录的版本类似，也会缓存问答链实例。
        注意：此方法虽然接收 chat_history 参数，但 QA_chain_self 本身不维护内部历史状态。
              它仅将当前问答追加到传入的 chat_history 列表中并返回。
        参数:
            question (str): 用户提出的问题。
            chat_history (list, optional): 界面显示的聊天历史记录列表。默认为 None，会初始化为空列表。
            model (str, optional): 使用的大语言模型名称。默认为 "deepseek-chat"。
            embedding (str, optional): 使用的文本嵌入模型名称。默认为 "m3e"。
            temperature (float, optional): LLM 生成文本时的温度参数。默认为 0.0。
            top_k (int, optional): 从向量数据库检索时返回的最相似文档数量。默认为 4。
            file_path (str, optional): 知识库源文件所在的目录路径。默认为 DEFAULT_DB_PATH。
            persist_path (str, optional): 向量数据库持久化存储的路径。默认为 DEFAULT_PERSIST_PATH。
        返回:
            tuple: 一个元组，包含两个元素：
                   - str: 错误信息（如果发生异常）或空字符串（如果成功）。
                   - list: 更新后的聊天历史记录列表（包含本次问答）。
        """
        if chat_history is None: # 初始化聊天历史
            chat_history = []
        if question is None or len(question) < 1: # 检查问题有效性
            return "", chat_history
        try:
            # 使用 (模型, embedding) 作为缓存的键
            cache_key = (model, embedding)
            # 检查缓存中是否已存在对应的问答链实例
            if cache_key not in self.qa_chain_self:
                # 如果不存在，则创建一个新的 QA_chain_self 实例并存入缓存
                print(f"Creating new QA_chain_self for {cache_key}") # 打印日志
                self.qa_chain_self[cache_key] = QA_chain_self(
                    model=model,                      # LLM 模型名称
                    temperature=temperature,          # 温度参数
                    top_k=top_k,                      # 检索 Top K
                    file_path=file_path,              # 知识库路径
                    persist_path=persist_path,        # 向量库路径
                    embedding=embedding               # Embedding 模型名称
                )
            # 从缓存中获取问答链实例
            chain = self.qa_chain_self[cache_key]
            # 调用问答链实例的 answer 方法获取回答
            answer = chain.answer(question, temperature, top_k)
            # 将当前的问答对追加到传入的 chat_history 列表中
            chat_history.append((question, answer))
            # 返回空错误信息和更新后的聊天历史
            return "", chat_history
        except Exception as e:
            # 捕获并处理异常
            error_msg = f"Error in qa_chain_self_answer: {e}"
            print(error_msg)
            traceback.print_exc()  # 打印完整的错误堆栈
            return error_msg, chat_history # 返回错误信息和原始聊天历史

    def clear_history(self):
        """
        清空所有已缓存的带历史问答链 (Chat_QA_chain_self) 实例的内部历史记录。
        这对于在 Gradio 界面上点击"清空聊天"按钮时重置对话状态非常有用。
        """
        if len(self.chat_qa_chain_self) > 0: # 检查缓存是否为空
            # 遍历缓存中所有的 Chat_QA_chain_self 实例
            for chain in self.chat_qa_chain_self.values():
                # 调用每个实例的 clear_history 方法（假设 Chat_QA_chain_self 类有此方法）
                if hasattr(chain, 'clear_history') and callable(getattr(chain, 'clear_history')):
                    chain.clear_history()  # 正确缩进
            print("Cleared history for all cached Chat_QA_chain_self instances.") # 打印日志

def format_chat_prompt(message, chat_history):
    """
    将用户的当前消息和聊天历史格式化为适用于某些大语言模型的标准输入 prompt 格式。
    通常格式为多轮对话形式，例如：
    User: 问题1
    Assistant: 回答1
    User: 问题2
    Assistant:
    参数:
        message (str): 用户当前输入的消息。
        chat_history (list): 聊天历史记录列表，每个元素是 (用户问题, AI回答) 的元组。
    返回:
        str: 格式化后的完整 prompt 字符串。
    """
    prompt = ""  # 初始化 prompt 字符串
    # 遍历聊天历史
    for turn in chat_history:
        user_message, bot_message = turn # 解包用户和AI的消息
        # 拼接历史对话到 prompt
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 拼接当前用户消息，并提示 AI 开始回答
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt # 返回最终的 prompt

def respond(message: str, chat_history: List[Tuple[str, str]], llm: str, history_len: int = 3, temperature: float = 0.1, max_tokens: int = 2048) -> Tuple[str, List[Tuple[str, str]]]:
    """
    直接调用大语言模型进行对话，不经过知识库检索。
    用于实现纯粹的聊天机器人功能。
    参数:
        message (str): 用户当前输入的消息。
        chat_history (list): 聊天历史记录列表。
        llm (str): 要调用的 LLM 模型名称。
        history_len (int, optional): 控制输入给 LLM 的历史对话轮数。默认为 3。
        temperature (float, optional): LLM 生成时的温度参数。默认为 0.1。
        max_tokens (int, optional): LLM 生成回复的最大 token 数量。默认为 2048。
    返回:
        tuple: 一个元组，包含两个元素：
               - str: 错误信息（如果发生异常）或空字符串（如果成功）。
               - list: 更新后的聊天历史记录列表（包含本次问答）。
    """
    # 增加详细的输入日志，帮助调试 422 问题
    print(f"--- Entering respond function ---")
    print(f"message: {message} (type: {type(message)})")
    print(f"chat_history: {chat_history} (type: {type(chat_history)})")
    print(f"llm: {llm} (type: {type(llm)})")
    print(f"history_len: {history_len} (type: {type(history_len)})")
    print(f"temperature: {temperature} (type: {type(temperature)})")
    print(f"max_tokens: {max_tokens} (type: {type(max_tokens)})")
    print(f"---------------------------------")

    if message is None or len(message) < 1: # 检查消息有效性
        return "", chat_history
    try:
        # 根据 history_len 截取最近的聊天历史
        limited_history = chat_history[-history_len:] if history_len > 0 else []
        # 使用 format_chat_prompt 格式化输入给 LLM 的 prompt
        formatted_prompt = format_chat_prompt(message, limited_history)
        # 调用 get_completion 函数与 LLM 交互
        bot_message = get_completion(
            prompt=formatted_prompt, # 格式化后的 prompt
            model=llm,                 # LLM 模型名称
            temperature=temperature, # 温度参数
            max_tokens=max_tokens    # 最大 token 数
        )
        # 将 LLM 返回的换行符 '\n' 替换为 HTML 的换行标签 '<br/>'，以便在 Gradio Chatbot 中正确显示
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # 将当前问答对追加到聊天历史中
        chat_history.append((message, bot_message))
        # 返回空错误信息和更新后的聊天历史
        return "", chat_history
    except Exception as e:
        # 捕获并处理异常
        error_msg = f"Error in respond: {e}"
        print(error_msg)
        traceback.print_exc()  # 打印完整的错误堆栈
        # 在返回给 Gradio 前，确保 chat_history 仍然是 list of tuples
        if not isinstance(chat_history, list):
             chat_history = [] # 或者尝试恢复之前的状态
        return error_msg, chat_history # 返回错误信息和原始聊天历史

# --- Gradio 界面构建 ---

# 定义用于处理知识库初始化的包装函数，包含错误处理
def handle_init_db(file_obj, embedding_model):
    """
    处理 Gradio 文件上传并调用知识库向量化函数。
    包含对 file_obj 类型的判断和异常捕获。
    参数:
        file_obj: Gradio File 组件返回的对象，可能是临时文件对象或路径字符串。
        embedding_model (str): 选择的 Embedding 模型名称。
    返回:
        str: 处理结果信息（成功或失败）。
    """
    print(f"handle_init_db 开始向量化处理，文件/目录: {file_obj}") # 打印日志
    print(f"handle_init_db 使用 Embedding 模型: {embedding_model}") # 打印日志
    try:
        if file_obj is None: # 检查是否上传了文件
            return "请先上传知识库文件"

        # Gradio File 组件返回的对象可能是 tempfile._TemporaryFileWrapper 或 str
        # 需要正确获取文件路径
        file_path = None
        if hasattr(file_obj, "name"):  # 如果是临时文件对象
            file_path = file_obj.name
        elif isinstance(file_obj, str): # 如果是字符串路径 (可能在某些 Gradio 版本或配置下出现)
            file_path = file_obj
        else:
            return f"无法识别的文件对象类型: {type(file_obj)}"

        print(f"开始向量化处理，文件/目录: {file_path}") # 打印日志
        print(f"使用 Embedding 模型: {embedding_model}") # 打印日志

        # 调用核心的向量化函数
        # 注意：create_db_info 预期接收的是目录路径或单个文件路径
        # 如果 Gradio 上传的是单个文件，需要确保 create_db_info 能处理
        # 或者在此处将文件移动到预期目录
        # 使用关键字参数传递，更清晰
        result = create_db_info(files=file_path, embeddings=embedding_model)
        success_msg = f"知识库向量化完成！结果: {result}" # 假设 create_db_info 返回一些状态信息
        print(success_msg)
        return success_msg # 返回成功信息
    except Exception as e:
        # 捕获向量化过程中可能出现的异常
        error_msg = f"知识库文件向量化失败: {str(e)}"
        traceback.print_exc() # 打印详细错误信息到控制台
        return error_msg # 返回错误信息给 Gradio 界面

model_center = Model_center()  # 在全局范围初始化问答链管理中心实例

# --- 新增：处理提交事件的包装函数 ---
def handle_submit(message: str,
                  chat_history: list,
                  llm_model: str,
                  embedding_model: str,
                  temperature: float,
                  top_k: int,
                  history_len: int,
                  max_tokens: int,
                  chat_mode: str) -> tuple:
    """
    统一处理聊天提交事件，根据选择的模式调用不同的后端函数。
    """
    print(f"--- Entering handle_submit ---")
    print(f"Chat Mode: {chat_mode}")
    print(f"Message: {message}")
    print(f"LLM: {llm_model}, Embedding: {embedding_model}")
    print(f"Params: temp={temperature}, top_k={top_k}, hist_len={history_len}, max_tokens={max_tokens}")
    print(f"-----------------------------")

    if chat_mode == "LLM 对话":
        # 调用纯 LLM 对话函数
        return respond(message, chat_history, llm_model, history_len, temperature, max_tokens)
    elif chat_mode == "知识库问答（有历史）":
        # 调用带历史的知识库问答包装函数
        # 注意 wrap_chat_qa 内部调用 model_center.chat_qa_chain_self_answer
        # 它需要 question, chat_history, model, embedding, temperature, top_k
        # history_len 和 max_tokens 在此模式下不直接传递给 wrap_chat_qa
        return wrap_chat_qa(message, chat_history, llm_model, embedding_model, temperature, top_k)
    elif chat_mode == "知识库问答（无历史）":
        # 调用不带历史的知识库问答包装函数
        # 注意 wrap_qa 内部调用 model_center.qa_chain_self_answer
        # 它需要 question, chat_history, model, embedding, temperature, top_k
        # history_len 和 max_tokens 在此模式下不直接传递给 wrap_qa
        return wrap_qa(message, chat_history, llm_model, embedding_model, temperature, top_k)
    else:
        # 未知模式，返回错误
        return f"未知的聊天模式: {chat_mode}", chat_history

# 增加详细的输入日志到其他处理函数，帮助调试 422 问题
def wrap_chat_qa(question: str, chat_history: list = None, model: str = "deepseek-chat", embedding: str = "m3e", temperature: float = 0.0, top_k: int = 4) -> tuple:
    print(f"--- Entering wrap_chat_qa (for chat_qa_chain_self_answer) ---")
    print(f"question: {question} (type: {type(question)})")
    print(f"chat_history: {chat_history} (type: {type(chat_history)})")
    print(f"model: {model} (type: {type(model)})")
    print(f"embedding: {embedding} (type: {type(embedding)})")
    print(f"temperature: {temperature} (type: {type(temperature)})")
    print(f"top_k: {top_k} (type: {type(top_k)})")
    print(f"----------------------------------------------------------")
    # 注意：这里没有传递 history_len, file_path, persist_path，使用的是 Model_center 内部的默认值或创建实例时的值
    result = model_center.chat_qa_chain_self_answer(question, chat_history, model, embedding, temperature, top_k)
    # 确保返回的 chat_history 是 list of tuples
    err_msg, hist = result
    if not isinstance(hist, list):
        hist = [] # 或者尝试恢复
    return err_msg, hist


def wrap_qa(question: str, chat_history: list = None, model: str = "deepseek-chat", embedding="m3e", temperature: float = 0.0, top_k: int = 4) -> tuple:
    print(f"--- Entering wrap_qa (for qa_chain_self_answer) ---")
    print(f"question: {question} (type: {type(question)})")
    print(f"chat_history: {chat_history} (type: {type(chat_history)})")
    print(f"model: {model} (type: {type(model)})")
    print(f"embedding: {embedding} (type: {type(embedding)})")
    print(f"temperature: {temperature} (type: {type(temperature)})")
    print(f"top_k: {top_k} (type: {type(top_k)})")
    print(f"---------------------------------------------------")
    # 注意：这里没有传递 file_path, persist_path，使用的是 Model_center 内部的默认值或创建实例时的值
    result = model_center.qa_chain_self_answer(question, chat_history, model, embedding, temperature, top_k)
    # 确保返回的 chat_history 是 list of tuples
    err_msg, hist = result
    if not isinstance(hist, list):
        hist = [] # 或者尝试恢复
    return err_msg, hist


block = gr.Blocks()  # 创建一个 Gradio Blocks 实例，作为 UI 布局的基础
with block as demo: # 使用 'with' 语句构建界面布局
    # 第一行：显示 Logo (当前代码已注释掉 Image 组件) 和标题
    with gr.Row(equal_height=True): # 创建一个行布局，内部元素等高
        # gr.Image(...) # 原本用于显示 AI Logo 的代码
        with gr.Column(scale=2): # 创建一个列布局，占据更多水平空间 (scale=2)
            # 使用 Markdown 组件显示居中的一级标题
            gr.Markdown("""<h1><center>个人知识库</center></h1>""")
        # gr.Image(...) # 原本用于显示 Datawhale Logo 的代码

    # 第二行：主体区域，包含聊天界面和配置选项
    with gr.Row(): # 创建一个行布局
        # 左侧列：聊天界面
        with gr.Column(scale=4): # 创建一个列布局，占据主要空间 (scale=4)
            # 定义一个内部函数 clear_all，用于清空聊天记录和输入框
            def clear_all(msg: str = "", chatbot: list = None) -> tuple:
                """清空聊天界面和模型中心的历史记录"""
                print("Clearing chat history and input.") # 打印日志
                model_center.clear_history()  # 调用模型中心的方法清空缓存中的历史
                return "", [] # 返回空字符串给输入框，空列表给 Chatbot
            # 创建 Chatbot 组件用于显示对话
            chatbot = gr.Chatbot(
                height=400,                 # 设置聊天框高度
                show_copy_button=True,      # 显示复制代码按钮
                show_share_button=True,     # 显示分享按钮 (可能需要 Gradio 托管服务支持)
                avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH) # 设置用户和 AI 的头像
            )
            # 创建 Textbox 组件作为用户输入框
            msg = gr.Textbox(label="Prompt/问题")
            # 创建一行按钮用于不同的聊天模式
            with gr.Row():
                # "带历史知识库问答"按钮
                db_with_his_btn = gr.Button("Chat db with history")
                # "不带历史知识库问答"按钮
                db_wo_his_btn = gr.Button("Chat db without history")
                # "仅与大模型对话"按钮
                llm_btn = gr.Button("Chat with llm")
            # 创建一行按钮用于清空操作
            with gr.Row():
                # "清空聊天"按钮
                clear = gr.Button("清空聊天")
                # 将 clear 按钮的 click 事件绑定到 clear_all 函数
                # outputs=[msg, chatbot] 指定 clear_all 函数的返回值更新 msg 和 chatbot 组件
                clear.click(clear_all, outputs=[msg, chatbot])

        # 右侧列：配置选项
        with gr.Column(scale=1): # 创建一个列布局，占据较小空间 (scale=1)
            # 创建 File 组件用于上传知识库文件/目录
            # 注意：Gradio 的 File 组件可能处理单个文件或目录，具体行为取决于后端实现
            # file_types 限制了可选的文件类型
            file = gr.File(
                label='请选择知识库目录/文件', # 组件标签
                file_types=['.txt', '.md', '.docx', '.pdf'] # 允许的文件扩展名
            )
            # 创建一行用于放置知识库初始化按钮
            with gr.Row():
                # "知识库文件向量化"按钮
                init_db = gr.Button("知识库文件向量化")
    with gr.Row():
        # 使用 gr.Group 替代 gr.Accordion 来组织参数配置
                with gr.Group(): # 使用 Group 组件替代 Accordion
                    gr.Markdown("### 参数配置") # 添加一个 Markdown 标题代替 Accordion 的标签
                    # 使用 gr.Row() 将滑块放在同一行
                    with gr.Row():
                        # 温度参数滑块
                        temperature = gr.Slider(
                            minimum=0, maximum=1, value=0.01, step=0.01, # 范围、默认值、步长
                            label="llm temperature", interactive=True # 标签、允许交互
                        )
                        # 向量数据库检索 Top K 滑块
                        top_k = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1, # 范围、默认值、步长
                            label="vector db top_k", interactive=True # 标签、允许交互
                        )
                        # 添加 history_len 滑块
                        history_len = gr.Slider(
                            minimum=0, maximum=10, value=3, step=1,
                            label="history length", interactive=True
                        )
                        # 添加 max_tokens 滑块
                        max_tokens = gr.Slider(
                            minimum=1, maximum=8192, value=2048, step=1, # 根据需要调整最大值
                            label="max tokens", interactive=True
                        )
                    # 使用 gr.Row() 将下拉菜单放在同一行
                    with gr.Row():
                        # LLM 模型选择下拉菜单
                        llm = gr.Dropdown(
                            LLM_MODEL_LIST, # 选项列表
                            label="large language model", # 标签
                            value=INIT_LLM, # 默认值
                            interactive=True # 允许交互
                        )
                        # Embedding 模型选择下拉菜单
                        embedding = gr.Dropdown(
                            EMBEDDING_MODEL_LIST, # 选项列表
                            label="embedding model", # 标签
                            value=INIT_EMBEDDING_MODEL, # 默认值
                            interactive=True # 允许交互
                        )

    # --- 事件绑定 ---
    # 将按钮点击事件绑定到对应的处理函数
    # 注意：确保 inputs 列表中的组件顺序与函数参数顺序一致
    # 使用包装函数来添加日志
    db_with_his_btn.click(
        wrap_chat_qa, # 使用包装函数
        inputs=[msg, chatbot, llm, embedding, temperature, top_k],
        outputs=[msg, chatbot] # 更新输入框和聊天机器人
    )
    db_wo_his_btn.click(
        wrap_qa, # 使用包装函数
        inputs=[msg, chatbot, llm, embedding, temperature, top_k],
        outputs=[msg, chatbot] # 更新输入框和聊天机器人
    )
    llm_btn.click(
        respond, # respond 函数内部已添加日志
        inputs=[msg, chatbot, llm, history_len, temperature, max_tokens], # 确保 history_len 和 max_tokens 在 inputs 中
        outputs=[msg, chatbot] # 更新输入框和聊天机器人
    )

    # 知识库初始化按钮事件绑定
    # 确保你的 .click() 调用是这样的结构：
    # 将 handle_init_db 函数绑定到 init_db 按钮的点击事件
    # inputs 列表中的组件按顺序对应 handle_init_db 函数的参数 (file_obj, embedding_model)
    # outputs 指定函数返回值要更新的组件
    print("init_db.click begin...") 
    init_db.click(
    fn=handle_init_db,                      # 要调用的函数
    inputs=[file, embedding],               # 输入：文件和 embedding 模型
    outputs=[msg]                        # 输出：状态消息到输入框
    )
    print("init_db.click end...") 

# 启动 Gradio 应用
# demo.queue().launch(...) # queue() 启用队列处理并发请求
# share=True 生成公共链接，enable_queue=True 启用队列
demo.queue().launch(share=True, inbrowser=True, server_name="127.0.0.1", server_port=7860)

# 在脚本结束时，显式关闭所有可能存在的 Gradio 实例，有助于释放端口和资源
gr.close_all()
# 启动 Gradio 应用
# share=True 会生成一个公开的链接（需要 Gradio 的联网服务），方便分享给他人临时访问
# 如果在本地运行，可以去掉 share=True
print("Launching Gradio interface...")
demo.launch(share=True)