from langchain.prompts import PromptTemplate  # 导入 PromptTemplate，用于自定义大模型的提示词模板
from langchain.chains import RetrievalQA  # 导入 RetrievalQA，用于实现检索增强型问答链
from langchain_community.vectorstores import Chroma  # 导入 Chroma，作为向量数据库存储和检索文本向量
from langchain.chains import ConversationalRetrievalChain  # 导入 ConversationalRetrievalChain，实现带历史对话的检索问答链
from langchain.memory import ConversationBufferMemory  # 导入 ConversationBufferMemory，用于存储对话历史
from langchain.chat_models import ChatOpenAI  # 导入 ChatOpenAI，调用 OpenAI 的对话模型
import sys  # 导入 sys 模块，用于操作 Python 运行环境
sys.path.append('../')  # 将自定义项目路径加入模块搜索路径，便于后续自定义模块导入
from qa_chain.model_to_llm import model_to_llm  # 导入 model_to_llm，将模型参数转换为具体 LLM 实例
from qa_chain.get_vectordb import get_vectordb  # 导入 get_vectordb，用于获取和初始化向量数据库对象
import re  # 导入正则表达式模块，用于文本处理（如换行符替换）
from langchain.schema import HumanMessage, AIMessage # 导入消息类型

class Chat_QA_chain_self:
    """
    带历史记录的问答链类
    主要功能：支持多种大模型和 embedding，结合历史对话，实现基于知识库的多轮问答。
    参数说明：
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embedding：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    """
    def __init__(self, model: str, temperature: float = 0.0, top_k: int = 4, chat_history: list = [], file_path: str = None, persist_path: str = None, appid: str = None, api_key: str = None, Spark_api_secret: str = None, Wenxin_secret_key: str = None, embedding="openai", embedding_key: str = None):
        self.model = model  # 保存模型名称
        self.temperature = temperature  # 保存温度系数
        self.top_k = top_k  # 保存检索返回的文档数
        # self.chat_history 仍然保留，用于外部（如Gradio）交互
        self.chat_history = list(chat_history) # 确保是列表副本
        self.file_path = file_path  # 知识库文件路径
        self.persist_path = persist_path  # 向量数据库持久化路径
        self.appid = appid  # 星火 APPID
        self.api_key = api_key  # API Key
        self.Spark_api_secret = Spark_api_secret  # 星火 API Secret
        self.Wenxin_secret_key = Wenxin_secret_key  # 文心 Secret Key
        self.embedding = embedding  # embedding 模型名称
        self.embedding_key = embedding_key  # embedding 模型的 key

        # 初始化向量数据库对象
        print("Chat_QA_chain_self 开始初始化向量数据库...")
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)
        print("Chat_QA_chain_self 初始化向量数据库完成！")

        # 初始化对话内存
        print("Chat_QA_chain_self 开始初始化对话内存...")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer' # 新增：明确指定 AI 回答对应的键
        )
        # 将传入的 chat_history 加载到内存中
        for human_msg, ai_msg in self.chat_history:
             self.memory.chat_memory.add_messages([
                 HumanMessage(content=human_msg),
                 AIMessage(content=ai_msg)
             ])
        print("Chat_QA_chain_self 初始化对话内存完成！")


    def clear_history(self):
        """
        清空历史记录，包括 self.chat_history 列表和内部 memory
        """
        self.chat_history.clear()
        self.memory.clear() # 清空 LangChain memory
        print("历史记录已清空") # 添加日志确认

    def change_history_length(self, history_len: int = 1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)  # 获取历史对话总数
        return self.chat_history[n - history_len:]  # 返回最近 history_len 条历史

    def answer(self, question: str = None, temperature=None, top_k=4):
        """
        核心方法，调用问答链，结合历史对话和知识库进行问答
        参数:
        - question：用户提问
        - temperature：生成温度（可选，默认用 self.temperature）
        - top_k：检索返回的文档数（可选，默认4）
        返回：
        - 更新后的历史对话列表 (self.chat_history)
        """
        if not question:
            print("问题不能为空！")
            return self.chat_history

        if temperature is None:
            temperature = self.temperature

        print(f"\n--- 回合开始 (问题: '{question}') ---")
        print("Chat_QA_chain_self 开始初始化 LLM...")
        llm = model_to_llm(self.model, temperature, self.appid, self.api_key, self.Spark_api_secret, self.Wenxin_secret_key)
        print("Chat_QA_chain_self 初始化 LLM 完成！")

        retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': top_k})

        # 构建带历史的检索问答链，使用 self.memory
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory, # 传入 memory 对象
            return_source_documents=True
        )
        print("Chat_QA_chain_self 开始调用问答链...")
        # 调用问答链，只需要传入问题，历史由 memory 管理
        # 注意：链内部会自动从 memory 加载 chat_history
        result = qa({"question": question}) # 不再传递 chat_history
        answer = result['answer']
        source_documents = result.get("source_documents", [])

        print("\n--- 源文档片段 ---")
        for i, doc in enumerate(source_documents):
            print(f"片段 {i+1}:\n{doc.page_content}\n---")

        answer_cleaned = answer

        # 更新外部交互用的 chat_history 列表
        self.chat_history.append((question, answer_cleaned))

        # memory 会由 ConversationalRetrievalChain 自动更新

        print(f"--- 回合结束 (回答: '{answer_cleaned}') ---")
        return self.chat_history # 返回更新后的外部历史记录列表

# --- 添加 main 测试块 ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv

    # 加载 .env 文件中的环境变量
    _ = load_dotenv(find_dotenv())

    # --- 配置测试参数 ---
    # !! 请确保你的向量数据库已经建立，并且路径正确 !!
    persist_path = "../vector_db" # 指定你的 Chroma 向量数据库持久化路径
    # 选择要测试的大模型
    # test_model = "gpt-3.5"
    test_model = "deepseek-chat" # 或者 "glm-4-flash-250414", "ERNIE 4.5", "Spark Lite" 等
    # 选择 embedding 模型 (需要与建库时使用的模型一致)
    test_embedding = "m3e" # 或者 "openai", "bge-large-zh-v1.5" 等

    # 定义多轮对话的问题
    test_questions = [
        "Langchain 是什么？",
        "它有哪些主要模块？",
        "这些模块如何协同工作？" # 这是一个依赖于前两轮对话的问题
    ]

    print(f"--- 开始测试 Chat_QA_chain_self (带历史记录) ---")
    print(f"使用模型: {test_model}")
    print(f"使用 Embedding: {test_embedding}")
    print(f"向量数据库路径: {persist_path}")
    print("-" * 40)

    try:
        # 实例化 Chat_QA_chain_self，初始历史记录为空列表
        chat_qa_chain_instance = Chat_QA_chain_self(
            model=test_model,
            persist_path=persist_path,
            embedding=test_embedding,
            chat_history=[] # 显式传递空列表作为初始历史
            # API Keys 会由内部函数从环境变量加载
        )

        print("\nChat_QA_chain_self 实例创建成功，开始多轮对话测试...")

        # 模拟多轮对话
        current_history = []
        for i, question in enumerate(test_questions):
            print(f"\n===== 第 {i+1} 轮对话 =====")
            # 调用 answer 方法，它会使用实例内部的 chat_history 并更新它
            current_history = chat_qa_chain_instance.answer(question=question)

            print("\n--- 当前完整对话历史 ---")
            for turn_q, turn_a in current_history:
                # 为了终端显示清晰，将 <br/> 替换回换行符
                turn_a_display = turn_a.replace('<br/>', '\n')
                print(f"用户: {turn_q}")
                print(f"模型: {turn_a_display}")
                print("-" * 20)
            print("=" * 25)

    except FileNotFoundError:
        print(f"错误：找不到向量数据库路径 '{persist_path}'。请确保路径正确且数据库已存在。")
    except ImportError as ie:
         print(f"错误：缺少必要的库。请确保已安装所有依赖。错误信息: {ie}")
    except ValueError as ve:
        print(f"错误：配置或参数错误。可能是缺少 API Key 或模型名称/路径不正确。错误信息: {ve}")
    except Exception as e:
        print(f"测试过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()

    print("--- 测试结束 ---")