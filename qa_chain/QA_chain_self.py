from langchain.prompts import PromptTemplate  # 导入 PromptTemplate，用于自定义大模型的提示词模板
from langchain.chains import RetrievalQA      # 导入 RetrievalQA，实现检索增强型问答链
from langchain_community.vectorstores import Chroma     # 导入 Chroma，作为向量数据库存储和检索文本向量
import sys                                    # 导入 sys，用于操作 Python 运行环境
sys.path.append("../")                        # 添加上级目录到模块搜索路径，便于导入自定义模块
from qa_chain.model_to_llm import model_to_llm    # 导入模型适配函数，将参数转换为具体 LLM 实例
from qa_chain.get_vectordb import get_vectordb    # 导入向量数据库获取函数
import sys                                    # 冗余导入，可去除
import re                                     # 导入正则表达式模块，用于文本处理

class QA_chain_self():
    """
    不带历史记录的问答链
    主要功能：支持多种大模型和 embedding，结合知识库实现单轮问答。
    参数说明：
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火需要输入
    - api_key：所有模型都需要
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型  
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板 default_template_rq
    """
    # 基于召回结果和 query 结合起来构建的 prompt 使用的默认提示模版
    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说"谢谢你的提问！"。
    {context}
    问题: {question}
    有用的回答:"""

    def __init__(self, model: str, temperature: float = 0.0, top_k: int = 4,  file_path: str = None, persist_path: str = None, appid: str = None, api_key: str = None, Spark_api_secret: str = None, Wenxin_secret_key: str = None, embedding="openai",  embedding_key=None, template=default_template_rq):
        """
        初始化 QA_chain_self 实例，配置大模型、向量数据库、检索器和 QA 链。
        """
        self.model = model  # 保存模型名称
        self.temperature = temperature  # 保存温度系数
        self.top_k = top_k  # 保存检索返回的文档数
        self.file_path = file_path  # 知识库文件路径
        self.persist_path = persist_path  # 向量数据库持久化路径
        self.appid = appid  # 星火 APPID
        self.api_key = api_key  # API Key
        self.Spark_api_secret = Spark_api_secret  # 星火 API Secret
        self.Wenxin_secret_key = Wenxin_secret_key  # 文心 Secret Key
        self.embedding = embedding  # embedding 模型名称
        self.embedding_key = embedding_key  # embedding 模型的 key
        self.template = template  # QA prompt 模板

        # 初始化向量数据库对象
        print("QA_chain_self 开始初始化向量数据库...")
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)
        print("QA_chain_self 向量数据库初始化完成。")
        # 初始化大模型对象
        print("QA_chain_self 开始初始化大模型...")
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret, self.Wenxin_secret_key)
        print("QA_chain_self 大模型初始化完成。")
        # 构建 QA prompt 模板
        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=self.template)
        # 构建检索器，指定检索方式和返回文档数
        self.retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={'k': self.top_k})  # 默认 similarity 检索，k=4
        # 自定义 QA 链，结合大模型、检索器和 prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
        )

    # 基于大模型的问答 prompt 使用的默认提示模版
    # default_template_llm = """请回答下列问题:{question}"""

    def answer(self, question: str = None, temperature=None, top_k=4):
        """
        核心方法，调用问答链，结合知识库进行问答
        参数:
        - question：用户提问
        - temperature：生成温度（可选，默认用 self.temperature）
        - top_k：检索返回的文档数（可选，默认4）
        返回：
        - answer：大模型生成的答案
        """
        if len(question) == 0:  # 如果问题为空，直接返回空字符串
            return ""

        if temperature is None:
            temperature = self.temperature  # 如果未指定温度，使用默认温度

        if top_k is None:
            top_k = self.top_k  # 如果未指定 top_k，使用默认 top_k

        # 调用 QA 链，传入用户问题、温度和 top_k
        # 注意：Langchain 的 RetrievalQA 的 __call__ 方法（或其封装的 invoke 等）
        # 通常只接受 'query' 作为主要输入。temperature 和 top_k 通常在链初始化时设置，
        # 或者通过修改 llm 或 retriever 的属性来动态调整。
        # 直接将 temperature 和 top_k 传入 __call__ 可能不会生效，取决于具体实现。
        # 这里保持原样，但需要注意 Langchain 的实际用法。
        # 更好的做法是在调用前修改 self.llm.temperature 或 self.retriever.search_kwargs['k']
        # self.llm.temperature = temperature # 示例：如果 llm 对象允许这样设置
        # self.retriever.search_kwargs['k'] = top_k # 示例：修改检索器参数
        print("QA_chain_self 开始调用 QA 链...")
        result = self.qa_chain({"query": question}) # 通常只传递 query
        answer = result["result"]  # 获取模型回答
        source_documents = result.get("source_documents", []) # 获取源文档（如果返回）

        print("\n--- 源文档片段 ---")
        for i, doc in enumerate(source_documents):
            print(f"片段 {i+1}:\n{doc.page_content}\n---")

        answer = re.sub(r"\\n", '<br/>', answer)  # 将换行符替换为 HTML 换行
        return answer  # 返回最终答案

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
    # 测试问题
    test_question = "请介绍一下 Langchain 是什么？" # 替换成你想问的、与知识库内容相关的问题

    print(f"--- 开始测试 QA_chain_self ---")
    print(f"使用模型: {test_model}")
    print(f"使用 Embedding: {test_embedding}")
    print(f"向量数据库路径: {persist_path}")
    print(f"测试问题: {test_question}")
    print("-" * 30)

    try:
        # 获取必要的 API Keys (根据所选模型和 embedding 从环境变量读取)
        # 注意：model_to_llm 和 get_vectordb 内部会尝试读取所需的环境变量
        # 这里可以预先检查，但不是必须
        api_key = os.getenv("deepseek_api_key") # 示例：根据 test_model 获取对应 key
        embedding_key = os.getenv("M3E_API_KEY") # 示例：根据 test_embedding 获取对应 key (如果需要)
        # 根据实际使用的模型和 embedding 添加其他 key 的获取逻辑
        # if test_model == "gpt-3.5": api_key = os.getenv("OPENAI_API_KEY")
        # if test_embedding == "openai": embedding_key = os.getenv("OPENAI_API_KEY")

        # 实例化 QA_chain_self
        # 注意：这里没有传递所有可能的 key，依赖于 model_to_llm 和 get_vectordb 内部的逻辑
        qa_chain_instance = QA_chain_self(
            model=test_model,
            persist_path=persist_path,
            embedding=test_embedding,
            # api_key=api_key, # 可以显式传递，但通常由内部函数处理
            # embedding_key=embedding_key # 可以显式传递
        )

        print("\nQA_chain_self 实例创建成功，开始调用 answer 方法...")

        # 调用 answer 方法
        answer_result = qa_chain_instance.answer(question=test_question)

        print("\n--- 模型回答 ---")
        # 打印原始答案（包含 <br/> 标签）
        # print(answer_result)
        # 打印替换回换行符的答案，更易读
        print(answer_result.replace('<br/>', '\n'))
        print("-" * 30)

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
