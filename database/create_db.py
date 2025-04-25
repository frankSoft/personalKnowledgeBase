'''
向量数据库创建与管理模块
提供文件加载、文档切分、向量化存储等功能
支持多种文件格式(PDF、Markdown、纯文本、Word)的处理
实现本地向量数据库的创建、持久化和加载
'''
try:
    import pymupdf
    print("--- PyMuPDF imported successfully by script! ---")
except ImportError as e:
    print(f"--- Failed to import PyMuPDF directly: {e} ---")
import os  # 导入os模块，用于操作系统相关功能，如文件路径处理
import sys  # 导入sys模块，用于访问与Python解释器相关的变量和函数，如此处的路径操作
import re  # 导入re模块，用于正则表达式操作，如在文件名中搜索特定模式
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 将项目的上级目录添加到Python模块搜索路径中，便于导入项目内其他模块
import tempfile  # 导入tempfile模块，用于创建临时文件和目录
from dotenv import load_dotenv, find_dotenv  # 从dotenv库导入函数，用于加载.env文件中的环境变量
from embedding.call_embedding import get_embedding  # 从自定义模块导入获取embedding模型的函数
from langchain_community.document_loaders import UnstructuredFileLoader  # 导入Langchain的非结构化文件加载器，用于加载纯文本等
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # 导入Langchain的Markdown文件加载器
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入Langchain的递归字符文本分割器，用于将长文档切分成小块
from langchain_community.document_loaders import PyMuPDFLoader  # 导入Langchain的PyMuPDF加载器，用于加载PDF文件
from langchain_community.vectorstores import Chroma  # 导入Langchain的Chroma向量数据库接口
from langchain_community.document_loaders import Docx2txtLoader  # 导入Langchain的Docx2txt加载器，用于加载Word(.docx)文件

# --- 基本配置 ---
DEFAULT_DB_PATH = "../knowledge_db"  # 定义默认的原始知识库文件存放目录路径
DEFAULT_PERSIST_PATH = "../vector_db"  # 定义默认的向量数据库持久化存储目录路径

def get_files(dir_path):
    """
    遍历指定目录，递归获取该目录下所有文件的完整路径列表。
    参数:
        dir_path (str): 需要遍历的目录路径。
    返回:
        file_list (list): 包含所有文件完整路径的列表。
    """
    file_list = []  # 初始化文件列表
    # os.walk会递归遍历目录，返回(当前路径, 子目录列表, 文件列表)
    for filepath, dirnames, filenames in os.walk(dir_path):
        # 遍历当前路径下的所有文件名
        for filename in filenames:
            # 构造文件的完整路径并添加到列表中
            file_list.append(os.path.join(filepath, filename))
    return file_list  # 返回收集到的文件路径列表

def file_loader(file, loaders):
    """
    根据文件扩展名选择合适的加载器加载文件内容，支持pdf、md、txt、docx格式。
    如果输入是目录，则递归加载目录下的所有文件。
    参数:
        file (str or tempfile._TemporaryFileWrapper): 文件路径或临时文件对象。
        loaders (list): 用于收集已初始化的Langchain文档加载器对象的列表。
    """
    # 检查输入是否为tempfile创建的临时文件对象
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name  # 获取临时文件的实际路径

    # 检查输入路径是否为目录
    if not os.path.isfile(file):
        try:
            # 如果是目录，则递归调用file_loader处理目录下的每个文件/子目录
            [file_loader(os.path.join(file, f), loaders) for f in os.listdir(file)]
        except Exception as e:
            # 打印读取目录时发生的错误
            print(f"读取目录 {file} 出错: {str(e)}")
        return # 处理完目录后直接返回

    # 尝试加载单个文件
    try:
        # 获取文件扩展名，并转换为小写以统一处理
        file_type = file.split('.')[-1].lower()
        # 根据文件类型选择加载器
        if file_type == 'pdf':
            try:
                print(f"加载PDF文件: {file}")
                # 使用PyMuPDFLoader加载PDF文件
                loaders.append(PyMuPDFLoader(file))
            except Exception as pdf_error:
                # 打印并重新抛出PDF加载错误
                print(f"加载PDF文件 {file} 出错: {str(pdf_error)}")
                raise Exception(f"PDF文件加载失败: {str(pdf_error)}")
        elif file_type == 'md':
            # 定义需要排除的文件名模式（包含"不存在"或"风控"）
            pattern = r"不存在|风控"
            # 在文件名中搜索模式
            match = re.search(pattern, file)
            # 如果文件名不匹配排除模式
            if not match:
                try:
                    print(f"加载Markdown文件: {file}")
                    # 使用UnstructuredMarkdownLoader加载Markdown文件
                    loaders.append(UnstructuredMarkdownLoader(file))
                except Exception as md_error:
                    # 打印Markdown加载错误
                    print(f"加载Markdown文件 {file} 出错: {str(md_error)}")
            else:
                print(f"跳过Markdown文件（匹配排除模式）: {file}")
        elif file_type == 'txt':
            try:
                print(f"加载TXT文件: {file}")
                # 使用UnstructuredFileLoader加载TXT文件
                loaders.append(UnstructuredFileLoader(file))
            except Exception as txt_error:
                # 打印TXT加载错误
                print(f"加载TXT文件 {file} 出错: {str(txt_error)}")
        elif file_type == 'docx':
            try:
                print(f"加载DOCX文件: {file}")
                # 使用Docx2txtLoader加载DOCX文件
                loaders.append(Docx2txtLoader(file))
            except Exception as docx_error:
                # 打印DOCX加载错误
                print(f"加载DOCX文件 {file} 出错: {str(docx_error)}")
        else:
            # 如果文件类型不支持，则打印提示信息
            print(f"不支持的文件类型: {file_type}，文件: {file}")
        # 此处可以继续添加对其他文件类型的支持，例如：
        # elif file_type == 'csv':
        #     loaders.append(CSVLoader(file))
    except Exception as e:
        # 捕获并打印加载过程中可能出现的任何其他异常
        print(f"加载文件 {file} 出错: {str(e)}")
    return # 函数结束

def create_db_info(files=DEFAULT_DB_PATH, embeddings="m3e", persist_directory=DEFAULT_PERSIST_PATH):
    """
    创建向量数据库的主入口函数。负责协调文件加载、向量化和存储过程。
    参数:
        files (str or list): 知识库文件或目录的路径（单个路径或路径列表）。默认为DEFAULT_DB_PATH。
        embeddings (str): 指定使用的embedding模型名称（如 'openai', 'm3e', 'zhipuai'）。默认为 'm3e'。
        persist_directory (str): 向量数据库持久化存储的目录路径。默认为DEFAULT_PERSIST_PATH。
    返回:
        str: 处理结果信息字符串，指示成功或具体的错误信息。
    """
    try:
        print("create_db_info 开始知识库文件向量化...") # 打印开始处理的日志信息
        # 检查是否传入了文件或目录路径
        if files is None:
            return "错误：未选择任何文件" # 如果没有文件，返回错误信息

        # 确保向量数据库持久化目录存在，如果不存在则创建
        os.makedirs(persist_directory, exist_ok=True)
        # 确保默认持久化路径的上级目录也存在（虽然上一步通常已包含，但为了更健壮）
        os.makedirs(os.path.dirname(DEFAULT_PERSIST_PATH), exist_ok=True)

        # 检查指定的embedding模型是否受支持
        if embeddings == 'openai' or embeddings == 'm3e' or embeddings == 'zhipuai':
            # 调用核心的create_db函数来创建数据库
            vectordb = create_db(files, persist_directory, embeddings)
            print(f"调用核心的create_db函数完成创建数据库... {len(vectordb)}") # 打印开始创建数据库的日志信息
            # 检查create_db的返回值是否为字符串（表示发生了错误）
            if isinstance(vectordb, str):
                # 特定错误处理：无法加载空文件
                if "can't load empty file" in vectordb:
                    return "错误：无法加载空文件"
                else:
                    # 返回其他类型的错误信息
                    return f"错误：{vectordb}"
        else:
            # 如果embedding模型名称无效
            return f"错误：不支持的embedding模型 '{embeddings}'"

        print("create_db_info 结束知识库文件向量化...") # 打印处理完成的日志信息
        return "知识库文件向量化成功！" # 返回成功信息
    except Exception as e:
        # 捕获在整个过程中可能发生的任何未预料的异常
        error_msg = f"知识库文件向量化失败: {str(e)}"
        print(error_msg) # 打印错误信息
        return error_msg # 返回错误信息

def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="m3e"):
    """
    核心函数：加载知识库文件，进行文本切分，生成文档的嵌入向量，
    然后使用Chroma创建向量数据库并将其持久化到磁盘。
    参数:
        files (str or list): 知识库文件或目录的路径（单个路径或路径列表）。
        persist_directory (str): 向量数据库持久化存储的目录路径。
        embeddings (str or object): embedding模型的名称或已初始化的embedding对象。
    返回:
        Chroma or str: 成功时返回创建的Chroma向量数据库对象，失败时返回错误信息字符串。
    """
    # 检查输入文件列表是否为None
    if files is None:
        return "can't load empty file" # 返回特定错误信息
    # 如果输入files不是列表，将其转换为单元素列表，以统一处理
    if not isinstance(files, list):
        files = [files]

    # 再次确保持久化目录存在
    # persist_directory = './vector_db/chroma' # 这行被注释掉了，使用函数参数传入的路径
    os.makedirs(persist_directory, exist_ok=True)
    print(f"向量数据库持久化目录: {persist_directory}")

    try:
        loaders = [] # 初始化加载器列表
        # 遍历文件/目录列表，调用file_loader将初始化好的加载器添加到loaders列表中
        [file_loader(file, loaders) for file in files]

        # 检查是否有成功初始化的加载器
        if len(loaders) == 0:
            return "未找到可处理的文件，请确认文件格式是否支持(pdf, md, txt, docx)" # 如果没有加载器，返回错误

        docs = [] # 初始化文档列表
        # 遍历所有加载器
        for loader in loaders:
            # 确保加载器不是None（虽然理论上file_loader不会添加None，但作为防御性编程）
            if loader is not None:
                try:
                    # 调用加载器的load方法加载文档内容
                    loaded_docs = loader.load()
                    print(f"加载到的文档: {loaded_docs}")
                    # 检查加载到的文档是否非空
                    if loaded_docs:
                        # 将加载到的文档添加到总文档列表中
                        docs.extend(loaded_docs)
                    else:
                        # 如果加载器返回空文档，打印警告信息
                        print(f"警告：加载器 {loader} 返回了空文档")
                except Exception as doc_error:
                    # 捕获并打印加载具体文档内容时发生的错误
                    print(f"加载文档内容出错: {str(doc_error)}")

        # 检查是否成功加载到任何文档内容
        print(f"加载到的文档数量: {len(docs)}")
        if len(docs) == 0:
            return "文档内容为空，无法创建向量数据库" # 如果没有文档内容，返回错误

        # 初始化文本分割器，设置块大小和重叠大小
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=150)
        # 使用分割器将加载的文档分割成更小的块
        split_docs = text_splitter.split_documents(docs)

        # 检查分割后的文档块列表是否为空
        print(f"分割后的文档块列表: {len(split_docs)}")
        if len(split_docs) == 0:
            return "文档切分后内容为空，无法创建向量数据库" # 如果分割后没有内容，返回错误

        # 获取embedding对象
        # 如果传入的embeddings参数是字符串（模型名称）
        if isinstance(embeddings, str):
            try:
                # 调用get_embedding函数获取对应的embedding模型对象
                embeddings = get_embedding(embedding=embeddings)
                print(f"获取的embedding模型对象: {embeddings}")
            except Exception as emb_error:
                # 如果获取embedding模型失败，返回错误信息
                return f"获取Embedding模型失败: {str(emb_error)}"
        # 如果传入的已经是embedding对象，则直接使用

        # 使用分割后的文档和embedding模型创建Chroma向量数据库
        print(f"分割后的文档 split_docs: {len(split_docs)}")
        print(f"embedding模型对象 embeddings: {type(embeddings)}")
        print(f"持久化目录 persist_directory: {persist_directory}")
        vectordb = Chroma.from_documents(
            documents=split_docs, # 传入分割后的文档块
            embedding=embeddings, # 传入embedding模型对象
            persist_directory=persist_directory  # 指定持久化目录
        )
        print(f"向量数据库类型vectordb: {type(vectordb)}")
        vectordb.persist()  # 将创建的向量数据库立即持久化到磁盘
        print(f"向量数据库类型 vectordb.persist(): {type(vectordb.persist())}")
        return vectordb # 成功创建并持久化后，返回Chroma数据库对象
    except Exception as e:
        # 捕获在创建数据库过程中可能发生的任何异常
        error_msg = f"创建向量数据库失败: {str(e)}"
        print(error_msg) # 打印错误信息
        return error_msg # 返回错误信息

# 注意：函数名 'presit_knowledge_db' 可能存在拼写错误，应为 'persist_knowledge_db'
def presit_knowledge_db(vectordb):
    """
    将给定的向量数据库对象持久化到磁盘。
    (注意：函数名可能存在拼写错误，建议修改为 persist_knowledge_db)
    参数:
        vectordb (Chroma): 需要持久化的Chroma向量数据库对象。
    """
    # 调用向量数据库对象的persist方法执行持久化操作
    vectordb.persist()

def load_knowledge_db(path, embeddings):
    """
    从磁盘加载之前已持久化的向量数据库。
    参数:
        path (str): 向量数据库持久化存储的目录路径。
        embeddings (object): 用于加载数据库的embedding模型对象（必须与创建时使用的兼容）。
    返回:
        Chroma: 加载的Chroma向量数据库对象。
    """
    # 初始化Chroma对象，指定持久化目录和embedding函数来加载数据库
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb # 返回加载后的向量数据库对象

# 当该脚本作为主程序直接运行时执行以下代码
if __name__ == "__main__":
    # 这是一个示例用法：
    # 调用create_db函数，使用默认的知识库路径(DEFAULT_DB_PATH)
    # 和默认的持久化路径(DEFAULT_PERSIST_PATH)，并指定使用"m3e" embedding模型。
    # 这会加载../knowledge_db目录下的文件，创建向量数据库，并将其存储在../vector_db目录下。
    create_db(embeddings="m3e")
    print("示例数据库创建（或尝试创建）完成。") # 打印完成信息
