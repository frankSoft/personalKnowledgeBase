'''
基于 DeepSeek 大模型自定义 LLM 类
提供 DeepSeek API 的封装，支持 DeepSeek 模型的调用
继承自 Self_LLM 基类，实现 DeepSeek 特有的调用逻辑（通过 OpenAI 兼容接口）
'''
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from llm.self_llm import Self_LLM
import openai # 导入 openai 库用于 API 调用
from langchain.callbacks.manager import CallbackManagerForLLMRun
# 导入 API key 解析工具函数 (如果需要自动从环境变量加载)
# from llm.call_llm import parse_llm_api_key

# DeepSeek API 端点
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

class DeepSeek_LLM(Self_LLM):
    """
    DeepSeek 大模型的自定义 LLM 类，继承自 Self_LLM。
    使用 OpenAI Python SDK 与 DeepSeek API (兼容 OpenAI 格式) 交互。
    """
    # 指定 DeepSeek API 的基础 URL
    api_base: str = DEEPSEEK_API_BASE
    # 默认模型设为 deepseek-chat
    model_name: str = "deepseek-chat"
    # DeepSeek API Key (可以从环境变量加载或在实例化时传入)
    api_key: Optional[str] = None
    # 最大 token 数
    max_tokens: int = 2048

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> str:
        """
        LLM 主调用方法，负责将 prompt 发送到 DeepSeek 服务并返回结果。
        参数:
            prompt: 用户输入的 prompt
            stop: 停止词 (目前 DeepSeek API 可能不支持此参数，但保留以兼容 LangChain)
            run_manager: 回调管理器
            kwargs: 其他参数 (可以覆盖默认参数，如 temperature, max_tokens)
        返回:
            DeepSeek 大模型的回复内容
        """
        # 检查 API Key 是否存在
        if not self.api_key:
            # 可以尝试从环境变量加载，如果需要的话
            # self.api_key = parse_llm_api_key("deepseek")
            # 如果仍然没有 key，则抛出错误
            if not self.api_key:
                raise ValueError("DeepSeek API Key not provided.")

        # 保存当前的 openai api_base 和 api_key，以便之后恢复
        original_api_base = openai.api_base
        original_api_key = openai.api_key

        try:
            # 设置 DeepSeek 的 API Key 和 Base URL
            openai.api_key = self.api_key
            openai.api_base = self.api_base

            # 准备发送给模型的消息体
            messages = [{"role": "user", "content": prompt}]

            # 合并默认参数和调用时传入的参数
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "request_timeout": self.request_timeout,
                **self.model_kwargs, # 添加 Self_LLM 中的 model_kwargs
                **kwargs,           # 允许调用时覆盖参数
            }
            # 移除值为 None 的参数，避免 API 报错
            params = {k: v for k, v in params.items() if v is not None}

            # 调用 OpenAI 兼容的 ChatCompletion 接口
            response = openai.ChatCompletion.create(**params)

            # 提取并返回模型生成的回复内容
            return response.choices[0].message["content"]

        except Exception as e:
            # 捕获并处理 API 调用过程中的错误
            print(f"调用 DeepSeek API 时出错: {e}")
            # 可以选择返回错误信息或重新抛出异常
            # return f"调用 DeepSeek API 失败: {e}"
            raise e # 重新抛出异常，让上层处理

        finally:
            # 恢复原始的 openai api_base 和 api_key
            openai.api_base = original_api_base
            openai.api_key = original_api_key

    @property
    def _llm_type(self) -> str:
        """
        返回 LLM 类型标识
        """
        return "DeepSeek"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        获取模型的标识参数，用于日志或调试。
        """
        # 合并基础参数和 DeepSeek 特有的 api_base
        return {
            **super()._identifying_params,
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
        }

# --- 添加 main 测试块 ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv # 假设你使用 python-dotenv 管理环境变量

    # 加载 .env 文件中的环境变量 (如果存在)
    load_dotenv()

    # 从环境变量获取 API Key
    api_key = os.getenv("deepseek_api_key")

    if not api_key:
        print("错误：请设置 deepseek_api_key 环境变量。")
    else:
        try:
            # 实例化 DeepSeek_LLM
            print("正在初始化 DeepSeek_LLM...")
            llm = DeepSeek_LLM(api_key=api_key)

            # 定义测试 prompt
            test_prompt = "你好，请介绍一下你自己。"
            print(f"发送 Prompt: {test_prompt}")

            # 调用 LLM (使用 LangChain 标准的 invoke 方法)
            response = llm.invoke(test_prompt)

            # 打印结果
            print("\n模型回复:")
            print(response)

        except ValueError as ve:
            print(f"配置错误: {ve}")
        except Exception as e:
            print(f"调用 DeepSeek API 时发生错误: {e}")
            # 可以打印更详细的错误信息
            # import traceback
            # traceback.print_exc()