"""
DeepSeek API集成 - 提供与DeepSeek API交互的功能
"""

import os
from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from openai import OpenAI
import dotenv

# 加载环境变量
dotenv.load_dotenv()

# 从环境变量读取 DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 从环境变量读取 DashScope 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 设置环境变量以供 OpenAI 库使用
if DEEPSEEK_API_KEY:
    os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
    os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL
if DASHSCOPE_API_KEY:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    os.environ["DASHSCOPE_API_BASE"] = DASHSCOPE_BASE_URL

# 导入必要的库
from langchain_openai import ChatOpenAI

class DeepSeekChatWrapper(BaseChatModel):
    """DeepSeek聊天模型包装器"""
    
    model_name: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.3)
    _model: Any = None
    
    def __init__(self, **kwargs):
        # 移除空tools数组，避免API错误
        if 'tools' in kwargs and (not kwargs['tools'] or len(kwargs['tools']) == 0):
            del kwargs['tools']
            
        super().__init__(**kwargs)
        self._model = ChatOpenAI(
            model=self.model_name,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=self.temperature,
            **kwargs
        )
        
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """调用DeepSeek生成回复"""
        # 移除空tools数组，避免API错误
        if 'tools' in kwargs and (not kwargs['tools'] or len(kwargs['tools']) == 0):
            del kwargs['tools']
            
        result = self._model.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=result)])
    
    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> ChatResult:
        """异步调用DeepSeek生成回复"""
        # 移除空tools数组，避免API错误
        if 'tools' in kwargs and (not kwargs['tools'] or len(kwargs['tools']) == 0):
            del kwargs['tools']
            
        result = await self._model.ainvoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=result)])
    
    def bind_tools(self, tools: List[Dict[str, Any]]):
        """绑定工具"""
        # 只在tools非空时绑定
        if tools and len(tools) > 0:
            return self._model.bind_tools(tools)
        return self
    
    def with_structured_output(self, output_schema):
        """设置结构化输出"""
        return self._model.with_structured_output(output_schema)
    
    @property
    def _llm_type(self) -> str:
        """返回模型类型"""
        return "deepseek"

class DashScopeEmbeddingsWrapper(Embeddings):
    """阿里云百炼嵌入模型包装器"""
    
    def __init__(self, model_name="text-embedding-v3", dimensions=1024, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
        print(f"使用阿里云百炼嵌入模型: {model_name}")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成嵌入"""
        if not texts:
            return []
            
        # 批处理以提高效率
        batch_size = 20  # 阿里云API可能有请求限制，使用较小的批次
        embeddings = []
        
        # 分批处理，避免API限制问题
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # 每个文本单独请求，避免超过token限制
                batch_embeddings = []
                for text in batch:
                    if not text.strip():  # 跳过空文本
                        batch_embeddings.append([0.0] * self.dimensions)
                        continue
                        
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=text,
                        dimensions=self.dimensions,
                        encoding_format="float"
                    )
                    # 提取嵌入向量
                    vector = response.data[0].embedding
                    batch_embeddings.append(vector)
                
                embeddings.extend(batch_embeddings)
                print(f"成功处理嵌入批次 {i//batch_size + 1}, 大小: {len(batch)}")
            except Exception as e:
                print(f"嵌入批次 {i//batch_size + 1} 出错: {e}")
                # 对于失败的批次，使用零向量
                embeddings.extend([[0.0] * self.dimensions] * len(batch))
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        if not text.strip():
            return [0.0] * self.dimensions
            
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"查询嵌入出错: {e}")
            return [0.0] * self.dimensions

class DeepSeekEmbeddingsWrapper(Embeddings):
    """DeepSeek嵌入模型包装器"""
    
    def __init__(self, model_name=None, **kwargs):
        super().__init__()
        # 几个适合中文的embedding模型选项
        chinese_embedding_models = {
            "default": "paraphrase-multilingual-MiniLM-L12-v2",  # 多语言支持，包括中文
            "chinese-general": "shibing624/text2vec-base-chinese",  # 专门针对中文优化
            "chinese-large": "GanymedeNil/text2vec-large-chinese",  # 更大的中文模型
            "chinese-new": "moka-ai/m3e-base",  # 新一代中文嵌入模型
            "chinese-light": "shibing624/text2vec-base-chinese-paraphrase"  # 更轻量级的中文模型
        }
        
        # 如果没有指定模型名称，使用针对中文优化的模型
        if not model_name or model_name not in chinese_embedding_models:
            model_name = "chinese-general"
        
        print(f"使用嵌入模型: {chinese_embedding_models[model_name]}")
        
        # 使用选定的嵌入模型
        self.model = SentenceTransformer(chinese_embedding_models[model_name])
        
        # 设置更高的max_seq_length以处理较长文本
        self.model.max_seq_length = 512
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成嵌入"""
        # 批处理以提高效率
        batch_size = 32
        embeddings = []
        
        # 分批处理，避免内存问题
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())
            
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        return self.model.encode(text, convert_to_tensor=False).tolist()

def get_chat_model(**kwargs):
    """获取DeepSeek聊天模型"""
    # 移除空tools数组，避免API错误
    if 'tools' in kwargs and (not kwargs['tools'] or len(kwargs['tools']) == 0):
        del kwargs['tools']
    return DeepSeekChatWrapper(**kwargs)

def get_embedding_model(model_name=None, **kwargs):
    """获取嵌入模型
    
    参数:
        model_name: 模型名称，可选值:
            - "default": 多语言模型
            - "chinese-general": 通用中文模型
            - "chinese-large": 大型中文模型
            - "chinese-new": 新一代中文模型
            - "chinese-light": 轻量级中文模型
            - "dashscope": 阿里云百炼嵌入模型
    """
    if model_name == "dashscope":
        return DashScopeEmbeddingsWrapper(**kwargs)
    return DeepSeekEmbeddingsWrapper(model_name=model_name, **kwargs) 