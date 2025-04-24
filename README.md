# LangGraph 智能文档检索系统

这是一个基于 LangGraph 框架开发的智能文档检索与问答系统。它能够从本地文档中创建向量检索库，并基于用户查询提供精确的答案。系统支持多种文档格式，包括 PDF、Word、TXT 和 Markdown 文件。

## 技术栈

### 核心框架
- **LangGraph**: 基于工作流的智能代理框架，用于构建复杂的AI应用
- **LangChain**: 提供文档加载、文本分割、向量存储等基础功能
- **ChromaDB**: 高性能向量数据库，用于存储和检索文档嵌入

### 大语言模型
- **DeepSeek**: 使用 DeepSeek API 进行智能问答和文档处理
  - 支持流式输出
  - 支持工具调用
  - 支持结构化输出

### 嵌入模型
- **阿里云百炼 (DashScope)**
  - 模型：text-embedding-v3
  - 维度：1024
  - 特点：针对中文优化，支持长文本
- **Sentence-Transformer**
  - 支持多种中文优化模型：
    - paraphrase-multilingual-MiniLM-L12-v2（多语言支持）
    - shibing624/text2vec-base-chinese（通用中文）
    - GanymedeNil/text2vec-large-chinese（大型中文）
    - moka-ai/m3e-base（新一代中文）
    - shibing624/text2vec-base-chinese-paraphrase（轻量级）

## 主要功能

- 📚 **多格式文档支持**：支持 PDF、Word、TXT、Markdown 等多种文档格式
- 🔍 **智能检索**：基于向量数据库的语义检索，准确找到相关文档
- 🤖 **智能问答**：使用 DeepSeek 大语言模型进行智能问答
- 🔄 **文档重写**：自动优化查询以获得更好的检索结果
- 📊 **文档评分**：自动评估文档相关性，确保回答质量

## 系统架构

系统主要由以下几个核心组件构成：

1. **文档加载器** (`local_loader.py`)
   - 基于 LangChain 的文档加载器
   - 支持多种文档格式的加载
   - 特别优化了 Word 文档的处理
   - 支持批量文档加载
   - 使用 python-docx 处理 Word 文档

2. **智能检索系统** (`agentic_rag.py`)
   - 基于 LangGraph 的工作流管理
   - 使用 ChromaDB 进行向量存储和检索
   - 文档检索和评分
   - 查询重写和优化
   - 支持相似度阈值过滤

3. **模型集成** (`deepseek_integration.py`)
   - DeepSeek API 集成
   - 阿里云百炼 API 集成
   - 多种中文优化嵌入模型支持
   - 支持批处理以提高效率

## 安装说明

1. 克隆项目到本地：
```bash
git clone [项目地址]
cd Langgraph
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
创建 `.env` 文件并添加以下配置：
```
DEEPSEEK_API_KEY=你的DeepSeek API密钥
DASHSCOPE_API_KEY=你的阿里云百炼API密钥（可选）
```

## 使用方法

### 1. 初始化文档库

```python
from main import index_local_documents

# 索引本地文档
retrieval_tools = index_local_documents(
    doc_dir="你的文档目录路径",
    file_patterns=["*.pdf", "*.txt", "*.md", "*.docx"],
    embedding_model="dashscope",  # 使用阿里云百炼嵌入模型
    dimensions=1024,  # 向量维度
    threshold=0.3,    # 相似度阈值
    chunk_size=1000,  # 文档分块大小
    chunk_overlap=200,# 分块重叠大小
    verbose=True
)
```

### 2. 进行问答

```python
from main import process_query

# 提出问题
answer = process_query(
    "你的问题",
    embedding_model="dashscope",  # 选择嵌入模型
    threshold=0.3,                # 设置相似度阈值
    verbose=True
)
print(answer)
```

### 3. 直接使用 LLM 回答

```python
from main import direct_answer

# 直接使用 LLM 回答
answer = direct_answer(
    "你的问题",
    temperature=0.3,  # 控制回答的随机性
    verbose=True
)
print(answer)
```

## 配置选项

### 嵌入模型配置
- `embedding_model`：嵌入模型选择
  - "dashscope"：阿里云百炼嵌入模型
  - "default"：多语言模型
  - "chinese-general"：通用中文模型
  - "chinese-large"：大型中文模型
  - "chinese-new"：新一代中文模型
  - "chinese-light"：轻量级中文模型

### 检索配置
- `dimensions`：向量维度（默认：1024）
- `threshold`：相似度阈值（默认：0.3）
- `chunk_size`：文档分块大小（默认：1000）
- `chunk_overlap`：分块重叠大小（默认：200）

### LLM配置
- `temperature`：控制回答的随机性（默认：0.3）
- `max_tokens`：最大输出长度
- `top_p`：采样概率阈值

## 性能优化

1. **批处理优化**
   - 文档加载使用批处理
   - 嵌入生成使用批处理
   - 向量存储使用批处理

2. **内存优化**
   - 使用分块处理大文档
   - 支持流式处理
   - 自动清理临时文件

3. **检索优化**
   - 支持相似度阈值过滤
   - 支持多种检索策略
   - 支持查询重写

## 注意事项

1. 确保已安装所有必要的依赖包
2. 需要有效的 API 密钥才能使用 DeepSeek 和阿里云百炼服务
3. 首次运行时会创建向量数据库，可能需要一些时间
4. 建议使用虚拟环境以避免依赖冲突
5. 注意 API 调用限制和费用

## 常见问题

1. **Q: 系统支持哪些文档格式？**
   A: 目前支持 PDF、Word、TXT、Markdown 等常见文档格式。

2. **Q: 如何提高检索准确度？**
   A: 可以调整 `threshold` 参数，或使用更高质量的嵌入模型。

3. **Q: 系统对中文支持如何？**
   A: 系统特别优化了对中文的支持，包括使用中文优化的嵌入模型。

4. **Q: 如何处理大文档？**
   A: 系统会自动将文档分块处理，可以通过 `chunk_size` 和 `chunk_overlap` 参数调整。

## 贡献指南

欢迎提交 Pull Request 或 Issue 来帮助改进项目。在提交代码前，请确保：

1. 代码符合 PEP 8 规范
2. 添加必要的测试
3. 更新相关文档

## 许可证

[许可证类型]