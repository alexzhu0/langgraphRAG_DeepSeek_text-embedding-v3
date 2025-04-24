#!/usr/bin/env python
"""
基于LangGraph的文档检索生成工具

该工具从本地文档目录创建向量检索库，并基于用户查询提供精确答案
支持PDF和文本文件，可根据需要扩展其他文件类型支持
"""

import os
import glob
import argparse
import sys
import time
import traceback
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)

from agentic_rag import initialize_retriever, process_query, tools
from local_loader import load_documents_from_directory
from deepseek_integration import get_chat_model

def load_existing_vector_store(
    embedding_model: str = "dashscope", 
    dimensions: int = 1024,
    threshold: float = 0.3,
    verbose: bool = False
):
    """
    尝试加载现有的向量存储
    
    参数:
        embedding_model: 嵌入模型名称
        dimensions: 向量维度
        threshold: 相似度阈值
        verbose: 是否输出详细日志
        
    返回:
        retrieval_tools: 检索工具列表，如果加载失败则返回None
    """
    try:
        if verbose:
            print(f"尝试加载现有向量库: chroma_db")
            
        persist_directory = "chroma_db"
        if not os.path.exists(persist_directory):
            if verbose:
                print(f"向量库目录不存在: {persist_directory}")
            return None
            
        # 初始化检索器
        retrieval_tools = initialize_retriever(
            embedding_model=embedding_model,
            dimensions=dimensions,
            threshold=threshold,
            verbose=verbose
        )
        
        if retrieval_tools is None:
            if verbose:
                print("无法初始化检索器")
            return None
            
        return retrieval_tools
        
    except Exception as e:
        if verbose:
            print(f"加载向量库时发生错误: {e}")
            print(traceback.format_exc())
        return None

def load_document(file_path: str, verbose: bool = False):
    """
    加载单个文档文件
    
    参数:
        file_path: 文件路径
        verbose: 是否输出详细日志
        
    返回:
        docs: Document对象列表
    """
    try:
        if verbose:
            print(f"加载文档: {file_path}")
            
        if not os.path.exists(file_path):
            if verbose:
                print(f"文件不存在: {file_path}")
            return None
            
        # 根据文件扩展名选择合适的加载器
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        elif ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext in ['.doc', '.docx']:
            from local_loader import CustomWordLoader
            try:
                loader = CustomWordLoader(file_path)
            except Exception as e:
                if verbose:
                    print(f"使用自定义Word加载器失败: {e}, 尝试使用Unstructured加载器")
                loader = UnstructuredWordDocumentLoader(file_path)
        else:
            if verbose:
                print(f"不支持的文件类型: {file_path}")
            return None
            
        docs = loader.load()
        
        if verbose:
            print(f"成功加载文档: {file_path}，包含 {len(docs)} 页")
            
        return docs
        
    except Exception as e:
        if verbose:
            print(f"文档加载失败: {file_path}")
            if verbose:
                print(f"错误详情: {e}")
        return None
    
def index_local_documents(
    doc_dir: str, 
    file_patterns: List[str] = ["*.pdf", "*.txt", "*.md", "*.docx"],
    chunk_size: int = 1000, 
    chunk_overlap: int = 200,
    embedding_model: str = "dashscope",
    dimensions: int = 1024,
    threshold: float = 0.3,
    verbose: bool = False
):
    """
    索引本地文档
    
    参数:
        doc_dir: 文档目录
        file_patterns: 文件模式列表
        chunk_size: 文档分块大小
        chunk_overlap: 分块重叠大小
        embedding_model: 嵌入模型名称
        dimensions: 向量维度
        threshold: 相似度阈值
        verbose: 是否输出详细日志
        
    返回:
        retrieval_tools: 检索工具列表，如果索引失败则返回None
    """
    try:
        if verbose:
            print(f"索引本地文档目录: {doc_dir}")
            print(f"使用嵌入模型: {embedding_model}, 维度: {dimensions}")
            print(f"文档分块设置: 大小={chunk_size}, 重叠={chunk_overlap}")
            
        # 设置分块参数
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
        
        # 检查文档目录是否存在
        if not os.path.exists(doc_dir):
            if verbose:
                print(f"文档目录不存在: {doc_dir}")
            return None
            
        # 使用glob模式加载文档
        all_docs = []
        
        # 查找所有匹配的文件
        all_files = []
        from local_loader import find_files_with_patterns
        all_files = find_files_with_patterns(doc_dir, file_patterns)
        
        if all_files:
            if verbose:
                print(f"找到 {len(all_files)} 个匹配的文件")
        else:
            if verbose:
                print(f"未找到匹配的文件，请检查目录和文件模式")
            return None
        
        # 加载所有文件
        for file_path in all_files:
            try:
                docs = load_document(file_path, verbose=verbose)
                if docs:
                    all_docs.extend(docs)
                else:
                    if verbose:
                        print(f"跳过文件: {file_path}")
            except Exception as e:
                if verbose:
                    print(f"加载文档失败: {file_path}, {e}")
                    if verbose:
                        print(traceback.format_exc())
        
        if not all_docs:
            if verbose:
                print("未能成功加载任何文档")
            return None
            
        if verbose:
            print(f"成功加载 {len(all_docs)} 个文档")
            
        # 使用RAG系统的initialize_retriever函数创建向量存储
        retrieval_tools = initialize_retriever(
            urls=None,  # 不使用URL，使用本地文档
            embedding_model=embedding_model,
            dimensions=dimensions,
            threshold=threshold,
            force_reindex=True,  # 强制重新索引
            verbose=verbose
        )
        
        return retrieval_tools
        
    except Exception as e:
        if verbose:
            print(f"索引文档时发生错误: {e}")
            if verbose:
                print(traceback.format_exc())
        return None

def direct_answer(
    query: str, 
    verbose: bool = False
):
    """
    直接使用LLM生成回答，不依赖检索系统
    
    参数:
        query: 用户查询
        verbose: 是否输出详细日志
        
    返回:
        answer: 生成的回答
    """
    try:
        if verbose:
            print(f"直接回答模式，查询: '{query}'")
            
        from deepseek_integration import get_chat_model
        from langchain_core.messages import HumanMessage
        
        # 获取模型
        model = get_chat_model()
        
        # 构造提示
        prompt = f"""请回答以下问题。如果你不知道答案，请直接说明你没有相关信息，不要编造答案。
        
        问题: {query}
        """
        
        # 调用模型
        response = model.invoke([HumanMessage(content=prompt)])
        answer = response.content
        
        if verbose:
            print(f"生成回答完成，长度: {len(answer)} 字符")
            
        return answer
        
    except Exception as e:
        if verbose:
            print(f"直接回答时出错: {e}")
            if verbose:
                print(traceback.format_exc())
        return f"抱歉，生成回答时出错: {str(e)}"

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于LangGraph的文档检索生成工具")
    parser.add_argument("--query", "-q", type=str, help="要查询的问题，如果不提供则进入交互模式")
    parser.add_argument("--doc-dir", "-d", type=str, default="./docs", help="本地文档目录")
    parser.add_argument(
        "--file-patterns", 
        "-p", 
        nargs="+", 
        default=["*.pdf", "*.txt", "*.md", "*.docx"],
        help="文件模式，例如 *.pdf *.txt"
    )
    parser.add_argument("--force-reindex", "-f", action="store_true", help="强制重新索引文档")
    parser.add_argument("--chunk-size", "-c", type=int, default=1000, help="文档分块大小")
    parser.add_argument("--chunk-overlap", "-o", type=int, default=200, help="分块重叠大小")
    parser.add_argument(
        "--embedding-model", 
        "-e", 
        type=str, 
        default="dashscope", 
        help="嵌入模型名称，可选值有'dashscope'、'deepseek'等"
    )
    parser.add_argument("--dimensions", type=int, default=1024, help="向量维度")
    parser.add_argument("--threshold", "-t", type=float, default=0.3, help="相似度阈值")
    parser.add_argument("--show-docs", "-s", action="store_true", help="是否在回答中显示文档内容")
    parser.add_argument("--verbose", "-v", action="store_true", help="是否输出详细日志")
    parser.add_argument("--direct", action="store_true", help="是否直接使用LLM生成回答，不依赖检索系统")
    args = parser.parse_args()
    
    # 输出配置信息
    print("=" * 50)
    print("基于LangGraph的文档检索生成工具")
    print("=" * 50)
    print(f"文档目录: {args.doc_dir}")
    print(f"文件模式: {args.file_patterns}")
    print(f"嵌入模型: {args.embedding_model}, 维度: {args.dimensions}")
    print(f"相似度阈值: {args.threshold}")
    print(f"直接模式: {args.direct}")
    print(f"显示文档: {args.show_docs}")
    print(f"详细模式: {args.verbose}")
    print("=" * 50)
    
    # 初始化检索工具
    retrieval_tools = None
    
    # 直接模式，不需要初始化检索工具
    if not args.direct:
        # 尝试加载现有向量库
        if not args.force_reindex:
            retrieval_tools = load_existing_vector_store(
                embedding_model=args.embedding_model,
                dimensions=args.dimensions,
                threshold=args.threshold,
                verbose=args.verbose
            )
        
        # 如果需要强制重新索引或加载失败，则重新索引
        if retrieval_tools is None or args.force_reindex:
            retrieval_tools = index_local_documents(
                doc_dir=args.doc_dir,
                file_patterns=args.file_patterns,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_model=args.embedding_model,
                dimensions=args.dimensions,
                threshold=args.threshold,
                verbose=args.verbose
            )
    
    # 如果检索工具初始化失败且非直接模式，尝试使用直接模式
    if retrieval_tools is None and not args.direct:
        print("错误: 无法创建文档检索器，请检查文档目录和参数设置")
        print("将尝试使用直接回答模式")
        args.direct = True
    
    # 保持向量存储
    if retrieval_tools:
        # 设置全局工具
        global tools
        tools = retrieval_tools
    
    # 创建LangGraph处理图
    graph = None
    if not args.direct:
        try:
            # 仅在非直接模式下创建图
            from agentic_rag import create_graph
            graph = create_graph(retrieval_tools, verbose=args.verbose)
        except Exception as e:
            if args.verbose:
                print(f"创建LangGraph图时出错: {e}")
                print(traceback.format_exc())
            print("将改为使用直接回答模式")
            args.direct = True
    
    # 处理查询
    if args.direct:
        print(f"{'启用' if args.direct else '禁用'}直接回答模式")
    else:
        if retrieval_tools is None:
            print("错误: 无法初始化检索工具，将退出程序")
            sys.exit(1)
        print(f"{'启用' if not args.direct else '禁用'}检索增强回答模式")
            
    if args.show_docs:
        print(f"{'显示' if args.show_docs else '隐藏'}检索文档内容")
    
    # 如果提供了查询，则处理查询
    if args.query:
        try:
            if args.direct:
                answer = direct_answer(args.query, verbose=args.verbose)
            else:
                # 使用LangGraph处理查询
                answer = process_query(
                    args.query, 
                    graph=graph, 
                    show_docs=args.show_docs, 
                    verbose=args.verbose
                )
            print("\n回答:")
            print("=" * 50)
            print(answer)
            print("=" * 50)
        except Exception as e:
            print(f"处理查询时出错: {e}")
            if args.verbose:
                print(traceback.format_exc())
    else:
        # 进入交互模式
        print("进入交互模式，输入'退出'或'exit'以结束对话")
        while True:
            query = input("用户: ")
            if query.lower() in ['退出','exit','quit','q']:
                print("已退出交互模式。")
                break
            answer = process_query(query, verbose=True)
            print("\n回答:")
            print(answer)

# 交互式模式：进入持续对话模式，输入'退出'或'exit'结束会话
if __name__ == "__main__":
    print("进入交互模式，输入'退出'或'exit'以结束对话")
    while True:
        query = input("用户: ")
        if query.lower() in ['退出','exit','quit','q']:
            print("已退出交互模式。")
            break
        answer = process_query(query, verbose=True)
        print("\n回答:")
        print(answer) 