"""
智能RAG系统 - 基于LangGraph实现 (DeepSeek版)
"""

import os
import time
import traceback
import json
from typing import Annotated, Literal, Sequence, Optional, List, Any
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_core.messages.tool import ToolMessage

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from pydantic import BaseModel, Field

# 导入DeepSeek集成模块
from deepseek_integration import get_chat_model, get_embedding_model

# 全局工具变量
tools = None
globals = {"tools": None, "verbose": False}

class AgentState(TypedDict):
    """代理状态，使用消息列表作为状态存储"""
    # add_messages函数定义了如何处理更新，默认是替换，add_messages表示"追加"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def initialize_retriever(
    embedding_model: str = "dashscope", 
    dimensions: int = 1024,
    threshold: float = 0.3,
    force_reindex: bool = False,
    verbose: bool = False
):
    embedding_function = get_embedding_model(model_name=embedding_model, dimensions=dimensions)
    persist_directory = "chroma_db"
    os.makedirs(persist_directory, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    ) if not force_reindex and os.listdir(persist_directory) else None
    if vectorstore is None or vectorstore._collection.count() == 0:
        data = []
        # 本地文档加载逻辑按需扩展
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vectorstore.persist()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    @tool
    def retrieve_documents(query: str) -> str:
        """从向量存储中检索与查询相关的文档并返回文本内容。"""
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs]) if docs else ""
    return [retrieve_documents]

def grade_documents(state):
    """
    评估文档相关性，决定下一个节点。将结果存入state['next']并返回state。
    """
    try:
        print("开始检查文档相关性...")
        # 获取问题和文档列表
        question = state.get("messages", [])[0].content if state.get("messages") else ""
        docs = state.get("documents", [])
        print(f"问题: '{question}'，检索到 {len(docs)} 个文档")
        # 如果无文档，直接生成回答
        if not docs:
            print("没有检索到文档，直接生成回答")
            state["next"] = "generate"
            return state
        # 初始化模型
        chat = get_chat_model(temperature=0)
        # 遍历文档进行评分
        for idx, doc in enumerate(docs):
            content = getattr(doc, "page_content", str(doc)) or ""
            if len(content.strip()) < 10:
                print(f"文档 {idx+1} 内容过短，跳过")
                continue
            print(f"评分文档 {idx+1}/{len(docs)}, 长度 {len(content)} 字符")
            # 构建评分提示
            prompt = f"""
判断以下文档是否与问题相关。
问题: {question}
文档内容: {content}
只返回JSON格式: {{"score":0或1}}，1表示相关
"""
            try:
                resp = chat.invoke([HumanMessage(content=prompt)], response_format={"type":"json_object"})
                data = json.loads(resp.content)
                score = int(data.get("score", 0))
                if score >= 1:
                    print(f"文档 {idx+1} 相关，生成回答")
                    state["next"] = "generate"
                    return state
                else:
                    print(f"文档 {idx+1} 不相关，继续")
            except Exception as e:
                print(f"评分文档 {idx+1} 时出错: {e}")
                state["next"] = "generate"
                return state
        # 所有文档均不相关，则重新检索
        print("所有文档不相关，执行检索")
        state["next"] = "retrieve"
        return state
    except Exception as e:
        print(f"grade_documents出错: {e}")
        traceback.print_exc()
        state["next"] = "generate"
        return state

def agent(state, verbose: bool = False):
    """
    调用代理模型基于当前状态生成响应。
    使用检索工具查询相关文档。
    
    参数:
        state (messages): 当前状态
        verbose: 是否输出详细日志
        
    返回:
        dict: 更新后的状态，包含检索指令
    """
    try:
        if verbose:
            print("---调用代理检索文档---")
        
        # 获取用户问题
        messages = state["messages"]
        human_messages = [m for m in messages if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"]
        
        if not human_messages:
            if verbose:
                print("未找到用户问题，无法执行检索")
            return {"messages": [AIMessage(content="无法找到用户问题，请提供有效的查询")]}
            
        user_query = human_messages[0].content
        
        if verbose:
            print(f"用户查询: '{user_query}'")
        
        # 使用DeepSeek模型
        model = get_chat_model()
        global tools
        
        # 确保tools变量存在且非空
        if not (tools and isinstance(tools, list) and len(tools) > 0):
            if verbose:
                print("警告: 检索工具未定义或为空")
            return {"messages": [AIMessage(content="检索系统未正确初始化，无法执行文档检索")]}
            
        if verbose:
            print(f"可用工具列表：{[t.name for t in tools if hasattr(t, 'name')]}")
            
        try:
            # 直接构造一个工具调用的结果并返回
            from langchain_core.messages.tool import ToolMessage, ToolCall as LC_ToolCall
            
            retriever_tool = None
            for tool in tools:
                if hasattr(tool, 'name') and tool.name == "retrieve_documents":
                    retriever_tool = tool
                    break
                    
            if not retriever_tool:
                if verbose:
                    print("未找到检索工具")
                return {"messages": [AIMessage(content="检索工具未找到，无法执行文档检索")]}
                
            # 创建一个toolcall，使用检索文档工具
            tool_call = LC_ToolCall(
                name="retrieve_documents",
                args={"query": user_query},
                id="call_retriever_1"
            )
            
            if verbose:
                print(f"创建工具调用: {tool_call.name}，参数: {tool_call.args}")
            
            # 返回工具调用消息
            return {
                "messages": [
                    AIMessage(
                        content="我需要检索文档来回答这个问题",
                        tool_calls=[tool_call]
                    )
                ]
            }
        except Exception as e:
            if verbose:
                print(f"触发工具调用时出错: {e}")
                print(traceback.format_exc())
            # 发生错误时的后备计划：返回一个无法检索的消息
            return {"messages": [AIMessage(content=f"无法检索相关文档。错误: {str(e)}")]}
            
    except Exception as e:
        if verbose:
            print(f"代理执行失败: {e}")
            print(traceback.format_exc())
        return {"messages": [AIMessage(content=f"代理执行过程中出错: {str(e)}")]}

def rewrite(state, verbose: bool = False):
    """
    转换查询以产生更好的问题
    
    参数:
        state (messages): 当前状态
        verbose: 是否输出详细日志
        
    返回:
        dict: 更新后的状态，包含重新表述的问题
    """
    try:
        if verbose:
            print("---转换查询以获取更相关的文档---")
            
        messages = state["messages"]
        
        # 获取用户的原始问题
        human_messages = [m for m in messages if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"]
        if not human_messages:
            if verbose:
                print("未找到用户问题，无法重写查询")
            return {"messages": messages}
            
        original_question = human_messages[0].content
        
        # 获取检索到的文档内容（如果有）
        tool_messages = [m for m in messages if hasattr(m, "tool_call_id") or getattr(m, "type", "") == "tool"]
        doc_content = ""
        if tool_messages:
            doc_content = tool_messages[-1].content
            
        if verbose:
            print(f"原始问题: {original_question}")
            if doc_content:
                print(f"文档内容长度: {len(doc_content)} 字符")
            else:
                print("没有可用的文档内容")
                
        # 使用模型重写问题
        chat_model = get_chat_model()
        
        # 构建文档内容部分
        docs_section = "已检索文档内容（不够相关）:\n" + doc_content[:2000] if doc_content else "没有检索到文档。"
        
        prompt = f"""
        你的任务是重写用户的原始问题，使其能够更好地匹配相关文档。
        基于我们已检索到的文档内容（如果有），识别关键术语和概念，使重写后的问题能够更好地检索相关文档。
        
        原始问题: {original_question}
        
        {docs_section}
        
        重写问题需要：
        1. 保持原始问题的本质意图不变
        2. 增加相关关键词和术语，提高检索精度
        3. 使用更具体、清晰的表达
        4. 简洁明了，通常不超过原始问题长度的2倍
        
        只提供重写后的问题，不要有任何解释或额外内容。
        """
        
        try:
            if verbose:
                print("使用LLM重写查询...")
                
            # 调用模型
            messages = [
                HumanMessage(content=prompt)
            ]
            
            response = chat_model.invoke(messages)
            rewritten_question = response.content.strip()
            
            if not rewritten_question or len(rewritten_question) < 5:
                if verbose:
                    print("重写失败，使用原始问题")
                return {"messages": state["messages"]}
                
            if verbose:
                print(f"重写后的问题: {rewritten_question}")
                
            # 创建新消息
            return {
                "messages": [
                    HumanMessage(content=rewritten_question),
                ]
            }
            
        except Exception as e:
            if verbose:
                print(f"重写查询时出错: {e}")
                print(traceback.format_exc())
            return {"messages": state["messages"]}
            
    except Exception as e:
        if verbose:
            print(f"查询重写过程失败: {e}")
            print(traceback.format_exc())
        return {"messages": state["messages"]}

def generate(state, show_docs: bool = False, verbose: bool = False):
    """
    生成最终回答
    
    参数:
        state (messages): 当前状态
        show_docs: 是否在回答中包含文档内容
        verbose: 是否输出详细日志
        
    返回:
        dict: 更新后的状态，包含生成的回答
    """
    try:
        if verbose:
            print("---生成最终回答---")
            
        messages = state["messages"]
        
        # 获取用户问题
        human_messages = [m for m in messages if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human"]
        if not human_messages:
            if verbose:
                print("未找到用户问题，无法生成回答")
            return {
                "messages": messages + [AIMessage(content="无法找到用户问题，请提供有效的查询")]
            }
            
        user_question = human_messages[-1].content
        
        # 获取检索到的文档内容
        tool_messages = [m for m in messages if hasattr(m, "tool_call_id") or getattr(m, "type", "") == "tool"]
        
        doc_content = ""
        if tool_messages:
            doc_content = tool_messages[-1].content
            
        if verbose:
            print(f"用户问题: {user_question}")
            if doc_content:
                print(f"文档内容长度: {len(doc_content)} 字符")
            else:
                print("没有检索到文档内容")
                
        # 如果没有文档内容，提供一个友好的回复
        if not doc_content or len(doc_content.strip()) < 50 or "未找到相关文档" in doc_content:
            no_docs_message = f"抱歉，我无法找到与您问题相关的文档: '{user_question}'。请尝试用不同的方式提问，或者确认您的问题是否与当前文档集合相关。"
            
            if verbose:
                print("无可用文档，返回提示信息")
                
            return {
                "messages": messages + [AIMessage(content=no_docs_message)]
            }
            
        # 使用模型生成回答
        chat_model = get_chat_model()
        
        prompt = f"""
        我需要你帮我回答用户的问题，基于检索到的文档内容。

        用户问题: {user_question}

        检索到的文档内容:
        {doc_content}

        请按照以下要求回答:
        1. 回答必须基于提供的文档内容，不要添加未在文档中提到的信息
        2. 对回答中的关键点，请指明来自哪个文档
        3. 如果文档内容不足以完全回答问题，请明确说明
        4. 使用连贯、自然的中文回答，适合口语对话
        5. 回答应该简洁明了，通常不超过300字
        
        直接提供最终回答，不要解释你的思考过程或方法。
        """
        
        try:
            if verbose:
                print("使用LLM生成回答...")
                
            # 调用模型
            llm_messages = [
                HumanMessage(content=prompt)
            ]
            
            response = chat_model.invoke(llm_messages)
            answer = response.content.strip()
            
            if not answer:
                if verbose:
                    print("生成的回答为空，使用默认回答")
                answer = "抱歉，我无法基于提供的文档生成有效回答。请尝试用不同方式提问。"
                
            if verbose:
                print(f"生成回答长度: {len(answer)} 字符")
                
            # 如果需要显示文档
            if show_docs:
                if verbose:
                    print("包含文档内容在回答中")
                answer += "\n\n------\n参考文档内容:\n" + doc_content
                
            # 返回最终回答
            return {
                "messages": messages + [AIMessage(content=answer)]
            }
            
        except Exception as e:
            if verbose:
                print(f"生成回答时出错: {e}")
                print(traceback.format_exc())
            return {
                "messages": messages + [AIMessage(content=f"生成回答时出错: {str(e)}")]
            }
            
    except Exception as e:
        if verbose:
            print(f"回答生成过程失败: {e}")
            print(traceback.format_exc())
        return {
            "messages": state["messages"] + [AIMessage(content=f"回答生成过程中出错: {str(e)}")]
        }

def create_graph(tools, verbose: bool = False):
    """
    创建代理工作流图
    
    参数:
        tools: 工具列表
        verbose: 是否输出详细日志
        
    返回:
        graph: 工作流图
    """
    try:
        if verbose:
            print("创建代理工作流图...")
            
        # 初始化全局工具变量
        global globals
        globals = {"tools": tools, "verbose": verbose}
        
        builder = StateGraph(AgentState)
        
        # 检查工具是否有效
        if not tools or not isinstance(tools, list):
            if verbose:
                print("未提供有效的工具列表，创建无工具的工作流图")
        
        # 配置节点
        builder.add_node("agent", lambda state: agent(state, verbose=verbose))
        builder.add_node("retrieve", ToolNode(tools))
        builder.add_node("grade_documents", lambda state: grade_documents(state))
        builder.add_node("rewrite", lambda state: rewrite(state, verbose=verbose))
        builder.add_node("generate", lambda state: generate(state, verbose=verbose))
        
        # 创建边
        builder.set_entry_point("agent")
        builder.add_edge("agent", "retrieve")
        builder.add_edge("retrieve", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("next", "generate"),
            {
                "rewrite": "rewrite",
                "generate": "generate",
                "retrieve": "retrieve"
            }
        )
        builder.add_edge("rewrite", "agent")
        builder.add_edge("generate", END)
        
        if verbose:
            print("工作流图创建完成")
            
        return builder.compile()
        
    except Exception as e:
        if verbose:
            print(f"创建工作流图失败: {e}")
            print(traceback.format_exc())
        # 尝试创建一个简化的工作流图
        try:
            if verbose:
                print("尝试创建简化的工作流图...")
                
            builder = StateGraph(AgentState)
            builder.add_node("agent", lambda state: agent(state, verbose=verbose))
            if tools and isinstance(tools, list):
                builder.add_node("retrieve", ToolNode(tools))
                builder.add_edge("agent", "retrieve")
                builder.add_edge("retrieve", "generate")
            else:
                builder.add_edge("agent", "generate")
            builder.add_node("generate", lambda state: generate(state, verbose=verbose))
            builder.add_edge("generate", END)
            builder.set_entry_point("agent")
            
            if verbose:
                print("简化工作流图创建完成")
                
            return builder.compile()
            
        except Exception as e2:
            if verbose:
                print(f"创建简化工作流图也失败: {e2}")
                print(traceback.format_exc())
            raise RuntimeError(f"无法创建工作流图: {str(e)} -> {str(e2)}")

def process_query(query: str, verbose: bool = False) -> str:
    """
    直接调用检索工具并通过 LLM 生成回答，返回字符串。
    """
    # 初始化并获取检索工具
    tools = initialize_retriever(verbose=verbose)
    if not tools or len(tools) == 0:
        return "未找到检索工具，请确认初始化成功。"
    retrieve = tools[0]
    # 执行文档检索
    try:
        docs = retrieve.invoke(query)
    except Exception as e:
        return f"检索文档出错: {e}"
    if verbose:
        print(f"检索到文档长度: {len(docs)} 字符")
    # 构建生成提示
    prompt = f"""
下面是从文档集合检索到的内容，请严格基于以下内容回答问题，禁止添加、补充或篡改任何未在文档中出现的信息；
如果文档中未提及相关信息，请回答"文档中未提及相关信息"。

文档内容：
{docs}

问题：{query}
"""
    try:
        chat = get_chat_model()
        response = chat.invoke([HumanMessage(content=prompt)])
        ans = response.content.strip()
        return ans or "抱歉，未能生成有效回答。"
    except Exception as e:
        traceback.print_exc()
        return f"生成回答时出错: {e}"

def clean_response(text):
    """
    清理响应文本，去除不必要的前缀
    
    参数:
        text: 原始文本
        
    返回:
        str: 清理后的文本
    """
    # 去除常见的前缀
    prefixes = [
        "根据文档内容，",
        "根据提供的文档，",
        "基于提供的文档，",
        "根据文档，",
        "根据提供的信息，",
        "基于文档内容，"
    ]
    
    result = text
    for prefix in prefixes:
        if result.startswith(prefix):
            result = result[len(prefix):]
            break
            
    return result.strip()

if __name__ == "__main__":
    # 初始化工具
    tools = initialize_retriever()
    
    # 创建图
    graph = create_graph(tools)
    
    # 处理查询示例
    query = input("请输入您的问题: ")
    response = process_query(query)
    print("\n回答:")
    print(response) 