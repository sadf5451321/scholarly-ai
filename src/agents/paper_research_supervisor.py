from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.openreview_agent import openreview_agent
from agents.rag_assistant import rag_assistant
from core import get_model, settings
from core.logging_config import get_logger

logger = get_logger(__name__)


class AgentState(MessagesState, total=False):
    """Supervisor agent state that can route to sub-agents."""
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
You are a research supervisor assistant that coordinates between two specialized agents:
1. **OpenReview Agent**: Searches for academic papers from OpenReview (ICML, NeurIPS, ICLR, etc.)
2. **RAG Assistant**: Answers questions based on papers stored in a vector database

Today's date is {current_date}.

Your workflow:
- When users want to **search for papers**, use `transfer_to_openreview_agent`
- When users want to **ask questions about papers** in the database, use `transfer_to_rag_assistant`
- You can also coordinate both: first search papers, then answer questions about them

Available agents:
- `openreview_agent`: For searching academic papers from OpenReview
- `rag_assistant`: For querying information from the vector database of papers

Guidelines:
- If the user asks to search for papers or find papers, route to openreview_agent
- If the user asks questions about papers (that should be in the database), route to rag_assistant
- You can transfer between agents multiple times in one conversation
- Always provide clear context when transferring between agents
"""


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrap the model with tools for agent transfer."""
    # Create transfer tools
    from langchain_core.tools import tool
    
    @tool
    def transfer_to_openreview_agent(query: str) -> str:
        """Transfer to OpenReview agent to search for academic papers.
        
        Use this when the user wants to:
        - Search for papers from conferences (ICML, NeurIPS, ICLR, etc.)
        - Find papers by topic, venue, or conference
        - Get paper lists or summaries
        
        Args:
            query: The user's request for searching papers
        """
        return f"Transferring to OpenReview agent with query: {query}"
    
    @tool
    def transfer_to_rag_assistant(query: str) -> str:
        """Transfer to RAG assistant to answer questions about papers in the database.
        
        Use this when the user wants to:
        - Ask questions about papers that are already in the vector database
        - Query information from stored papers
        - Get detailed answers based on paper content
        
        Args:
            query: The user's question about papers in the database
        """
        return f"Transferring to RAG assistant with query: {query}"
    
    tools = [transfer_to_openreview_agent, transfer_to_rag_assistant]
    bound_model = model.bind_tools(tools)
    
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Call the supervisor model to decide which agent to route to."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """Check input safety."""
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """Block unsafe content."""
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


async def call_openreview_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """直接复用 openreview_agent - 调用现有的 OpenReview agent 搜索论文."""
    logger.info("调用 OpenReview agent (复用现有 agent)")
    
    # 获取最后一个 supervisor 的消息和工具调用
    last_message = state["messages"][-1]
    query = ""
    tool_call_id = None
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # 从工具调用中提取查询
        for tool_call in last_message.tool_calls:
            if "transfer_to_openreview_agent" in tool_call.get("name", ""):
                query = tool_call.get("args", {}).get("query", "")
                tool_call_id = tool_call.get("id")
                break
    
    # 如果没有从工具调用获取查询，使用原始用户消息
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
    
    # 直接复用 openreview_agent - 使用相同的状态和配置
    # 将当前状态传递给子 agent，保持消息历史
    agent_input = state  # 直接传递整个状态，让子 agent 处理
    
    # 使用相同的配置，确保 thread_id 等保持一致
    sub_config = RunnableConfig(
        configurable=config.get("configurable", {}),
        run_id=config.get("run_id"),
        callbacks=config.get("callbacks", []),
    )
    
    # 直接调用现有的 openreview_agent
    try:
        result = await openreview_agent.ainvoke(agent_input, sub_config)
        
        # 获取 agent 的响应
        agent_messages = result.get("messages", [])
        if agent_messages:
            # 找到最后一个 AI 消息作为响应
            last_agent_message = None
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage):
                    last_agent_message = msg
                    break
            
            if last_agent_message:
                content = last_agent_message.content if hasattr(last_agent_message, "content") else str(last_agent_message)
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id or "unknown"
                )
                return {"messages": [tool_message]}
    except Exception as e:
        logger.error(f"调用 openreview_agent 时出错: {e}", exc_info=True)
        tool_message = ToolMessage(
            content=f"调用 OpenReview agent 时出错: {str(e)}",
            tool_call_id=tool_call_id or "unknown"
        )
        return {"messages": [tool_message]}
    
    return {"messages": []}


async def call_rag_assistant(state: AgentState, config: RunnableConfig) -> AgentState:
    """直接复用 rag_assistant - 调用现有的 RAG assistant 回答论文相关问题."""
    logger.info("调用 RAG assistant (复用现有 agent)")
    
    # 获取最后一个 supervisor 的消息和工具调用
    last_message = state["messages"][-1]
    query = ""
    tool_call_id = None
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # 从工具调用中提取查询
        for tool_call in last_message.tool_calls:
            if "transfer_to_rag_assistant" in tool_call.get("name", ""):
                query = tool_call.get("args", {}).get("query", "")
                tool_call_id = tool_call.get("id")
                break
    
    # 如果没有从工具调用获取查询，使用原始用户消息
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break
    
    # 直接复用 rag_assistant - 使用相同的状态和配置
    # 将当前状态传递给子 agent，保持消息历史
    agent_input = state  # 直接传递整个状态，让子 agent 处理
    
    # 使用相同的配置，确保 thread_id 等保持一致
    sub_config = RunnableConfig(
        configurable=config.get("configurable", {}),
        run_id=config.get("run_id"),
        callbacks=config.get("callbacks", []),
    )
    
    # 直接调用现有的 rag_assistant
    try:
        result = await rag_assistant.ainvoke(agent_input, sub_config)
        
        # 获取 agent 的响应
        agent_messages = result.get("messages", [])
        if agent_messages:
            # 找到最后一个 AI 消息作为响应
            last_agent_message = None
            for msg in reversed(agent_messages):
                if isinstance(msg, AIMessage):
                    last_agent_message = msg
                    break
            
            if last_agent_message:
                content = last_agent_message.content if hasattr(last_agent_message, "content") else str(last_agent_message)
                tool_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id or "unknown"
                )
                return {"messages": [tool_message]}
    except Exception as e:
        logger.error(f"调用 rag_assistant 时出错: {e}", exc_info=True)
        tool_message = ToolMessage(
            content=f"调用 RAG assistant 时出错: {str(e)}",
            tool_call_id=tool_call_id or "unknown"
        )
        return {"messages": [tool_message]}
    
    return {"messages": []}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("supervisor", acall_model)
agent.add_node("openreview_agent", call_openreview_agent)
agent.add_node("rag_assistant", call_rag_assistant)

agent.set_entry_point("guard_input")


# Check for unsafe input
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "supervisor"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Route from supervisor to appropriate agent
def route_to_agent(state: AgentState) -> Literal["openreview_agent", "rag_assistant", "done"]:
    """Route to the appropriate sub-agent based on tool calls."""
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage):
        return "done"
    
    if not last_message.tool_calls:
        return "done"
    
    # Check which tool was called
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        if "transfer_to_openreview_agent" in tool_name:
            return "openreview_agent"
        elif "transfer_to_rag_assistant" in tool_name:
            return "rag_assistant"
    
    return "done"


agent.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "openreview_agent": "openreview_agent",
        "rag_assistant": "rag_assistant",
        "done": END,
    },
)

# After sub-agents finish, return to supervisor to decide next action
agent.add_edge("openreview_agent", "supervisor")
agent.add_edge("rag_assistant", "supervisor")


paper_research_supervisor = agent.compile()

