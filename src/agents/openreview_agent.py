from datetime import datetime
from typing import Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import openreview_search
from core import get_model, settings
from core.logging_config import get_logger

logger = get_logger(__name__)

class AgentState(MessagesState, total=False):

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


tools = [openreview_search]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful academic paper search assistant specialized in finding papers from OpenReview.
    Today's date is {current_date}.

    You have access to the OpenReview_Search tool which can search for papers from conferences like:
    - ICML (International Conference on Machine Learning)
    - NeurIPS (Neural Information Processing Systems)
    - ICLR (International Conference on Learning Representations)
    - And other conferences hosted on OpenReview

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE DIRECTLY.

    Guidelines:
    - When searching for papers, you can specify the venue (e.g., "ICML 2025 oral", "NeurIPS 2024")
    - You can search for papers from different conferences by adjusting the venue and domain parameters
    - Always provide a summary of the papers found, including titles, authors, and key information
    - If the user asks about a specific conference or year, use the appropriate parameters
    - Format your responses clearly with paper titles, authors, and brief summaries
    - You can search for papers by default parameters or customize based on user requests
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    # 让模型知道有哪些工具可以调用
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        # 添加系统提示词
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    logger.info(f"preprocessor: {preprocessor}")
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    logger.info(f"format_safety_message_safety: {safety}")
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"acall_model_state: {state}")
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    logger.info(f"config: {config}")

    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
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
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info(f"block_unsafe_content_state: {state}")
    safety: LlamaGuardOutput = state["safety"]
    logger.info(f"safety: {safety}")
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))



# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    logger.info(f"check_safety_state: {state}")
    safety: LlamaGuardOutput = state["safety"]
    logger.info(f"safety: {safety}")
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"

agent.set_entry_point("guard_input")



agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:

    logger.debug(f"pending_tool_calls_state: {state}")

    last_message = state["messages"][-1]

    logger.debug(f"last_message: {last_message}")

    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"

# After the tool finishes processing, the result is passed back to the large language model
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


openreview_agent = agent.compile()

