from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from agents.chatbot import chatbot
from agents.openreview_agent import openreview_agent
from agents.paper_research_supervisor import paper_research_supervisor
from agents.rag_assistant import rag_assistant
from schema import AgentInfo

DEFAULT_AGENT = "rag-assistant"


AgentGraph = CompiledStateGraph | Pregel  


@dataclass
class Agent:
    description: str
    graph_like : AgentGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph_like=chatbot),
    "rag-assistant": Agent(
        description="A RAG assistant with access to information in a database.",
        graph_like=rag_assistant,
    ),
    "openreview-agent": Agent(
        description="An agent specialized in searching academic papers from OpenReview (ICML, NeurIPS, ICLR, etc.).",
        graph_like=openreview_agent,
    ),
    "paper-research-supervisor": Agent(
        description="A supervisor agent that coordinates between OpenReview agent (for searching papers) and RAG assistant (for querying paper database). Can search papers and answer questions about them.",
        graph_like=paper_research_supervisor,
    ),
}


async def load_agent(agent_id: str) -> None:
    """Load lazy agents if needed."""
    graph_like = agents[agent_id].graph_like
    if isinstance(graph_like, ):
        await graph_like.load()


def get_agent(agent_id: str) -> AgentGraph:
    """Get an agent graph, loading lazy agents if needed."""
    agent_graph = agents[agent_id].graph_like

    # If it's a lazy loading agent, ensure it's loaded and return its graph
    if isinstance(agent_graph,):
        if not agent_graph._loaded:
            raise RuntimeError(f"Agent {agent_id} not loaded. Call load() first.")
        return agent_graph.get_graph()

    # Otherwise return the graph directly
    return agent_graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
