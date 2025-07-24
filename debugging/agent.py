from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START,END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from IPython.display import Image, display
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]="agentic_ai"


llm=init_chat_model(model="groq:llama3-8b-8192") # can also use chatgroq

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def make_tool_graph():
    @tool
    def add(a:float,b:float)->float:
        """
        Add two numbers together
        """
        return a+b
    tools=[add]
    tool_node=ToolNode(tools=tools)
    llm_with_tool=llm.bind_tools(tools=tools)

    def call_llm_model(state:State):
        return {"messages":[llm_with_tool.invoke(state["messages"])]}
    
    builder=StateGraph(State)
    builder.add_node("tool_calling_llm",call_llm_model)
    builder.add_node("tools",ToolNode(tools))
    builder.add_edge(START,"tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
        # if tool is called then go to tools node
        # if tool is not called then go to tool_calling_llm node and route to END
    )
    builder.add_edge("tools","tool_calling_llm")
    builder.add_edge("tool_calling_llm",END)
    graph=builder.compile()
    return graph

tool_graph=make_tool_graph()
