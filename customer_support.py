from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import StateGraph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, add_messages

# 1. Define the structure of your state
class AgentState(TypedDict):
    # Annotated with add_messages tells LangGraph to append new messages 
    # to the list rather than overwriting it
    messages: Annotated[list, add_messages]
    order: dict


# -- 1) Define our single business tool
@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order that hasn't shipped."""
    # (Here you'd call your real backend API)
    return f"Order {order_id} has been cancelled."
# -- 2) The agent "brain": invoke LLM, run tool, then invoke LLM again

def call_model(state):
    msgs = state["messages"]
    order = state.get("order", {"order_id": "UNKNOWN"})
    # System prompt tells the model exactly what to do
    prompt = (
    f'''You are an ecommerce support agent.
    ORDER ID: {order['order_id']}
    If the customer asks to cancel, call cancel_order(order_id)
    and then send a simple confirmation.
    Otherwise, just respond normally.'''
    )
    full = [SystemMessage(prompt)] + msgs
    # 1st LLM pass: decides whether to call our tool
    first = ChatOllama(model="llama3.2", temperature=0).invoke(full)
    out = [first]
    
    if getattr(first, "tool_calls", None):
    # run the cancel_order tool
        tc = first.tool_calls[0]
        result = cancel_order(**tc["args"])
        out.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    # 2nd LLM pass: generate the final confirmation text
        second = ChatOllama(model="llama3.2", temperature=0).invoke(full + out)
        out.append(second)
    return {"messages": out}

# -- 4) Wire it all up in a StateGraph
def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()

graph = construct_graph()

if __name__ == "__main__":
    example_order = {"order_id": "A12345"}
    convo = [HumanMessage(content="Please cancel my order A12345.")]
    result = graph.invoke({"order": example_order, "messages": convo})
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
g.add_edge("assistant", "node_3")