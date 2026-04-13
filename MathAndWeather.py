import math
import re
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

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y
@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y
@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'y' from 'x'."""
    return x - y
@tool
def divide(x: float, y: float) -> float:
    """Divide 'x' by 'y'."""
    if y == 0:
        return -1  # or raise an exception, depending on how you want to handle it
    return x / y
@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y

@tool
def logarithm(x: float, base: float = math.e) -> float:
    """Compute the logarithm of 'x' with the given 'base' (default: natural log). For example, logarithm(7, 5) returns log base 5 of 7."""
    if x <= 0:
        return float('nan')
    if base <= 0 or base == 1:
        return float('nan')
    return math.log(x) / math.log(base)

tools = [add, subtract, multiply, divide, exponentiate, logarithm]


def is_math_query(text: str) -> bool:
    text_l = text.lower()
    math_keywords = {
        "add", "plus", "sum", "subtract", "minus", "difference", "multiply",
        "times", "product", "divide", "quotient", "power", "exponent",
        "exponentiate", "log", "logarithm", "sqrt", "square", "cube", "calculate",
    }
    has_number = bool(re.search(r"\d", text_l))
    has_operator = any(op in text_l for op in ["+", "-", "*", "/", "^", "%"])
    has_math_word = any(word in text_l for word in math_keywords)
    return has_number or has_operator or has_math_word

#invoke the model with the tools, and then invoke it again with the tool results
def call_model(state):
    msgs = state["messages"]
    last_human_message = next(
        (m.content for m in reversed(msgs) if isinstance(m, HumanMessage)),
        "",
    )

    # Deterministic guardrail: avoid tool routing for clearly non-math requests.
    if not is_math_query(last_human_message):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I am a math assistant. I can help with calculations such as add, "
                        "subtract, multiply, divide, powers, and logarithms."
                    )
                )
            ]
        }

    prompt = (
    f'''You are a helpful assistant for doing math.
    If the user asks you to do a calculation, call the appropriate tool with the right arguments.
    If the user asks something non-math, do not call any tool and say you are a math assistant.'''
    )
    full = [SystemMessage(prompt)] + msgs
    model = ChatOllama(model="llama3.2", temperature=0)
    first = model.bind_tools(tools).invoke(full)
    out = [first]
    print("First pass output:", first)
    
    if getattr(first, "tool_calls", None):
        for tc in first.tool_calls:
            tool_func = next((t for t in tools if t.name == tc["name"]), None)
            if tool_func is None:
                out.append(
                    ToolMessage(
                        content=f"Unknown tool: {tc['name']}",
                        tool_call_id=tc["id"],
                    )
                )
                continue

            try:
                result = tool_func.invoke(tc["args"])
                out.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as exc:
                out.append(
                    ToolMessage(
                        content=f"Tool execution error: {exc}",
                        tool_call_id=tc["id"],
                    )
                )
        second = model.invoke(full + out)
        out.append(second)
    return {"messages": out}

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("assistant", call_model)
    g.set_entry_point("assistant")
    return g.compile()
graph = construct_graph()

if __name__ == "__main__":
    # Example usage
    state = {"messages": [HumanMessage("What is india")] }
    result = graph.invoke(state)
    print(result["messages"][-1].content)  # Should print the final response from the assistant
