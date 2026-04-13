from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, add_messages
from typing import Annotated, Any, TypedDict
import asyncio
import re

from langchain_mcp_adapters.client import MultiServerMCPClient

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

MCP_TOOLS: list[Any] | None = None

mcp_client = MultiServerMCPClient(
    {
        "weather": {
            "command": "python3",
            "args": ["src/common/mcp/MCP_weather_server.py"],
            "transport": "stdio",
        },
        "math": {
            "url": "http://127.0.0.1:8001/mcp",
            "transport": "streamable_http",
        },
    }
)

async def get_mcp_tools():
    global MCP_TOOLS
    if MCP_TOOLS is None:
        MCP_TOOLS = await mcp_client.get_tools()
        print("Loaded MCP tools:", [t.name for t in MCP_TOOLS])
    return MCP_TOOLS

def extract_city(text: str) -> str:
    lowered = text.lower()
    if "weather in" in lowered:
        return text.split("weather in", 1)[1].strip().rstrip("?")
    return text.strip()

def extract_two_numbers(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None, None

async def call_mcp_tools(state: AgentState) -> dict[str, Any]:
    user_text = str(state["messages"][-1].content)
    q = user_text.lower()

    tools = await get_mcp_tools()
    if not tools:
        return {"messages": [AIMessage(content="MCP tools not available.")]}

    # Weather route
    if "weather" in q:
        tool_obj = next((t for t in tools if t.name == "get_weather"), None)
        if not tool_obj:
            return {"messages": [AIMessage(content=f"Weather tool not found. Available: {[t.name for t in tools]}")]}
        city = extract_city(user_text)
        result = await tool_obj.ainvoke({"city": city})
        return {"messages": [AIMessage(content=str(result))]}

    # Math route
    x, y = extract_two_numbers(user_text)
    if x is not None and y is not None:
        if any(k in q for k in ["add", "plus", "+"]):
            name = "add"
        elif any(k in q for k in ["subtract", "minus", "-"]):
            name = "subtract"
        elif any(k in q for k in ["multiply", "times", "*"]):
            name = "multiply"
        elif any(k in q for k in ["divide", "/"]):
            name = "divide"
        else:
            return {"messages": [AIMessage(content="Specify operation: add, subtract, multiply, or divide.")]}

        tool_obj = next((t for t in tools if t.name == name), None)
        if not tool_obj:
            return {"messages": [AIMessage(content=f"Math tool '{name}' not found. Available: {[t.name for t in tools]}")]}

        result = await tool_obj.ainvoke({"x": x, "y": y})
        return {"messages": [AIMessage(content=f"{name}({x}, {y}) = {result}")]}
    
    return {"messages": [AIMessage(content="Ask weather or a math query with two numbers.")]}

def construct_graph():
    g = StateGraph(AgentState)
    g.add_node("mcp_agent", call_mcp_tools)
    g.set_entry_point("mcp_agent")
    return g.compile()

if __name__ == "__main__":
    graph = construct_graph()

    async def main():
        state = {"messages": [HumanMessage(content="multiply 12 and 3")]}
        result = await graph.ainvoke(state)
        print("Result:", result["messages"][-1].content)

    asyncio.run(main())