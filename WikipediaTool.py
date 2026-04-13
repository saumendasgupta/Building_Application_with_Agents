from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import StateGraph
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, add_messages
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(wiki_client=any,top_k_results=3, doc_content_chars_max=1000)
tools = [WikipediaQueryRun(api_wrapper=api_wrapper)]

llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

messages: list[BaseMessage] = [HumanMessage(content="What was the most impressive thing about Buzz Aldrin?")]
ai_message = llm_with_tools.invoke(messages)
messages = messages + [ai_message]
print(ai_message)

for tool_call in ai_message.tool_calls:
    tool_msg = tools[0].invoke(tool_call)
    print(tool_msg.name)
    print(tool_call['args'])
    print(tool_msg.content)
    messages.append(tool_msg)
    print()
final_response = llm_with_tools.invoke(messages)
print(final_response.content)