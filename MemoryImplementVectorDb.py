from typing import Annotated
from typing_extensions import TypedDict
from langchain_ollama.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import StateGraph, MessagesState, START

llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

# Use Chroma instead of vectordb2
memory = Chroma(
    collection_name="ml_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

text = """
    Machine learning is a method of data analysis that automates analytical
    model building. It is a branch of artificial intelligence based on the
    idea that systems can learn from data, identify patterns and make
    decisions with minimal human intervention.
"""
metadata = {"title": "Introduction to Machine Learning", "url":
    "https://learn.microsoft.com/en-us/training/modules/" +
    "introduction-to-machine-learning"}
memory.add_texts([text], metadatas=[metadata])

text2 = """
    Artificial intelligence (AI) is the simulation of human intelligence in machines
    that are programmed to think like humans and mimic their actions.
    The term may also be applied to any machine that exhibits traits associated with
    a human mind such as learning and problem-solving.
    AI research has been highly successful in developing effective techniques for
    solving a wide range of problems, from game playing to medical diagnosis.
"""
metadata2 = {"title": "Artificial Intelligence for Beginners", "url":
"https://microsoft.github.io/AI-for-Beginners"}
memory.add_texts([text2], metadatas=[metadata2])

query = "What is the relationship between AI and machine learning?"
results = memory.similarity_search(query, k=3)

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()

input_message = {"type": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, {}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print([r.page_content for r in results])