import os
import requests
import logging
import json
import faiss
import numpy as np

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama


# Models
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or "mxbai-embed-large"

# Tool descriptions used for retrieval
tool_descriptions = {
    "query_wolfram_alpha": "Use Wolfram Alpha to compute mathematical expressions or retrieve factual results.",
    "trigger_zapier_webhook": "Trigger a Zapier webhook to execute predefined automation workflows.",
    "send_slack_message": "Send a message to a Slack channel.",
}

tool_names = list(tool_descriptions.keys())
desc_texts = [tool_descriptions[name] for name in tool_names]

# Build embeddings + FAISS index
tool_embeddings = np.array(embeddings.embed_documents(desc_texts), dtype="float32")
faiss.normalize_L2(tool_embeddings)

dimension = tool_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine via normalized inner product
index.add(tool_embeddings)

index_to_tool = {i: tool_names[i] for i in range(len(tool_names))}


@tool
def query_wolfram_alpha(expression: str) -> str:
    """
    Query Wolfram Alpha to compute expressions or retrieve information.
    Args: expression (str): The mathematical expression or query to evaluate.
    Returns: str: The result of the computation or the retrieved information.
    """
    api_url = f'''https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid=3RKGL6A2EV'''
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.text
        else: raise ValueError(f"Wolfram Alpha API Error:{response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to query Wolfram Alpha: {e}")


@tool
def send_slack_message(channel: str, message: str) -> str:
    """ Send a message to a specified Slack channel.
    Args:
    channel (str): The Slack channel ID or name where the message will be sent.
    message (str): The content of the message to send.
    Returns:
    str: Confirmation message upon successful sending of the Slack message.
    Raises: ValueError: If the API request fails or returns an error.
    """
    api_url = "https://slack.com/api/chat.postMessage"
    headers = { "Authorization": "Bearer YOUR_SLACK_BOT_TOKEN","Content-Type": "application/json" }
    payload = { "channel": channel, "text": message }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        if response.status_code == 200 and response_data.get("ok"):
            return f"Message successfully sent to Slack channel '{channel}'."
        else:
            error_msg = response_data.get("error", "Unknown error")
            raise ValueError(f"Slack API Error: {error_msg}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f'''Failed to send message to Slack channel"{channel}": {e}''')


@tool
def trigger_zapier_webhook(zap_id: str, payload: dict) -> str:
    """ Trigger a Zapier webhook to execute a predefined Zap.
    Args:
    zap_id (str): The unique identifier for the Zap to be triggered.
    payload (dict): The data to send to the Zapier webhook.
    Returns:
    str: Confirmation message upon successful triggering of the Zap.
    Raises: ValueError: If the API request fails or returns an error.
    """
    zapier_webhook_url = f"https://hooks.zapier.com/hooks/catch/{zap_id}/"
    try:
        response = requests.post(zapier_webhook_url, json=payload)
        if response.status_code == 200:
            return f"Zapier webhook '{zap_id}' successfully triggered."
        else:
            raise ValueError(f"Zapier API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to trigger Zapier webhook '{zap_id}': {e}")

tool_registry = {
    "query_wolfram_alpha": query_wolfram_alpha,
    "trigger_zapier_webhook": trigger_zapier_webhook,
    "send_slack_message": send_slack_message,
}


def select_tool(query: str, top_k: int = 1) -> list[str]:
    query_embedding = np.array(embeddings.embed_query(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    _, I = index.search(query_embedding, top_k)
    return [index_to_tool[idx] for idx in I[0] if idx in index_to_tool]


def determine_parameters(query: str, tool_name: str) -> dict:
    schema_map = {
        "query_wolfram_alpha": {"expression": "string"},
        "trigger_zapier_webhook": {"zap_id": "string", "payload": "object"},
        "send_slack_message": {"channel": "string", "message": "string"},
    }

    prompt = f"""
Return ONLY valid JSON (no markdown) for tool parameters.

Tool: {tool_name}
Required schema: {json.dumps(schema_map[tool_name])}
User query: {query}
"""

    ai_msg = llm.invoke([HumanMessage(content=prompt)])
    content = ai_msg.content if isinstance(ai_msg.content, str) else ""
    raw = (content or "").strip()

    # tolerate accidental ```json fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        params = json.loads(raw)
        if isinstance(params, dict):
            return params
    except Exception:
        pass

    # Safe fallback defaults
    if tool_name == "query_wolfram_alpha":
        return {"expression": query}
    if tool_name == "trigger_zapier_webhook":
        return {"zap_id": "3RKGL6A2EV", "payload": {"data": query}}
    if tool_name == "send_slack_message":
        return {"channel": "#general", "message": query}
    return {}


if __name__ == "__main__":
    user_query = "Solve this equation: 2x + 3 = 7 where x is 5"

    selected_tools = select_tool(user_query, top_k=1)
    tool_name = selected_tools[0] if selected_tools else None

    if not tool_name:
        print("No tool was selected.")
    else:
        args = determine_parameters(user_query, tool_name)
        try:
            tool_fn = tool_registry[tool_name]
            result = tool_fn.invoke(args)
            print(f"Selected tool: {tool_name}")
            print(f"Args: {args}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error invoking tool '{tool_name}': {e}")