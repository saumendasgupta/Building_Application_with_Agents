from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)


class SupportState(TypedDict, total=False):
    user_message: str
    user_id: str
    issue_type: str
    sub_issue: str
    step_result: str
    response: str


# 1. Node definitions
def categorize_issue(state: SupportState) -> SupportState:
    prompt = (
        "Classify this support request as exactly one word: 'billing' or 'technical'.\n\n"
        f"Message: {state['user_message']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)]).content
    content = response if isinstance(response, str) else (response[0] if isinstance(response, list) else str(response))
    kind = str(content).strip().lower()
    print(f"Categorized issue type: '{kind}'")
    if "billing" in kind:
        kind = "billing"
    elif "technical" in kind:
        kind = "technical"
    else:
        kind = "technical"  # fallback
    return {**state, "issue_type": kind}


def classify_billing_sub_issue(state: SupportState) -> SupportState:
    msg = state["user_message"].lower()
    sub = "invoice" if "invoice" in msg else "refund"
    print(f"Classified billing sub-issue: '{sub}'")
    return {**state, "sub_issue": sub}


def classify_technical_sub_issue(state: SupportState) -> SupportState:
    msg = state["user_message"].lower()
    sub = "login" if "login" in msg else "performance"
    print(f"Classified technical sub-issue: '{sub}'")
    return {**state, "sub_issue": sub}


def handle_invoice(state: SupportState) -> SupportState:
    return {**state, "step_result": f"Invoice details for {state['user_id']}"}


def handle_refund(state: SupportState) -> SupportState:
    return {**state, "step_result": "Refund process initiated"}


def handle_login(state: SupportState) -> SupportState:
    return {**state, "step_result": "Password reset link sent"}


def handle_performance(state: SupportState) -> SupportState:
    return {**state, "step_result": "Performance metrics analyzed"}


def summarize_response(state: SupportState) -> SupportState:
    details = state.get("step_result", "")
    prompt = f"Write a concise customer support reply based on: {details}"
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content if hasattr(response, 'content') else str(response)
    summary = content[0] if isinstance(content, list) else content
    summary = summary.strip() if isinstance(summary, str) else str(summary).strip()
    return {**state, "response": summary}


# 2. Build the graph
graph_builder = StateGraph(SupportState)

graph_builder.add_node("categorize_issue", categorize_issue)
graph_builder.add_node("classify_billing_sub_issue", classify_billing_sub_issue)
graph_builder.add_node("classify_technical_sub_issue", classify_technical_sub_issue)
graph_builder.add_node("handle_invoice", handle_invoice)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_login", handle_login)
graph_builder.add_node("handle_performance", handle_performance)
graph_builder.add_node("summarize_response", summarize_response)

graph_builder.add_edge(START, "categorize_issue")


def top_router(state: SupportState) -> str:
    return "billing" if state.get("issue_type") == "billing" else "technical"


graph_builder.add_conditional_edges(
    "categorize_issue",
    top_router,
    {
        "billing": "classify_billing_sub_issue",
        "technical": "classify_technical_sub_issue",
    },
)


def billing_router(state: SupportState) -> str:
    return "invoice" if state.get("sub_issue") == "invoice" else "refund"


graph_builder.add_conditional_edges(
    "classify_billing_sub_issue",
    billing_router,
    {
        "invoice": "handle_invoice",
        "refund": "handle_refund",
    },
)


def tech_router(state: SupportState) -> str:
    return "login" if state.get("sub_issue") == "login" else "performance"


graph_builder.add_conditional_edges(
    "classify_technical_sub_issue",
    tech_router,
    {
        "login": "handle_login",
        "performance": "handle_performance",
    },
)

graph_builder.add_edge("handle_invoice", "summarize_response")
graph_builder.add_edge("handle_refund", "summarize_response")
graph_builder.add_edge("handle_login", "summarize_response")
graph_builder.add_edge("handle_performance", "summarize_response")
graph_builder.add_edge("summarize_response", END)

graph = graph_builder.compile()


# 3. Execute the graph
if __name__ == "__main__":
    initial_state: SupportState = {
        "user_message": "Hi, I need help with my invoice and possibly a refund.",
        "user_id": "U1234",
    }
    result = graph.invoke(initial_state)
    print(result["response"])