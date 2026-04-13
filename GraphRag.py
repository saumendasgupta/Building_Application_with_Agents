import re
from typing import List

import networkx as nx
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama


DOCUMENTS = [
    "https://www.gutenberg.org/cache/epub/103/pg103.txt"
]

client = ChatOllama(model="llama3.2", temperature=0)


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    """Run a chat completion and normalize content to plain text."""
    response = client.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def load_documents(document_sources: List[str], max_chars: int = 3600) -> List[str]:
    """Fetch URLs and trim content so local models can process quickly."""
    loaded = []
    for source in document_sources:
        if source.startswith("http://") or source.startswith("https://"):
            try:
                resp = requests.get(source, timeout=20)
                resp.raise_for_status()
                loaded.append(resp.text[:max_chars])
            except requests.RequestException as exc:
                print(f"Skipping {source}: {exc}")
        else:
            loaded.append(source[:max_chars])
    return loaded


# 1. Source Documents → Text Chunks
def split_documents_into_chunks(documents, chunk_size=700, overlap_size=150):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks


def limit_chunks(chunks: List[str], max_chunks: int = 6) -> List[str]:
    """Bound chunk count so local-model demos finish quickly."""
    return chunks[:max_chunks]


# 2. Text Chunks → Element Instances
def extract_elements_from_chunks(chunks):
    elements = []
    for index, chunk in enumerate(chunks):
        print(f"Chunk index {index} of {len(chunks)}:")
        entities_and_relations = _llm_text(
            "Extract key entities and relationships from the text. "
            "Return plain text with two sections:\n"
            "Entities:\n- entity\nRelationships:\n- source -> relation -> target",
            chunk,
        )
        print(entities_and_relations)
        elements.append(entities_and_relations)
    return elements


# 3. Element Instances → Element Summaries
def summarize_elements(elements):
    summaries = []
    for index, element in enumerate(elements):
        print(f"Element index {index} of {len(elements)}:")
        summary = _llm_text(
            "Normalize this extraction into exactly this format:\n"
            "Entities:\n- ...\nRelationships:\n- A -> relation -> B",
            element,
        )
        print("Element summary:", summary)
        summaries.append(summary)
    return summaries


# 4. Element Summaries → Graph Communities
def build_graph_from_summaries(summaries):
    graph = nx.Graph()
    for index, summary in enumerate(summaries):
        print(f"Summary index {index} of {len(summaries)}:")
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        for line in lines:
            cleaned = line.strip().replace("**", "")
            lowered = cleaned.lower()
            if lowered.startswith("entities:"):
                entities_section = True
                relationships_section = False
                continue
            if lowered.startswith("relationships:"):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and cleaned:
                item = re.sub(r"^[-*\d.\s]+", "", cleaned).strip()
                if item:
                    graph.add_node(item)
                continue
            if relationships_section and cleaned:
                item = re.sub(r"^[-*\d.\s]+", "", cleaned).strip()
                parts = [p.strip() for p in item.split("->")]
                if len(parts) >= 3:
                    source = parts[0]
                    target = parts[-1]
                    relation = " -> ".join(parts[1:-1]).strip()
                    graph.add_node(source)
                    graph.add_node(target)
                    graph.add_edge(source, target, label=relation)
    return graph


# 5. Graph Communities → Community Summaries
def detect_communities(graph):
    # Connected components are a stable fallback when graph-tool/cdlib is unavailable.
    communities = [list(component) for component in nx.connected_components(graph)]
    print("Communities from detect_communities:", communities)
    return communities


def summarize_communities(communities, graph):
    community_summaries = []
    for index, community in enumerate(communities):
        print(f"Summarize Community index {index} of {len(communities)}:")
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)
        summary = _llm_text(
            "Summarize this entity community in 3-4 bullet points.",
            description,
        )
        community_summaries.append(summary)
    return community_summaries


# 6. Community Summaries → Community Answers → Global Answer
def generate_answers_from_communities(community_summaries, query):
    intermediate_answers = []
    for index, summary in enumerate(community_summaries):
        print(f"Summary index {index} of {len(community_summaries)}:")
        answer = _llm_text(
            "Answer the query only using the summary. If uncertain, say so.",
            f"Query: {query}\n\nSummary: {summary}",
        )
        print("Intermediate answer:", answer)
        intermediate_answers.append(answer)

    final_answer = _llm_text(
        "Combine the intermediate answers into one concise response.",
        f"Intermediate answers:\n{intermediate_answers}",
    )
    return final_answer


# Putting It All Together
def graph_rag_pipeline(documents, query, chunk_size=700, overlap_size=150, max_chunks: int = 6):
    loaded_documents = load_documents(documents)
    if not loaded_documents:
        raise RuntimeError("No documents could be loaded.")

    # Step 1: Split documents into chunks
    chunks = split_documents_into_chunks(
        loaded_documents, chunk_size, overlap_size)
    chunks = limit_chunks(chunks, max_chunks=max_chunks)
    print(f"Using {len(chunks)} chunks for this run")

    # Step 2: Extract elements from chunks
    elements = extract_elements_from_chunks(chunks)

    # Step 3: Summarize elements
    summaries = summarize_elements(elements)

    # Step 4: Build graph and detect communities
    graph = build_graph_from_summaries(summaries)
    print(f"graph: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Graph is empty; cannot continue.")

    communities = detect_communities(graph)
    print("communities count:", len(communities))
    # Step 5: Summarize communities
    community_summaries = summarize_communities(communities, graph)

    # Step 6: Generate answers from community summaries
    final_answer = generate_answers_from_communities(
        community_summaries, query)

    return final_answer


if __name__ == "__main__":
    query = "What are the main themes in these documents?"
    print("Query:", query)
    answer = graph_rag_pipeline(DOCUMENTS, query)
    print("Answer:", answer)