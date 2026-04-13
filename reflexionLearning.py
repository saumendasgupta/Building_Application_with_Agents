from typing import Any, Dict, List
from typing_extensions import TypedDict
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage
import os

llm = ChatOllama(model="llama3.2", temperature=0)

reflections = []

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

reflexion_prompt = f"""You will be given the history of a past experience in which you were 
placed in an environment and given a task to complete. You were unsuccessful in 
completing the task. Do not summarize your environment, but rather think about 
the strategy and path you took to attempt to complete the task. 
Devise a concise, new plan of action that accounts for your mistake with reference 
to specific actions that you should have taken. For example, if you tried A and B but forgot C, 
then devise a plan to achieve C with environment-specific actions. You will need this 
later when you are solving the same task. Give your plan after "Plan". 

Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.]
Observation: OK.

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The item has options 'variety pack' and '0.8 ounce (pack of 24)' and seems good to buy.]
Observation: OK.

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]

STATUS: FAIL

Plan:
"""


def get_completion(prompt: str) -> str:
    # Wraps our `call_model` helper for one‐off text completions
    result = llm.invoke([{"role": "user", "content": prompt}])
    content = result.content
    return content if isinstance(content, str) else str(content)

def _generate_reflection_query(trial_log: str, recent_reflections: List[str]):
    history = "\n\n".join(recent_reflections)
    return f'''{history}
        {trial_log}
        Based on the above, what plan would you follow next? Plan:'''

def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]):
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    if not full_log.strip():
        env_logs: List[str] = [reflexion_prompt]
    else:
        env_logs = full_log.split('#####\n\n#####')

    # If caller did not pass configs, create defaults so memory can be generated.
    if not env_configs:
        env_configs = [
            {"is_success": False, "skip": False, "memory": []}
            for _ in env_logs
        ]

    assert len(env_logs) == len(env_configs), print(f'bad: {env_logs}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory = env['memory']
            reflection_query = _generate_reflection_query(env_logs[i], memory)
            reflection = get_completion(reflection_query)
            env_configs[i]['memory'] += [reflection]

    return env_configs


def write_clean_memory(output_path: str, env_configs: List[Dict[str, Any]]) -> None:
    """Write reflections in a clean, readable format."""
    lines: List[str] = []
    for i, env in enumerate(env_configs, start=1):
        lines.append(f"=== Environment {i} ===")
        memory_items = env.get("memory", [])
        if not memory_items:
            lines.append("No reflections generated.")
        else:
            for j, item in enumerate(memory_items, start=1):
                lines.append(f"Plan {j}:")
                lines.append(item.strip())
                lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content)


def extract_latest_plan(result: Dict[str, Any]) -> str:
    messages = result.get("messages", [])
    for message in reversed(messages):
        text = _message_to_text(message).strip()
        if text:
            return text
    return "No plan generated."


def write_latest_plan(output_path: str, plan: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Environment 1 ===\n")
        f.write("Plan 1:\n")
        f.write(plan.strip() + "\n")


def find_or_create_trial_log() -> str:
    """Find reflection.log in common locations; create one if missing."""
    candidates = [
        os.path.join(os.getcwd(), "reflection.log"),
        os.path.join(os.path.dirname(__file__), "reflection.log"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Create a default log in cwd so memory generation can proceed.
    default_path = candidates[0]
    with open(default_path, "w", encoding="utf-8") as f:
        f.write(reflexion_prompt.strip() + "\n")
    return default_path

builder = StateGraph(MessagesState)
builder.add_node("reflexion", call_model)
builder.add_edge(START, "reflexion")
graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(reflexion_prompt)
            ]
        }
    )
    reflections.append(result)
    print(result)

    trial_log_path = find_or_create_trial_log()
    print(f"Using log file: {trial_log_path}")

    latest_plan = extract_latest_plan(result)

    output_path = os.path.join(os.path.dirname(trial_log_path), "reflection_memory.txt")
    write_latest_plan(output_path, latest_plan)

    # Keep reflection.log synchronized with latest generated plan.
    write_latest_plan(trial_log_path, latest_plan)
    print(f"Memory written to: {output_path}")
    print(f"Log updated at: {trial_log_path}")