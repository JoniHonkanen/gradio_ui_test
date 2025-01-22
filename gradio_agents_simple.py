import gradio as gr
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class State(TypedDict):
    user_input:str
    answer:str


graph_builder = StateGraph(State)


def answer_generator(state: State) -> State:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Answer for: {state['user_input']}",
            }
        ],
    )

    state["answer"] = response.choices[0].message.content
    return state


# A single node
graph_builder.add_node("answer_generator", answer_generator)

# Start -> answer -> End
graph_builder.add_edge(START, "answer_generator")
graph_builder.add_edge("answer_generator", END)

# Compile
graph = graph_builder.compile()

with gr.Blocks(title="Multi agent system - gradio") as demo:
    msg = gr.Textbox(placeholder="Enter your question...")
    output = gr.Textbox(label="Generated question", interactive=False)
    output2 = gr.Textbox(label="Generated question", interactive=False)

    def process_request(txt):
        state = {"user_input": txt} # Statessa pitää olla jotain...
        graph_state = graph.invoke(state)  # pistää agentit käyntiin ja palauttaa staten
        return graph_state["answer"], graph_state["user_input"]  # pitää täsmätä outputtien määrään

    msg.submit(fn=process_request, inputs=msg, outputs=[output, output2])

demo.launch()
