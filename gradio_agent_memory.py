import uuid
import gradio as gr
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


class State(TypedDict):
    user_input: str
    answer: str


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

with gr.Blocks(title="Multi agent system - gradio") as demo:
    memory = MemorySaver()
    # Compile
    graph = graph_builder.compile(
        checkpointer=memory,
    )
    msg = gr.Textbox(placeholder="Enter your question...")
    output = gr.Textbox(label="Generated question", interactive=False)
    output2 = gr.Textbox(label="Generated question", interactive=False)
    thread_id = gr.State(None)  # this is needed for persistent user state (memory)
    state = gr.State(None)

    def process_request(msg, thread_id, state):
        if not thread_id:
            thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = {"user_input": msg}  # Statessa pitää olla jotain...
        graph_state = graph.invoke(
            state, config
        )  # pistää agentit käyntiin ja palauttaa staten

        return (
            graph_state["answer"],
            graph_state["user_input"],
        )  # pitää täsmätä outputtien määrään

    msg.submit(
        fn=process_request, inputs=[msg, thread_id, state], outputs=[output, output2]
    )
    with gr.Row():
        submit_button = gr.Button("Submit")
    submit_button.click(
        fn=process_request,
        inputs=[msg, thread_id, state],
        outputs=[output, output2],
        show_progress="true",
    )


demo.launch()
