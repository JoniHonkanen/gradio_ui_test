import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

#Basic gradio example with streaming

# Load environment variables once
load_dotenv()
# Shared LLM instance
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Global message history (shared across all sessions for simplicity)
message_history = []

# Add a system prompt
system_prompt = SystemMessage(
    content="You are a helpful assistant that is helping in this streaming example with Gradio."
)
# Preload the system prompt in the message history
message_history.append(system_prompt)


def test_stream(prompt):
    # Append the user's message to the history
    message_history.append(HumanMessage(content=prompt))
    stream = llm.stream(message_history)
    result = ""
    for response in stream:
        result += response.content
        yield result

    # Append the AI's response to the history
    message_history.append(AIMessage(content=result))
    print(message_history)


demo = gr.Interface(
    fn=test_stream,
    inputs=[
        gr.Textbox("World", label="Name", info="Enter your name here!"),
    ],
    outputs="text",
    flagging_mode="never",
)
demo.launch()  # .launch(share=True) # To share the app
