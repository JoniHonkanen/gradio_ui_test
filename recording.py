import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from transformers import pipeline
import numpy as np
import time

# Gradio example for recording audio and transcribing it

# Load environment variables once
load_dotenv()
# Shared LLM instance
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Global variables for summary timing
last_summary_time = time.time()  # Tracks the last time the summary was generated
summary_interval = 5  # Generate summary every 5 seconds
global_transcription = ""  # Stores the full transcription


def transcribe_and_summarize_continuously(state, audio_chunk):
    sr, y = audio_chunk

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # Combine current audio chunk with previous audio state
    if state is not None:
        state = np.concatenate([state, y])
    else:
        state = y

    # Transcribe the speech
    transcription = transcriber({"sampling_rate": sr, "raw": state})["text"]

    # Generate a summary using OpenAI LLM
    summary_prompt = (
        f"Generate a live summary of the following text:\n\n{transcription}"
    )
    summary = llm([HumanMessage(content=summary_prompt)]).content

    return state, transcription, summary


demo = gr.Interface(
    fn=transcribe_and_summarize_continuously,
    inputs=["state", gr.Audio(sources=["microphone"], streaming=True)],
    outputs=[
        gr.State(),  # Keeps audio state
        gr.Textbox(label="Transcription"),  # RECORDING
        gr.Textbox(label="Summary"),  # AI SUMMARY OF RECORDING
    ],
    flagging_mode="never",
    live=True,
)
demo.launch()  # .launch(share=True) # To share the app
