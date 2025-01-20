import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from transformers import pipeline
import numpy as np
import time

#Gradio example for recording audio and transcribing it with better UI and summary generation

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Initialize the transcriber
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

def transcribe_and_summarize_continuously(state, audio_chunk):
    if state is None:
        state = {
            "audio_buffer": np.array([]),  # Buffer to store audio chunks
            "full_transcription": "",      # Full accumulated transcription
            "last_summary_time": time.time(),
            "summary_text": ""             # Summary text
        }

    sr, y = audio_chunk

    # Process audio chunk
    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert stereo to mono
    y = y.astype(np.float32)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y /= max_val  # Normalize
    else:
        y = np.zeros_like(y)

    # Silence detection using a threshold
    silence_threshold = 0.1  # Adjust this based on microphone noise levels
    if np.max(np.abs(y)) < silence_threshold:
        print("Silence detected in this audio chunk.")
        return state, state["full_transcription"], state["summary_text"]

    print(f"Audio data: {y[:10]} (First 10 samples)")

    # Add the current audio chunk to the buffer
    state["audio_buffer"] = np.concatenate([state["audio_buffer"], y])
    transcription = transcriber({"sampling_rate": sr, "raw": state["audio_buffer"]})["text"]

    # Log the transcription
    print(f"Transcription: '{transcription}'")

    # Append new transcription to the full transcription if it's not empty
    if transcription.strip():
        state["full_transcription"] += f" {transcription}"

    # Clear the audio buffer after processing
    state["audio_buffer"] = np.array([])

    # Generate a summary every 5 seconds
    if time.time() - state["last_summary_time"] > 5:
        prompt = f"Generate a live summary of the following text:\n\n{state['full_transcription']}"
        response = llm.invoke([HumanMessage(content=prompt)])
        state["summary_text"] = response.content
        state["last_summary_time"] = time.time()

    return state, state["full_transcription"], state["summary_text"]


with gr.Blocks() as demo:
    gr.Markdown("## Continuous Transcription and Summary")

    state = gr.State()

    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], streaming=True, label="Microphone Input")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transcription")
            transcription_box = gr.Textbox(show_label=False, lines=8)

        with gr.Column():
            gr.Markdown("### Summary")
            summary_box = gr.Textbox(show_label=False, lines=8)

    # Set up the streaming input and outputs
    audio_in.stream(
        fn=transcribe_and_summarize_continuously,
        inputs=[state, audio_in],
        outputs=[state, transcription_box, summary_box],
    )

demo.launch()
