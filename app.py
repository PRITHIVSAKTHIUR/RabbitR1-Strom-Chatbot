import os
import re
import gradio as gr
import edge_tts
import asyncio
import time
import tempfile
from huggingface_hub import InferenceClient

DESCRIPTION = """ # <center><b>Rabbit R1 🐰</b></center>
        ### <center>Rabbit’s Little Walkie-Talkie 🥤
        ### <center>Voice 2 Voice Coming Soon 🚧 </center>
        """

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

system_instructions = "[INST] Answers by 🐰🚀, Keep conversation very short, clear, friendly and concise."

async def generate(prompt):
    generate_kwargs = dict(
        temperature=0.6,
        max_new_tokens=256,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )
    formatted_prompt = system_instructions + prompt + "[/INST]"
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=True)
    output = ""
    for response in stream:
        output += response.token.text

    communicate = edge_tts.Communicate(output)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    yield tmp_path

with gr.Blocks(css="style.css") as demo:    
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        user_input = gr.Textbox(label="Prompt")
        input_text = gr.Textbox(label="Input Text", elem_id="important")
        output_audio = gr.Audio(label="Audio", type="filepath",
                        interactive=False,
                        autoplay=True,
                        elem_classes="audio")
    with gr.Row():
        translate_btn = gr.Button("Response")
        translate_btn.click(fn=generate, inputs=user_input,
                            outputs=output_audio, api_name="translate")        

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
