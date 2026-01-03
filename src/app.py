import gradio as gr
from transformers.tools.base import launch_gradio_demo

# Version: 2.0.0 - Fixed deprecated transformers.tools import, updated Gradio
iface = gr.Interface(
    fn=launch_gradio_demo,
    inputs=gr.Textbox(lines=3, label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="text-to-video",
    description="text to video"
)

if __name__ == "__main__":
    iface.launch()

