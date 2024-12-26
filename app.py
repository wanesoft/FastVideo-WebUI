import gradio as gr
import subprocess
import time
import os


VIDEO_DIR = "outputs_video/hunyuan_quant/nf4/"
os.makedirs(os.path.dirname(VIDEO_DIR), exist_ok=True)

def get_latest_video_file():
    if not os.path.exists(VIDEO_DIR):
        return None
    files = [
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if os.path.isfile(os.path.join(VIDEO_DIR, f)) and f.endswith(".mp4")
    ]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def generate_video(prompt, resolution, num_frames):
    command = [
        "torchrun", "--nnodes=1", "--nproc_per_node=1", "--master_port", "12345",
        "third_party/FastVideo/fastvideo/sample/sample_t2v_diffusers_hunyuan.py",
        "--height", "720",
        "--width", "1280",
        "--num_frames", str(num_frames),
        "--num_inference_steps", "6",
        "--guidance_scale", "1",
        "--embedded_cfg_scale", "6",
        "--flow_shift", "17",
        "--flow-reverse",
        "--prompt", prompt,
        "--seed", "1024",
        "--output_path", VIDEO_DIR,
        "--model_path", "data/FastHunyuan-diffusers",
        "--quantization", "nf4",
        "--cpu_offload",
    ]

    print(f"Starting command: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

    output = ""
    for line in process.stdout:
        output += line
        time.sleep(0.1) # Небольшая задержка
        yield gr.update(value=output), gr.update(value=None)

    process.wait()
    if process.returncode != 0:
        yield gr.update(value=output + f"\nError ocured: {process.returncode}"), gr.update(value=None)
    else:
        latest_video = get_latest_video_file()
        yield gr.update(value=output), gr.update(value=latest_video)

with gr.Blocks() as demo:
    gr.Markdown("## FastVideo WebUI")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", lines=3)
        resolution = gr.Dropdown(["720p"], label="Resolution", value="720p")
        num_frames = gr.Slider(minimum=20, maximum=45, step=1, value=45, label="Frames number")
    
    btn = gr.Button("Generate video")

    video_out = gr.Video(label="Result", show_label=True)
    output = gr.Textbox(label="Log", lines=5)

    btn.click(fn=generate_video, inputs=[prompt, resolution, num_frames], outputs=[output, video_out])

demo.launch()