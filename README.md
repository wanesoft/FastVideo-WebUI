# FastVideo-WebUI

This is a Gradio-based WebUI for the [FastVideo](https://github.com/hao-ai-lab/FastVideo) project

The code has been tested on RTX 4090 (24GB VRAM) with Python 3.10.10, CUDA 12.5, and Ubuntu 22

## How to use

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
cd third_party/FastVideo && ./env_setup.sh fastvideo && cd ../..
pip install -r requirements.txt

# Download models
python third_party/FastVideo/scripts/huggingface/download_hf.py --repo_id=FastVideo/FastHunyuan-diffusers --local_dir=data/FastHunyuan-diffusers --repo_type=model

# Run
python app.py
```

## Thanks

We would like to express our sincere gratitude to the developers of the [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) projects for their incredible work and for making this WebUI possible
