#!/bin/bash
set -e

python -m pip install gdown

# Set up diffusers
cd diffusers
python -m pip install --no-build-isolation -e ".[torch]"
cd ..

# Download Grounding DINO and SAM model checkpoints
cd Grounded-Segment-Anything
wget -nc https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# Set up InstaOrder
cd InstaOrder
if [ ! -f InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar ]; then
    gdown 1_GEmCmofLSkJZnidfp4vsQb2Nqq5aqBU
    python -m zipfile -e InstaOrder_ckpt.zip .
    rm InstaOrder_ckpt.zip
    rm -rf __MACOSX
fi
cd ..

# Set up LaMa
if python -c "import torch, torchvision, torchaudio" >/dev/null 2>&1; then
    echo "PyTorch, torchvision, and torchaudio are already installed; skipping conda install."
else
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
fi

# Install requirements
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
