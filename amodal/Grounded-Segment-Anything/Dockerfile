FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-12.8/

RUN mkdir -p /home/ubuntu22-zy/occlusion/amodal/Grounded-Segment-Anything

RUN apt-get update && apt-get install --no-install-recommends -y \
    wget ffmpeg libsm6 libxext6 git nano vim \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

COPY . /home/ubuntu22-zy/occlusion/amodal/Grounded-Segment-Anything/

WORKDIR /home/ubuntu22-zy/occlusion/amodal/Grounded-Segment-Anything
RUN python -m pip install --no-cache-dir -e segment_anything

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel
RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

WORKDIR /home/ubuntu22-zy/occlusion/amodal
RUN pip install --no-cache-dir \
    diffusers[torch] opencv-python pycocotools matplotlib \
    onnxruntime onnx ipykernel scipy gradio openai
