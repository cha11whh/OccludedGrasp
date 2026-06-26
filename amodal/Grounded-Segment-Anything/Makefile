# Get version of CUDA and enable it for compilation if CUDA > 11.0
# This solves https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/53
# and https://github.com/IDEA-Research/Grounded-Segment-Anything/issues/84
# when running in Docker
USE_CUDA := 1
TORCH_CUDA_ARCH_LIST := "8.0;8.6;8.9;9.0;12.0"
BUILD_MESSAGE := "Building the image with CUDA support"


build-image:
	@echo $(BUILD_MESSAGE)
	docker build --build-arg USE_CUDA=$(USE_CUDA) \
	--build-arg TORCH_ARCH=$(TORCH_CUDA_ARCH_LIST) \
	-t gsa:v0 .

checkpoints:
ifeq (,$(wildcard ./sam_vit_h_4b8939.pth))
	wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
endif
ifeq (,$(wildcard ./groundingdino_swint_ogc.pth))
	wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
endif

run: checkpoints
	docker run -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /home/ubuntu22-zy/occlusion/amodal:/home/ubuntu22-zy/occlusion/amodal \
	-e DISPLAY=$$DISPLAY \
	--name=gsa \
	--ipc=host -it gsa:v0

run-gpu: checkpoints
	docker run --gpus all -it --rm --net=host --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v /home/ubuntu22-zy/occlusion/amodal:/home/ubuntu22-zy/occlusion/amodal \
	-e DISPLAY=$$DISPLAY \
	--name=gsa \
	--ipc=host -it gsa:v0
