.PHONY: test smoke-cpu-build smoke-cpu-up smoke-cpu-health smoke-cpu-down docker-gpu-build docker-gpu-push

IMAGE ?= jwindsor1/open-speech
TAG ?= dev
GPU_TAG ?= $(IMAGE):cuda-$(TAG)
CPU_TAG ?= $(IMAGE):cpu

# Fast local gate. Install dev deps first with: uv sync --frozen --extra cpu --extra dev
test:
	pytest -q

# CPU Docker smoke workflow. Avoids the CUDA image build path.
smoke-cpu-build:
	docker compose -f docker-compose.cpu.yml build

smoke-cpu-up:
	docker compose -f docker-compose.cpu.yml up -d

smoke-cpu-health:
	curl -fsSk https://localhost:8100/health

smoke-cpu-down:
	docker compose -f docker-compose.cpu.yml down

# Intended for a machine with Docker and enough disk for the CUDA image.
docker-gpu-build:
	docker build -f Dockerfile -t $(GPU_TAG) -t $(IMAGE):latest .

docker-gpu-push:
	docker push $(GPU_TAG)
	docker push $(IMAGE):latest
