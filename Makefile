# notlame-train Makefile
# Run `make help` for usage

SHELL := /bin/bash
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# Directories
DATA_RAW := data/raw
DATA_PROCESSED := data/processed
CHECKPOINTS := checkpoints
MODELS := models
TEST_SAMPLES := tests/samples

# Training config
BATCH_SIZE ?= 32
LEARNING_RATE ?= 1e-4
TRAIN_STEPS ?= 100000
DEVICE ?= cuda
WORKERS ?= 4
MODEL_VARIANT ?= default
CACHE ?= 1
FRAMES ?= 4

# Evaluation config
BITRATE ?= 192

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

.PHONY: help setup install download validate prepare train evaluate export clean test test-losses all

# Default target
help:
	@echo ""
	@echo "$(GREEN)notlame-train$(NC) - Neural MP3 encoder training pipeline"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make setup              Create venv and install dependencies"
	@echo "  make install            Install dependencies only"
	@echo ""
	@echo "$(YELLOW)Data Download:$(NC)"
	@echo "  make download           Download test samples (quick, ~50MB)"
	@echo "  make download-gtzan     Download GTZAN music (1.2GB WAV, 8hrs)"
	@echo "  make download-fma       Download FMA-small music (7.2GB, 66hrs)"
	@echo "  make download-librispeech  Download LibriSpeech (6.3GB, 100hrs)"
	@echo "  make download-all       Download test + gtzan + librispeech-dev"
	@echo "  make download-list      List all available datasets"
	@echo ""
	@echo "$(YELLOW)Data Pipeline:$(NC)"
	@echo "  make validate           Validate audio files in DATA_RAW"
	@echo "  make prepare            Convert audio to MDCT frames"
	@echo ""
	@echo "$(YELLOW)Training:$(NC)"
	@echo "  make train          Train the model"
	@echo "  make train-resume   Resume training from latest checkpoint"
	@echo "  make tensorboard    Start TensorBoard server"
	@echo ""
	@echo "$(YELLOW)Evaluation & Export:$(NC)"
	@echo "  make evaluate       Evaluate model against LAME"
	@echo "  make export         Export model to ONNX"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  make test           Run module tests"
	@echo "  make clean          Remove generated files"
	@echo "  make clean-all      Remove everything including venv"
	@echo "  make all            Full pipeline: setup → export"
	@echo ""
	@echo "$(YELLOW)Configuration (override with VAR=value):$(NC)"
	@echo "  DATA_RAW=$(DATA_RAW)"
	@echo "  DATA_PROCESSED=$(DATA_PROCESSED)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  LEARNING_RATE=$(LEARNING_RATE)"
	@echo "  TRAIN_STEPS=$(TRAIN_STEPS)"
	@echo "  DEVICE=$(DEVICE)"
	@echo "  MODEL_VARIANT=$(MODEL_VARIANT)"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make setup && make download && make prepare && make train"
	@echo "  make train BATCH_SIZE=64 DEVICE=cuda TRAIN_STEPS=200000"
	@echo "  make validate DATA_RAW=/path/to/my/audio"
	@echo ""

# =============================================================================
# Setup
# =============================================================================

$(VENV)/bin/activate:
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)/bin/activate install
	@echo "$(GREEN)Setup complete!$(NC)"

install: $(VENV)/bin/activate
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install torch torchaudio numpy soundfile scipy tqdm tensorboard
	$(PIP) install onnx onnxruntime onnxscript
	$(PIP) install librosa pydub joblib matplotlib pandas PyYAML
	@echo "$(GREEN)Dependencies installed!$(NC)"

# =============================================================================
# Data Pipeline
# =============================================================================

download: $(VENV)/bin/activate
	@echo "$(GREEN)Downloading test samples...$(NC)"
	@mkdir -p $(DATA_RAW)
	$(PY) scripts/download_data.py --output $(DATA_RAW) --datasets test-samples
	@echo "$(GREEN)Download complete!$(NC)"

download-gtzan: $(VENV)/bin/activate
	@echo "$(GREEN)Downloading GTZAN music dataset (1.2GB WAV)...$(NC)"
	@mkdir -p $(DATA_RAW)
	$(PY) scripts/download_data.py --output $(DATA_RAW) --datasets gtzan
	@echo "$(GREEN)Download complete!$(NC)"

download-fma: $(VENV)/bin/activate
	@echo "$(GREEN)Downloading FMA-small music dataset (7.2GB)...$(NC)"
	@mkdir -p $(DATA_RAW)
	$(PY) scripts/download_data.py --output $(DATA_RAW) --datasets fma-small
	@echo "$(GREEN)Download complete!$(NC)"

download-fma-large: $(VENV)/bin/activate
	@echo "$(GREEN)Downloading FMA-large music dataset (93GB, 879 hours)...$(NC)"
	@mkdir -p $(DATA_RAW)
	$(PY) scripts/download_data.py --output $(DATA_RAW) --datasets fma-large
	@echo "$(GREEN)Download complete!$(NC)"

download-librispeech: $(VENV)/bin/activate
	@echo "$(GREEN)Downloading LibriSpeech speech dataset (6.3GB)...$(NC)"
	@mkdir -p $(DATA_RAW)
	$(PY) scripts/download_data.py --output $(DATA_RAW) --datasets librispeech-clean-100
	@echo "$(GREEN)Download complete!$(NC)"

download-all: $(VENV)/bin/activate
	@echo "$(GREEN)Downloading all recommended datasets...$(NC)"
	@mkdir -p $(DATA_RAW)
	$(PY) scripts/download_data.py --output $(DATA_RAW) --datasets test-samples gtzan librispeech-dev-clean
	@echo "$(GREEN)Download complete!$(NC)"

download-list: $(VENV)/bin/activate
	$(PY) scripts/download_data.py --list

validate: $(VENV)/bin/activate
	@echo "$(GREEN)Validating audio files...$(NC)"
	@if [ ! -d "$(DATA_RAW)" ] || [ -z "$$(ls -A $(DATA_RAW) 2>/dev/null)" ]; then \
		echo "$(RED)Error: No audio files in $(DATA_RAW)$(NC)"; \
		echo "Run 'make download' first or set DATA_RAW=/path/to/audio"; \
		exit 1; \
	fi
	$(PY) -m notlame_train.validate_dataset \
		--input $(DATA_RAW) \
		--output validation_report.json
	@echo "$(GREEN)Validation complete! See validation_report.json$(NC)"

prepare: $(VENV)/bin/activate
	@echo "$(GREEN)Preparing MDCT dataset...$(NC)"
	@if [ ! -f "validation_report.json" ]; then \
		echo "$(YELLOW)Warning: No validation report found, processing all files$(NC)"; \
		$(PY) scripts/prepare_dataset.py \
			--input $(DATA_RAW) \
			--output $(DATA_PROCESSED) \
			--workers $(WORKERS); \
	else \
		$(PY) scripts/prepare_dataset.py \
			--input $(DATA_RAW) \
			--output $(DATA_PROCESSED) \
			--validation-report validation_report.json \
			--workers $(WORKERS); \
	fi
	@echo "$(GREEN)Dataset preparation complete!$(NC)"

# =============================================================================
# Training
# =============================================================================

test-losses: $(VENV)/bin/activate
	@echo "$(GREEN)Testing loss functions...$(NC)"
	PYTHONPATH=. $(PY) tests/test_losses.py
	@echo ""

train: $(VENV)/bin/activate test-losses
	@echo "$(GREEN)Starting training...$(NC)"
	@if [ ! -d "$(DATA_PROCESSED)" ] || [ -z "$$(ls -A $(DATA_PROCESSED)/*.npy 2>/dev/null)" ]; then \
		echo "$(RED)Error: No processed data in $(DATA_PROCESSED)$(NC)"; \
		echo "Run 'make prepare' first"; \
		exit 1; \
	fi
	$(PY) -m notlame_train.train \
		--data-dir $(DATA_PROCESSED) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE) \
		--steps $(TRAIN_STEPS) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--model $(MODEL_VARIANT) \
		--checkpoint-dir $(CHECKPOINTS) \
		--frames $(FRAMES) \
		$(if $(filter 1,$(CACHE)),--cache,)
	@echo "$(GREEN)Training complete!$(NC)"

train-resume: $(VENV)/bin/activate
	@echo "$(GREEN)Resuming training from latest checkpoint...$(NC)"
	@if [ ! -f "$(CHECKPOINTS)/latest.pt" ]; then \
		echo "$(RED)Error: No checkpoint found at $(CHECKPOINTS)/latest.pt$(NC)"; \
		exit 1; \
	fi
	$(PY) -m notlame_train.train \
		--data-dir $(DATA_PROCESSED) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LEARNING_RATE) \
		--steps $(TRAIN_STEPS) \
		--device $(DEVICE) \
		--workers $(WORKERS) \
		--model $(MODEL_VARIANT) \
		--checkpoint-dir $(CHECKPOINTS) \
		--frames $(FRAMES) \
		--resume $(CHECKPOINTS)/latest.pt \
		$(if $(filter 1,$(CACHE)),--cache,)
	@echo "$(GREEN)Training complete!$(NC)"

tensorboard: $(VENV)/bin/activate
	@echo "$(GREEN)Starting TensorBoard on http://localhost:6006$(NC)"
	$(VENV)/bin/tensorboard --logdir runs --bind_all

# =============================================================================
# Evaluation & Export
# =============================================================================

evaluate: $(VENV)/bin/activate
	@echo "$(GREEN)Evaluating model...$(NC)"
	@if [ ! -f "$(CHECKPOINTS)/best.pt" ] && [ ! -f "$(CHECKPOINTS)/final.pt" ]; then \
		echo "$(RED)Error: No checkpoint found$(NC)"; \
		exit 1; \
	fi
	@CKPT=$$([ -f "$(CHECKPOINTS)/best.pt" ] && echo "$(CHECKPOINTS)/best.pt" || echo "$(CHECKPOINTS)/final.pt"); \
	if [ ! -d "$(TEST_SAMPLES)" ] || [ -z "$$(ls -A $(TEST_SAMPLES) 2>/dev/null)" ]; then \
		echo "$(YELLOW)Warning: No test samples in $(TEST_SAMPLES), using $(DATA_RAW)$(NC)"; \
		$(PY) -m notlame_train.evaluate \
			--checkpoint $$CKPT \
			--test-dir $(DATA_RAW) \
			--bitrate $(BITRATE) \
			--model $(MODEL_VARIANT) \
			--max-files 20 \
			--output evaluation_report.json; \
	else \
		$(PY) -m notlame_train.evaluate \
			--checkpoint $$CKPT \
			--test-dir $(TEST_SAMPLES) \
			--bitrate $(BITRATE) \
			--model $(MODEL_VARIANT) \
			--output evaluation_report.json; \
	fi
	@echo "$(GREEN)Evaluation complete! See evaluation_report.json$(NC)"

export: $(VENV)/bin/activate
	@echo "$(GREEN)Exporting model to ONNX...$(NC)"
	@if [ ! -f "$(CHECKPOINTS)/best.pt" ] && [ ! -f "$(CHECKPOINTS)/final.pt" ]; then \
		echo "$(RED)Error: No checkpoint found$(NC)"; \
		exit 1; \
	fi
	@CKPT=$$([ -f "$(CHECKPOINTS)/best.pt" ] && echo "$(CHECKPOINTS)/best.pt" || echo "$(CHECKPOINTS)/final.pt"); \
	$(PY) -m notlame_train.export_onnx \
		--checkpoint $$CKPT \
		--output $(MODELS)/psycho_v1.onnx \
		--model $(MODEL_VARIANT)
	@echo "$(GREEN)Export complete! Model saved to $(MODELS)/psycho_v1.onnx$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: $(VENV)/bin/activate
	@echo "$(GREEN)Running module tests...$(NC)"
	@echo ""
	@echo "$(YELLOW)Testing model.py...$(NC)"
	$(PY) -m notlame_train.model
	@echo ""
	@echo "$(YELLOW)Testing differentiable_mp3.py...$(NC)"
	$(PY) -m notlame_train.differentiable_mp3
	@echo ""
	@echo "$(YELLOW)Testing losses.py...$(NC)"
	$(PY) -m notlame_train.losses
	@echo ""
	@echo "$(GREEN)All tests passed!$(NC)"

test-quick: $(VENV)/bin/activate
	@echo "$(GREEN)Running quick training test...$(NC)"
	@mkdir -p $(DATA_PROCESSED)
	@$(PY) -c "import numpy as np; [np.save('$(DATA_PROCESSED)/test_{}.npy'.format(i), np.random.randn(100, 576).astype(np.float32)) for i in range(5)]"
	$(PY) -m notlame_train.train \
		--data-dir $(DATA_PROCESSED) \
		--batch-size 8 \
		--steps 50 \
		--device cpu \
		--workers 0
	@echo "$(GREEN)Quick test passed!$(NC)"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	rm -rf $(CHECKPOINTS)/*.pt
	rm -rf $(DATA_PROCESSED)/*.npy
	rm -rf $(MODELS)/*.onnx $(MODELS)/*.onnx.data
	rm -rf runs/
	rm -f validation_report.json evaluation_report.json
	rm -rf __pycache__ notlame_train/__pycache__
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	rm -rf $(VENV)
	rm -rf $(DATA_RAW)/*
	@echo "$(GREEN)Full clean complete!$(NC)"

# =============================================================================
# Full Pipeline
# =============================================================================

all: setup download validate prepare train evaluate export
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Full pipeline complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Model exported to: $(MODELS)/psycho_v1.onnx"
	@echo ""
	@echo "Next: Copy to notlame-lib:"
	@echo "  cp $(MODELS)/psycho_v1.onnx ../notlame-lib/models/"
	@echo ""

# Pipeline without download (use your own audio)
pipeline: setup validate prepare train evaluate export
	@echo ""
	@echo "$(GREEN)Pipeline complete!$(NC)"
	@echo "Model: $(MODELS)/psycho_v1.onnx"
	@echo ""
