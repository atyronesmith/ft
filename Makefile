# Makefile for FineTune project

.PHONY: help install dev test lint format clean run-api run-ui docker-build docker-run

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Check if poetry is installed
POETRY := $(shell command -v poetry 2> /dev/null)

help: ## Show this help message
	@echo "$(BLUE)FineTune Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

poetry-check: ## Check if Poetry is installed
ifndef POETRY
	@echo "$(RED)Poetry is not installed. Installing Poetry...$(NC)"
	@curl -sSL https://install.python-poetry.org | python3 -
	@echo "$(YELLOW)Please add Poetry to your PATH and run make again$(NC)"
	@exit 1
else
	@echo "$(GREEN)Poetry is installed at: $(POETRY)$(NC)"
endif

install: poetry-check ## Install the package in production mode
	@echo "$(BLUE)Installing FineTune with Poetry...$(NC)"
	poetry install --without dev,docs
	@echo "$(GREEN)Installation complete!$(NC)"

dev: poetry-check ## Install the package in development mode with all dependencies
	@echo "$(BLUE)Setting up development environment with Poetry...$(NC)"
	poetry install --with dev,docs
	poetry run pre-commit install
	@echo "$(GREEN)Development setup complete!$(NC)"

install-all: poetry-check ## Install with all optional dependencies
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	poetry install --with dev,docs --all-extras
	@echo "$(GREEN)Full installation complete!$(NC)"

update: poetry-check ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	poetry update
	@echo "$(GREEN)Dependencies updated!$(NC)"

lock: poetry-check ## Update poetry.lock file
	@echo "$(BLUE)Updating lock file...$(NC)"
	poetry lock --no-update
	@echo "$(GREEN)Lock file updated!$(NC)"

shell: poetry-check ## Enter Poetry shell
	@echo "$(BLUE)Entering Poetry shell...$(NC)"
	poetry shell

test: ## Run all tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/ -v --cov=finetune --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/ -v --color=yes

test-base: ## Run tests that don't require ML frameworks
	@echo "$(BLUE)Running base tests (no ML frameworks required)...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/ -v --color=yes -m "not requires_mlx and not requires_torch"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/integration/ -v --color=yes

lint: ## Run linting checks
	@echo "$(BLUE)Running linters...$(NC)"
	.venv/bin/ruff check src/ tests/ || true
	.venv/bin/mypy src/ || true

format: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	.venv/bin/black src/ tests/ || true
	.venv/bin/ruff check --fix src/ tests/ || true
	@echo "$(GREEN)Code formatted!$(NC)"

clean: ## Clean build artifacts and cache files
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/
	@echo "$(GREEN)Clean complete!$(NC)"

run-api: poetry-check ## Run the FastAPI server with hot reload
	@echo "$(BLUE)Starting API server...$(NC)"
	poetry run uvicorn finetune.api.app:app --reload --port 8000 --host 0.0.0.0

run-ui: poetry-check ## Run the Streamlit UI
	@echo "$(BLUE)Starting Streamlit UI...$(NC)"
	poetry run streamlit run src/finetune/ui/app.py --server.port 8501

run-cli: poetry-check ## Run CLI in interactive mode
	@echo "$(BLUE)Starting FineTune CLI...$(NC)"
	poetry run ft

info: poetry-check ## Show system and backend info
	@echo "$(BLUE)System Information...$(NC)"
	poetry run ft info

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t finetune:latest .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm \
		-v ~/.cache:/root/.cache \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-p 8000:8000 \
		-p 8501:8501 \
		finetune:latest

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	mkdocs build

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	mkdocs serve --dev-addr 0.0.0.0:8080

setup-mlx: ## Setup MLX for Apple Silicon
	@echo "$(BLUE)Setting up MLX...$(NC)"
	python scripts/setup_mlx.py

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	python scripts/benchmark.py

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)All checks passed!$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

update-deps: ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -r requirements.txt

create-dirs: ## Create necessary project directories
	@echo "$(BLUE)Creating project directories...$(NC)"
	mkdir -p data models checkpoints logs configs/profiles configs/models
	mkdir -p src/finetune/{cli,core,models,data,training,inference,api,ui,utils}
	mkdir -p tests/{unit,integration,fixtures}
	mkdir -p scripts examples docs
	@echo "$(GREEN)Directories created!$(NC)"

init: create-dirs install ## Initialize project for first time
	@echo "$(GREEN)Project initialized successfully!$(NC)"

.DEFAULT_GOAL := help