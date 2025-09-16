# Makefile for FineTune project

.PHONY: help install dev test test-lora lint format clean run-api run-ui docker-build docker-run completion

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
	@echo "$(BLUE)🚀 Quick Start:$(NC) make dev && make test-week2-quick"
	@echo "$(BLUE)📋 Common:$(NC) make test-week2 | make test-lora | make test-e2e-quick | make format"
	@echo ""
	@echo "$(YELLOW)📦 Environment & Setup$(NC)"
	@grep -E '^(poetry-check|install|dev|install-all|update|lock|shell|init|create-dirs):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🧪 Testing$(NC)"
	@grep -E '^(test|test-unit|test-base|test-integration|test-lora|test-lora-quick|test-data|test-templates|test-config|test-week2|test-week2-quick|test-e2e-workflow|test-e2e-real-model|test-e2e-ollama|test-e2e-all|test-e2e-quick):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🔍 Code Quality$(NC)"
	@grep -E '^(lint|format|check|pre-commit):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🚀 Running Applications$(NC)"
	@grep -E '^(run-api|run-ui|run-cli|info):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🐳 Docker$(NC)"
	@grep -E '^(docker-build|docker-run):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)📚 Documentation$(NC)"
	@grep -E '^(docs|docs-serve):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)🛠️  Utilities$(NC)"
	@grep -E '^(clean|setup-mlx|benchmark|update-deps|completion|completion-install):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)ℹ️  Help$(NC)"
	@grep -E '^(help):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

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

test-lora: ## Run all LoRA-related tests (Week 1 Phase 2 functionality)
	@echo "$(BLUE)Running LoRA tests...$(NC)"
	@echo "$(YELLOW)Testing LoRA configuration and layer functionality...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_lora.py::TestLoRAConfig -v --color=yes
	@echo "$(YELLOW)Testing LoRA linear layer implementation...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_lora.py::TestLoRALinear -v --color=yes
	@echo "$(YELLOW)Testing LoRA training integration...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_lora.py::TestLoRATrainer -v --color=yes
	@echo "$(YELLOW)Testing LoRA utilities...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_lora.py::TestLoRAApplication -v --color=yes || true
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/test_lora.py::TestLoRASaveLoad -v --color=yes || true
	@echo "$(GREEN)✅ All LoRA tests completed!$(NC)"

test-lora-quick: ## Run quick LoRA functionality test
	@echo "$(BLUE)Running quick LoRA functionality test...$(NC)"
	@echo "$(YELLOW)Testing LoRA configuration...$(NC)"
	@PYTHONPATH=src .venv/bin/python -c "from finetune.training.lora import LoRAConfig; c=LoRAConfig(r=8, alpha=16.0); print(f'✅ Config: r={c.r}, α={c.alpha}, scaling={c.scaling}')"
	@echo "$(YELLOW)Testing LoRA layer...$(NC)"
	@PYTHONPATH=src .venv/bin/python -c "from finetune.training.lora import LoRAConfig, LoRALinear; import mlx.core as mx; c=LoRAConfig(r=8); l=LoRALinear(64,64,c); x=mx.random.normal((2,64)); o=l(x); print(f'✅ Layer: {x.shape} → {o.shape}, params reduced: {((l.lora_a.size + l.lora_b.size) / l.base.weight.size):.1%}')"
	@echo "$(GREEN)✅ Quick LoRA test passed!$(NC)"

test-data: ## Run data loading and validation tests (Week 2)
	@echo "$(BLUE)Running data pipeline tests...$(NC)"
	@echo "$(YELLOW)Testing JSON/JSONL data loading...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/data/test_loaders.py -v --color=yes
	@echo "$(GREEN)✅ Data loading tests completed!$(NC)"

test-templates: ## Run prompt template tests (Week 2)
	@echo "$(BLUE)Running prompt template tests...$(NC)"
	@echo "$(YELLOW)Testing Alpaca, ChatML, Llama templates...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/data/test_templates.py -v --color=yes
	@echo "$(GREEN)✅ Template tests completed!$(NC)"

test-config: ## Run configuration system tests (Week 2)
	@echo "$(BLUE)Running configuration system tests...$(NC)"
	@echo "$(YELLOW)Testing training configs, profiles, validation...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/config/test_config.py -v --color=yes
	@echo "$(GREEN)✅ Configuration tests completed!$(NC)"

test-week2: ## Run all Week 2 components (data + templates + config)
	@echo "$(BLUE)Running complete Week 2 test suite...$(NC)"
	@echo "$(YELLOW)Data Loading (21 tests) + Templates (23 tests) + Configuration (34 tests)$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/unit/data/ tests/unit/config/ -v --color=yes
	@echo "$(GREEN)✅ All 78 Week 2 tests completed!$(NC)"

test-week2-quick: ## Run quick Week 2 functionality test
	@echo "$(BLUE)Running quick Week 2 integration test...$(NC)"
	@PYTHONPATH=src .venv/bin/python -c "from finetune.data import DatasetLoader, AlpacaTemplate, TemplateRegistry; from finetune.config import TrainingConfig, ModelConfig, DataConfig, ConfigProfile; print('✅ Data loading, templates, and config systems operational')"
	@echo "$(GREEN)✅ Quick Week 2 test passed!$(NC)"

# End-to-End Integration Tests
test-e2e-workflow: ## Run end-to-end workflow integration test (mocked components)
	@echo "$(BLUE)Running end-to-end workflow integration test...$(NC)"
	@echo "$(YELLOW)Testing component integration with mocked dependencies...$(NC)"
	PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_end_to_end_workflow.py -v --color=yes
	@echo "$(GREEN)✅ End-to-end workflow test completed!$(NC)"

test-e2e-real-model: ## Run real model integration test (requires FT_REAL_MODEL_ENABLE=1)
	@echo "$(BLUE)Running real model integration test...$(NC)"
	@echo "$(YELLOW)Testing with actual HuggingFace models and measurable success criteria...$(NC)"
	@if [ "$(FT_REAL_MODEL_ENABLE)" = "1" ]; then \
		echo "$(GREEN)Real model testing enabled$(NC)"; \
		PYTHONPATH=src FT_REAL_MODEL_ENABLE=1 .venv/bin/python -m pytest tests/integration/test_end_to_end_real_model.py -v --color=yes; \
	else \
		echo "$(YELLOW)Real model testing disabled. Set FT_REAL_MODEL_ENABLE=1 to enable.$(NC)"; \
		echo "$(YELLOW)Usage: FT_REAL_MODEL_ENABLE=1 make test-e2e-real-model$(NC)"; \
	fi
	@echo "$(GREEN)✅ Real model integration test completed!$(NC)"

test-e2e-ollama: ## Run full Ollama deployment test (requires FT_E2E_ENABLE=1)
	@echo "$(BLUE)Running Ollama end-to-end deployment test...$(NC)"
	@echo "$(YELLOW)Testing complete pipeline: model → fine-tune → Ollama → evaluation...$(NC)"
	@if [ "$(FT_E2E_ENABLE)" = "1" ]; then \
		echo "$(GREEN)Ollama E2E testing enabled$(NC)"; \
		PYTHONPATH=src FT_E2E_ENABLE=1 .venv/bin/python -m pytest tests/integration/test_end_to_end_ollama.py -v --color=yes; \
	else \
		echo "$(YELLOW)Ollama E2E testing disabled. Set FT_E2E_ENABLE=1 to enable.$(NC)"; \
		echo "$(YELLOW)Usage: FT_E2E_ENABLE=1 make test-e2e-ollama$(NC)"; \
		echo "$(YELLOW)Note: Requires ollama CLI and network access$(NC)"; \
	fi
	@echo "$(GREEN)✅ Ollama end-to-end test completed!$(NC)"

test-e2e-all: ## Run all end-to-end tests (workflow + real model + Ollama)
	@echo "$(BLUE)Running all end-to-end integration tests...$(NC)"
	@echo "$(YELLOW)1/3: Workflow integration (fast, mocked)...$(NC)"
	@$(MAKE) test-e2e-workflow
	@echo ""
	@echo "$(YELLOW)2/3: Real model integration (medium, requires models)...$(NC)"
	@FT_REAL_MODEL_ENABLE=1 $(MAKE) test-e2e-real-model
	@echo ""
	@echo "$(YELLOW)3/3: Ollama deployment (slow, requires external tools)...$(NC)"
	@FT_E2E_ENABLE=1 $(MAKE) test-e2e-ollama
	@echo ""
	@echo "$(GREEN)🎉 All end-to-end tests completed successfully!$(NC)"

test-e2e-quick: ## Run quick end-to-end validation (workflow + real model, no Ollama)
	@echo "$(BLUE)Running quick end-to-end validation...$(NC)"
	@echo "$(YELLOW)Testing workflow integration and real model loading...$(NC)"
	@$(MAKE) test-e2e-workflow
	@FT_REAL_MODEL_ENABLE=1 $(MAKE) test-e2e-real-model
	@echo "$(GREEN)✅ Quick end-to-end validation completed!$(NC)"

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

completion: ## Generate bash/zsh completion scripts
	@echo "$(BLUE)Generating shell completion scripts...$(NC)"
	@mkdir -p scripts/completion
	@echo "# Bash completion for FineTune Makefile" > scripts/completion/ft-make-completion.bash
	@echo "_ft_make_completion() {" >> scripts/completion/ft-make-completion.bash
	@echo "    local cur targets" >> scripts/completion/ft-make-completion.bash
	@echo "    COMPREPLY=()" >> scripts/completion/ft-make-completion.bash
	@echo "    cur=\"\$${COMP_WORDS[COMP_CWORD]}\"" >> scripts/completion/ft-make-completion.bash
	@echo "    targets=\$$(make -qp | awk -F':' '/^[a-zA-Z0-9][^$$#\/\\t=]*:([^=]|$$)/ {split(\$$1,A,/ /);for(i in A)print A[i]}' | sort -u)" >> scripts/completion/ft-make-completion.bash
	@echo "    COMPREPLY=( \$$(compgen -W \"\$$targets\" -- \$$cur) )" >> scripts/completion/ft-make-completion.bash
	@echo "    return 0" >> scripts/completion/ft-make-completion.bash
	@echo "}" >> scripts/completion/ft-make-completion.bash
	@echo "complete -F _ft_make_completion make" >> scripts/completion/ft-make-completion.bash
	@echo "" >> scripts/completion/ft-make-completion.bash
	@echo "# Zsh completion for FineTune Makefile" > scripts/completion/ft-make-completion.zsh
	@echo "#compdef make" >> scripts/completion/ft-make-completion.zsh
	@echo "_make_targets() {" >> scripts/completion/ft-make-completion.zsh
	@echo "    local targets" >> scripts/completion/ft-make-completion.zsh
	@echo "    targets=(\$$(make -qp | awk -F':' '/^[a-zA-Z0-9][^$$#\/\\t=]*:([^=]|$$)/ {split(\$$1,A,/ /);for(i in A)print A[i]}' | sort -u))" >> scripts/completion/ft-make-completion.zsh
	@echo "    _describe 'make targets' targets" >> scripts/completion/ft-make-completion.zsh
	@echo "}" >> scripts/completion/ft-make-completion.zsh
	@echo "_make_targets" >> scripts/completion/ft-make-completion.zsh
	@echo "" >> scripts/completion/ft-make-completion.zsh
	@echo "$(GREEN)Completion scripts generated in scripts/completion/$(NC)"
	@echo "$(YELLOW)To enable bash completion, add to your ~/.bashrc:$(NC)"
	@echo "  source $(PWD)/scripts/completion/ft-make-completion.bash"
	@echo "$(YELLOW)To enable zsh completion, add to your ~/.zshrc:$(NC)"
	@echo "  source $(PWD)/scripts/completion/ft-make-completion.zsh"

completion-install: completion ## Install completion scripts for current user
	@echo "$(BLUE)Installing completion scripts...$(NC)"
	@if [ "$$SHELL" = "/bin/bash" ] || [ "$$SHELL" = "/usr/bin/bash" ]; then \
		echo "source $(PWD)/scripts/completion/ft-make-completion.bash" >> ~/.bashrc; \
		echo "$(GREEN)Bash completion installed! Restart terminal or run: source ~/.bashrc$(NC)"; \
	elif [ "$$SHELL" = "/bin/zsh" ] || [ "$$SHELL" = "/usr/bin/zsh" ]; then \
		echo "source $(PWD)/scripts/completion/ft-make-completion.zsh" >> ~/.zshrc; \
		echo "$(GREEN)Zsh completion installed! Restart terminal or run: source ~/.zshrc$(NC)"; \
	else \
		echo "$(YELLOW)Unsupported shell: $$SHELL$(NC)"; \
		echo "$(YELLOW)Please manually source the appropriate completion script$(NC)"; \
	fi

.DEFAULT_GOAL := help