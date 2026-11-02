IMAGE := churn-api
NOTEBOOK := notebooks/tech_challenge_1_FIAP.ipynb

.PHONY: help setup train test lint format check notebook run mlflow frontend docker-build docker-run

help:
	@echo "setup         instala o Python 3.13 e as dependencias (uv.lock)"
	@echo "train         fluxo completo: baseline, comparacao, campeao e avaliacao no teste"
	@echo "test          testes (pytest)"
	@echo "lint          checagem de codigo (ruff)"
	@echo "format        formata o codigo (ruff)"
	@echo "check         lint + formatacao + testes"
	@echo "notebook      reexecuta o notebook salvando as saidas"
	@echo "run           sobe a API em http://127.0.0.1:8000/docs"
	@echo "mlflow        abre o MLflow em http://127.0.0.1:5000"
	@echo "frontend      sobe o dashboard em http://localhost:5173"
	@echo "docker-build  gera a imagem da API (exige models/champion.joblib)"
	@echo "docker-run    roda a imagem na porta 8000"

setup:
	uv python install 3.13
	uv sync

train:
	uv run python -m churn.main

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

notebook:
	uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 $(NOTEBOOK)

run:
	uv run uvicorn churn.api:app --reload

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

frontend:
	npm --prefix frontend install
	npm --prefix frontend run dev

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run -d --name $(IMAGE) -p 8000:8000 $(IMAGE)
