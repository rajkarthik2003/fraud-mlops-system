# filepath: Makefile
.PHONY: help install train export test docker-build docker-up docker-down ci

help:
	@echo "Fraud Detection MLOps - Available Commands"
	@echo "==========================================="
	@echo "make install        - Install dependencies"
	@echo "make train          - Train XGBoost model"
	@echo "make export         - Export model artifacts"
	@echo "make test           - Run unit tests"
	@echo "make docker-build   - Build Docker images"
	@echo "make docker-up      - Start all services"
	@echo "make docker-down    - Stop all services"
	@echo "make ci             - Run full CI pipeline"
	@echo "make lint           - Run code linting"

install:
	pip install -r requirements.txt

train:
	@echo "Training XGBoost model..."
	python training/train_xgboost.py

export:
	@echo "Exporting model artifacts..."
	python training/export_best_model.py

test:
	pytest tests/ -v --cov=app --cov-report=term-missing

lint:
	flake8 app --count --select=E9,F63,F7,F82 --show-source --statistics
	black --check app

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services starting..."
	@echo "  API:       http://localhost:8000"
	@echo "  Swagger:   http://localhost:8000/docs"
	@echo "  UI:        http://localhost:8501"
	@echo "  MLflow:    http://localhost:5001"
	@echo "  Prometheus: http://localhost:9090"

docker-down:
	docker-compose down -v

docker-logs:
	docker-compose logs -f

ci: lint test
	@echo "CI pipeline complete"

# Kubernetes commands (requires kubectl)
k8s-apply:
	kubectl apply -f k8s/

k8s-delete:
	kubectl delete namespace fraud-detection

k8s-status:
	kubectl get all -n fraud-detection