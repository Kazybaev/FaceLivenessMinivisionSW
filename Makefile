.PHONY: install dev test test-unit test-integration lint run docker clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,train]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check app/ tests/

run:
	python -m app.main

docker:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
