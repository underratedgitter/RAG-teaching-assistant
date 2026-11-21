.PHONY: start check help

help:
	@echo "make start   - set up .venv, check ffmpeg and Ollama, open the dashboard"
	@echo "make check   - check ffmpeg, Ollama and the models only (needs a prior make start)"

start:
	./scripts/start.sh

check:
	.venv/bin/python preflight.py
