.PHONY: preflight smoke test core eval figures bundle clean

preflight:
	uv run --extra dev python -c "import jax; import jax_evogym; import sensing_edges; print('OK')"
	uv run --extra dev python -m pytest tests/test_configs.py -x -v --tb=short

smoke:
	uv run --extra dev python -m pytest tests/ -x -v --tb=short

test:
	uv run --extra dev python -m pytest tests/ -v --tb=short

core:
	@echo "Run core ablation matrix via paper_protocol.py"
	uv run python scripts/paper_protocol.py core

eval:
	@echo "Run held-out evaluation via paper_protocol.py"
	uv run python scripts/paper_protocol.py eval

figures:
	uv run python scripts/paper_protocol.py figures

bundle:
	uv run python scripts/export_release_bundle.py

clean:
	rm -rf outputs/*
