PY ?= python3

build:
	$(PY) setup.py build_ext --inplace

# full correctness pass: engine build + suite + bit-exactness gate
test: build
	$(PY) tests/test_ratec.py
	$(PY) tests/test_v04.py
	$(PY) tests/test_profiles.py
	$(PY) tests/gate_bitexact.py

# pytest over the standalone scripts (conftest.py builds the engine first)
pytest: build
	$(PY) -m pytest tests/ -q

# benches (tiktoken bench needs `pip install tiktoken`; C-sweep is tiktoken-free)
bench: build
	$(PY) bench/bench_c_sweep.py
	$(PY) bench/bench_vs_tiktoken.py

clean:
	rm -rf build ratec.egg-info dist __pycache__ ratec/__pycache__ tests/__pycache__

.PHONY: build test pytest bench clean
