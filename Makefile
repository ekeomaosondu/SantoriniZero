PYTHON ?= python3
CXX ?= c++
CKPT ?= checkpoints/latest.pkl
PORT ?= 8000

.PHONY: install build test-cpp test-bindings test smoke train play clean

install:            ## install Python deps and build the C++ module
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

build:              ## (re)build the C++ module in place
	$(PYTHON) setup.py build_ext --inplace

test-cpp:           ## C++ correctness suite (~1-3 min)
	$(CXX) -O2 -march=native -std=c++17 -pthread -o cpp/test cpp/test.cpp
	./cpp/test

test-bindings:      ## quick checks of the Python bindings
	$(PYTHON) tests/check_bindings.py

test: test-cpp test-bindings

smoke:              ## tiny end-to-end training run (~1 min)
	$(PYTHON) train.py --smoke

train:              ## full training run in runs/default
	$(PYTHON) train.py

play:               ## play against the agent at http://127.0.0.1:$(PORT)
	$(PYTHON) play/santorini_agent.py --ckpt $(CKPT) --port $(PORT)

clean:
	rm -rf build *.egg-info santorini*.so cpp/test
