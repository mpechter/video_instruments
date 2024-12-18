.PHONY: default test
default: install

REPO_PATH=$(CURDIR)/src/
PROJECT_NAME = vip-glue-jobs
PYTHON=./.venv/bin/python

bare:
	@python3.10 -m venv ./.venv

install:
	@python3.10 -m venv ./.venv && \
	echo 'creating virtual env' && \
	source ./.venv/bin/activate && \
	echo 'installing required packages' && \
	python3.10 -m pip install --upgrade pip && \
	python3.10 -m pip install -r ./requirements.txt

freeze:
	@python3.10 -m venv ./.venv && \
	source ./.venv/bin/activate && \
	python3.10 -m pip freeze

single:
	@printf "\n" && \
	source ./.venv/bin/activate && \
	export PYTHONPATH="$(REPO_PATH)" && \
	$(PYTHON) src/single_keyboard.py

run_tests:
	@source ./.venv/bin/activate && \
	export PYTHONPATH="$(REPO_PATH)" && \
	python3 -m pytest test \
	--verbose

clean:
	@rm -rf ./.venv
