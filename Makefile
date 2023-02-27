.PHONY:	style test

# Run code quality checks
style_check:
	black --check --preview .
	isort --check .

style:
	black --preview .
	isort .

# Run tests for the library
test:
	python -m pip install .[tests]
	python -m pytest tests

fast_test:
	python -m pip install .[tests]
	python -m pytest tests/test_transfor_to_npu.py

# Utilities to release to PyPi
build_dist_install_tools:
	pip install build
	pip install twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*
