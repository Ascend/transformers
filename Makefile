.PHONY: test

# Run tests for the library
test_accelerate:
    python -m pytest -s -v ./tests/accelerate
