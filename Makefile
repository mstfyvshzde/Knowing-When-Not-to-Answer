# Provides simple commands to format, lint, test, and check the project, making development tasks faster and more consistent.


# make format -> Automatically formats the project code using Black.
# Use it after writing or changing Python code.
format:
	python -m black src tests



# make lint -> Checks the code for style problems, mistakes, and suspicious code using Ruff.
# Use it after formatting and before committing code.
lint:
	python -m ruff check src tests



# make test -> Runs all project tests using pytest.
# Use it after changing code to check that everything still works.
test:
	python -m pytest -v



# make check -> Runs formatting check, linting, and all tests together.
# Use it as the final check before committing or pushing code to GitHub.
check:
	python -m black --check src tests
	python -m ruff check src tests
	python -m pytest -v
