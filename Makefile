fmt:
	poetry run black .
	poetry run isort .

test:
	poetry run pytest

cov:
	open htmlcov/index.html

.PHONY: docs
docs:
	cd docs && poetry run sphinx-build -M html . build -a
	open docs/build/html/index.html
