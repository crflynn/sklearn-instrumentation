fmt:
	poetry run black .
	poetry run isort .

test:
	poetry run pytest

cov:
	open htmlcov/index.html

.PHONY: docs
docs:
	poetry export --dev -f requirements.txt > docs/requirements.txt
	cd docs && poetry run sphinx-build -M html . build -a
	open docs/build/html/index.html

build:
	poetry build

clean:
	rm -rf dist

publish: clean build
	poetry publish

release: clean build
	ghr -u crflynn -r sklearn-instrumentation -c $(shell git rev-parse HEAD) -delete -b "release" -n $(shell poetry version | tail -c +25) $(shell poetry version | tail -c +25) dist/