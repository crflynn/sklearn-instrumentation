fmt:
	poetry run black .
	poetry run isort .

test:
	poetry run pytest

cov:
	open htmlcov/index.html

.PHONY: docs
docs:
	poetry export --dev --extras all --without-hashes -f requirements.txt > docs/requirements.txt
	cd docs && poetry run sphinx-build -M html . build -a
	open docs/build/html/index.html

build:
	poetry build

clean:
	rm -rf dist

publish: clean build
	poetry publish

release: clean build
	ghr -u crflynn -r sklearn-instrumentation -c $(shell git rev-parse HEAD) -delete -b "release" -n $(shell poetry version -s) $(shell poetry version -s) dist/

statsd:
	docker run --rm -it -p 8080:8080 -p 8125:8125/udp -p 8125:8125/tcp rapidloop/statsd-vis -statsdudp 0.0.0.0:8125 -statsdtcp 0.0.0.0:8125
