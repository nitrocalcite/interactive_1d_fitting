PACKAGE = ivfit

all: install

install:
	poetry install

update-env:
	poetry install

update-lockfile:
	poetry update
	poetry export -f requirements.txt --output requirements.lock

.PHONY: install update-env update-lockfile
