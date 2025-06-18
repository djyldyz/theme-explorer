#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = theme-explorer-gds
PYTHON_VERSION = 3.12

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


.PHONY: precommit
precommit:
	uv run --frozen pre-commit run --all-files

## Refresh Python Dependencies
.PHONY: env_refresh
env_refresh:
	uv sync

## Install Python dependencies without updating
.PHONY: env
env:
	uv sync --frozen
