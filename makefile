
# Default target
all: run


# First time setup
setup: venv-setup venv-activate install


# Venv
venv-setup:
ifeq ($(OS),Windows_NT)
	py -3.11 -m venv venv
else
	python3.11 -m venv venv
endif

venv-activate:
ifeq ($(OS),Windows_NT)
	@cmd /k "venv\Scripts\activate"
else
	@bash -c "source venv/bin/activate; exec bash"
endif

venv-deactivate:
	@echo "You must deactivate manually in the shell you're in."


# Install
install:
	@python -m pip install --upgrade pip
	@pip install -r requirements.txt -U


# Run
run:
	@python -m source.run

run-sbatch-sequential:
ifeq ($(OS),Windows_NT)
	@echo "Cannot run bash script 'sequential.sh' on a Windows environment."
else
	@sbatch ./scripts/sequential.sh
endif

run-sbatch-parallel:
	@python -m scripts.parallel


# Clean
clean:
	@rm -rf ./log ./log_tensorboard ./output/*


# Show sbatch user job queue
show:
	@squeue -u $$USER


# Aggregate results from parallel (sbatch) execution
aggregate:
	@python -m source.results
