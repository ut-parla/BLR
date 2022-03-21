.PHONY: all docker

run:
	@echo "conda activate parla"
	@echo "cd app"
	@echo "python3 main.py"

conda-install:
	wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
	sh Miniconda3-py38_4.11.0-Linux-x86_64.sh
	rm Miniconda3-py38_4.11.0-Linux-x86_64.sh

conda-init:
	conda create --name parla --file conda_deps.txt
	git submodule update --init --recursive
	conda activate parla && pip install -e Parla.py/
	@echo To activate, run:   conda activate parla
