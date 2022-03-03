.PHONY: all docker

all:
	sudo apt install -y python3.7 python3.7-dev python3.7-venv libunwind-dev
	python3.7 -m venv .parla
	. .parla/bin/activate && pip install wheel && pip install -r requirements.txt
	git submodule update --init --recursive
	. .parla/bin/activate && cd Parla.py && python3 setup.py install

	export LD_LIBRARY_PATH=/usr/local/cuda/lib64

run:
	@echo ". .parla/bin/activate"
	@echo "cd app"
	@echo "python3 main.py"