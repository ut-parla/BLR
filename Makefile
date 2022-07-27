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

create-inputs:
	mkdir inputs
	python app/main.py -mode gen -nblocks 4 -b 2500 -matrix inputs/matrix.npy -vector inputs/vector.npy

# run:
# 	CUDA_VISIBLE_DEVICES=0   python app/main.py run inputs/Arand10k.mat.npy inputs/xrand10k.mat.npy 1000
# 	CUDA_VISIBLE_DEVICES=0,1 python app/main.py run inputs/Arand10k.mat.npy inputs/xrand10k.mat.npy 1000
# 	python app/main.py run inputs/Arand10k.mat.npy inputs/xrand10k.mat.npy 1000

# 	CUDA_VISIBLE_DEVICES=0   python app/main.py run inputs/Arand20k.mat.npy inputs/xrand20k.mat.npy 1000
# 	CUDA_VISIBLE_DEVICES=0,1 python app/main.py run inputs/Arand20k.mat.npy inputs/xrand20k.mat.npy 1000
# 	python app/main.py run inputs/Arand20k.mat.npy inputs/xrand20k.mat.npy 1000

	
