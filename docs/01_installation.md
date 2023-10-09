# Installation



<br>

## 1. Setup Conda

```
# Conda installation

# For Linux
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For OSX
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

chmod +x ~/miniconda.sh    
./miniconda.sh  

source ~/.bashrc          # For Linux
source ~/.bash_profile    # For OSX
```


<br>

## 2. Setup Python environment for CPU

```
# Clone GitHub repo
conda install git
git clone https://github.com/snap-research/LargeGT.git
cd LargeGT

# Install python environment
conda create -n gt python=3.10
conda activate gt
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install wandb absl-py tensorboard einops matplotlib progressbar
pip install kmeans-pytorch torchviz fastcluster opentsne networkx pandas ogb kmedoids numba scikit-network
pip install torch_geometric
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html

conda clean --all
```





<br><br><br>
