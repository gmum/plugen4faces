#!/bin/sh

NAME=${1:-"plugen4faces"}

conda env create -f environment.yml -n $NAME
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $NAME
conda env config vars set CUDA_HOME=$CONDA_PREFIX -n $NAME
conda env config vars set CPATH=$CONDA_PREFIX/include:$CPATH -n $NAME
conda deactivate && conda activate $NAME
LDFLAGS="-I$CONDA_PREFIX/lib -I$CONDA_PREFIX/lib64" pip --no-input install dlib==19.24.0 --verbose
python -c "from plugen4faces.flow import SimpleRealNVP"
