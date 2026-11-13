#!/bin/bash
# source /tmp/$(id -un)-tmp/activate.sh
set -e
BASE=/tmp/$(id -un)-tmp

unset PYTHONPATH LD_LIBRARY_PATH BASH_ENV CONDARC
export HOME=$BASE/home TMPDIR=$BASE/tmp CONDA_PKGS_DIRS=$BASE/pkgs PIP_CACHE_DIR=$BASE/pip XDG_CACHE_HOME=$BASE/cache
mkdir -p $BASE $HOME $TMPDIR && chmod 700 $BASE

curl -fsSL -o $BASE/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash $BASE/miniforge.sh -b -u -p $BASE/miniforge3
$BASE/miniforge3/bin/conda env create -y -f environment.yml -p $BASE/env

cat > $BASE/activate.sh <<EOF
unset PYTHONPATH LD_LIBRARY_PATH BASH_ENV CONDARC
export HOME=$HOME TMPDIR=$TMPDIR CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS PIP_CACHE_DIR=$PIP_CACHE_DIR XDG_CACHE_HOME=$XDG_CACHE_HOME
source $BASE/miniforge3/bin/activate $BASE/env
EOF
echo "done: source $BASE/activate.sh"
