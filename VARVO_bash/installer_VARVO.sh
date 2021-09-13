#!/bin/bash
echo 'Downloading Miniconda...'
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O ~/miniconda.sh
echo 'Downloading Environment...'
wget https://anaconda.org/sbrgmoreno/VAVO_LINUX/2021.08.15.234939/download/varvo_linux_env.yml
echo 'Installing Miniconda...'
bash miniconda.sh -b -p $HOME/miniconda3
exec bash

