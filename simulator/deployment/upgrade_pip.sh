#!/bin/sh

pip install --upgrade protobuf

alias python=python3
pip install --upgrade pip
python -m pip install --upgrade setuptools
pip install --no-cache-dir  grpcio 
pip install --no-cache-dir  grpcio-tools 
pip install --no-cache-dir  python-numa
apt install python3.6-dev
python -m pip install psutil 
