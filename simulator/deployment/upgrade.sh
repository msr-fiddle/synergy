#!/bin/sh

pip install --upgrade protobuf

alias python=python3
pip3 install --upgrade pip
python -m pip install --upgrade setuptools
pip3 install --no-cache-dir  grpcio 
pip3 install --no-cache-dir  grpcio-tools 
pip3 install --no-cache-dir  python-numa
pip3 install --no-cache-dir  cvxpy
apt install python3.6-dev
python3 -m pip install psutil 
