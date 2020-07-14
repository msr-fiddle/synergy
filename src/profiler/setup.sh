#!/bin/bash
if [ "$#" -ne 1 ]; then  
	echo "Input path to add to PYTHONPATH"
	exit 1
fi

path=$1
export PYTHONPATH="${PYTHONPATH}:$path"
