#!/bin/bash

if [ "$#" -ne 1 ]; then  
	echo "Enter log file path"
	exit 1
fi

FILE=$1
log_tmp='tmp.csv'
csv="${FILE%.*}"_parsed.csv
sed 1d $FILE >>  $log_tmp

echo "0,1,2,3,4,5,6,7" >  $csv



