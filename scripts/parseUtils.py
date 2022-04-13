import json
from pprint import pprint
import os
import sys
import csv
import statistics

def print_stats(map):
    print("-"*50)
    print("Avg write : {:.2f} MB".format(statistics.mean(map['write'])/1024/1024))
    print("Avg read : {:.2f} MB".format(statistics.mean(map['read'])/1024/1024))
    print("Avg cpu used: {:.2f} %".format(100 - statistics.mean(map['idl'])))
    print("-"*50)
    print("Total write : {:.2f} GB".format(sum(map['write'])/1024/1024/1024))
    print("Total read : {:.2f} GB".format(sum(map['read'])/1024/1024/1024))    
    print("-"*50)

if len(sys.argv) < 2:
	print("Input csv file to parse")
	sys.exit()

infile = sys.argv[1]

map = {}

with open(infile, mode='r') as csv_file:
    for i in range(7):
        next(csv_file)
    reader = csv.reader(csv_file)
    for row in reader:
        map.setdefault('usr', []).append(float(row[0]))
        map.setdefault('sys', []).append(float(row[1]))
        map.setdefault('idl', []).append(float(row[2]))
        map.setdefault('wait', []).append(float(row[3]))        
        map.setdefault('read', []).append(float(row[6]))
        map.setdefault('write', []).append(float(row[7]))

print_stats(map)




