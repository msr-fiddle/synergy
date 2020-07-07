import json
from pprint import pprint
import os
import sys
import csv
import statistics

def print_stats(map):
    print("-"*50)
    print("Avg data time : {:.2f} s".format(statistics.mean(map['dtime'])))
    print("Avg memcpy time : {:.2f} s".format(statistics.mean(map['mtime'])))
    print("Avg compute time : {:.2f} s".format(statistics.mean(map['ctime'])))
    print("Avg chk time : {:.2f} s".format(statistics.mean(map['chktime'])))
    print("Avg iter time : {:.2f} s".format(statistics.mean(map['tottime'])))
    print("-"*50)
    print("Total data time : {:.2f} s".format(sum(map['dtime'])))
    print("Total memcpy time : {:.2f} s".format(sum(map['mtime'])))
    print("Total compute time : {:.2f} s".format(sum(map['ctime'])))
    print("Total chk time : {:.2f} s".format(sum(map['chktime'])))
    print("Total iter time : {:.2f} s".format(sum(map['tottime'])))  
    print("-"*50)

if len(sys.argv) < 2:
	print("Input csv file to parse")
	sys.exit()

infile = sys.argv[1]

map = {}

with open(infile, mode='r') as csv_file:
    for i in range(3):
        next(csv_file)
    reader = csv.reader(csv_file)
    for row in reader:
        map.setdefault('iter', []).append(float(row[1]))
        map.setdefault('dtime', []).append(float(row[2]))
        map.setdefault('mtime', []).append(float(row[3]))
        map.setdefault('ftime', []).append(float(row[4]))        
        map.setdefault('ctime', []).append(float(row[5]))
        map.setdefault('ttime', []).append(float(row[6]))
        map.setdefault('chktime', []).append(float(row[7]))
        map.setdefault('tottime', []).append(float(row[8]))
print_stats(map)




