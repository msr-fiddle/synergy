import json
from pprint import pprint
import os
import sys
import csv
import re
#import statistics


def atoi(text):
	return int(text) if text.isdigit() else text

def str_keys(text):
	return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def sort_runs(dname_list):
	dname_list.sort(key=str_keys)

if len(sys.argv) < 2:
	print("Input dir with run<n> subdir each with acc files")
	sys.exit()

indir = sys.argv[1]

if not os.path.exists(indir):
	print(" Input dir does not exist")
	sys.exit(1)

dir_name_list = [dname for dname in os.listdir(indir) if os.path.isdir(indir + '/' + dname)]
sort_runs(dir_name_list)
print(dir_name_list)
dir_list = [indir + '/' + dname for dname in dir_name_list]
#dir_list.sort()
acc_list = [dname + '/acc-0.csv' for dname in dir_list]
time_list = [dname + '/stdout.out' for dname in dir_list]
#print(acc_list)

acc_map={}
epoch = 1

partial_epoch_time=  0
cumulative_time = 0 
for idx, acc_file in enumerate(acc_list):
	#Full file path exists
	start_epoch = epoch
	acc_this_run = []
	time_this_run = []
	try:
		with open(acc_file, 'r') as af:
			reader = csv.reader(af, delimiter=",")
			for i, line in enumerate(reader):
				acc = float(line[-1])
				if epoch not in acc_map:
					acc_map.setdefault(epoch, [])
				acc_map[epoch].append(idx)
				acc_map[epoch].append(acc)
				epoch += 1
				acc_this_run.append(acc)
	except:
		continue

	if (len(acc_this_run) == 0):
		continue

	with open(time_list[idx], 'r') as tf:
		for line in tf:
			if 'Time_stat' in line:
				time_this_run.append(float(line.split()[-1]))

	#print("Run {}".format(idx))
	#print(acc_this_run, time_this_run)
	#Remove the setup time
	time_this_run.pop(0)
	#Add any leftover time from previous run here
	time_this_run[0] += partial_epoch_time

	if len(acc_this_run) == len(time_this_run):
		partial_epoch_time = 0
	elif len(acc_this_run) + 1 == len(time_this_run):
		#There's an incomplete epoch
		partial_epoch_time = time_this_run[-1]
		time_this_run.pop(-1)
	else:
		print("Incorrect parsed info. Exiting")
		sys.exit(-2)

	if len(acc_this_run) != len(time_this_run):
		print("Mismatch in timimg and accuracy info. Exiting")
		sys.exit(-2)

	for i, acc in enumerate(acc_this_run):
		ep =  start_epoch + i
		time = time_this_run[i]
		cumulative_time += time
		acc_map[ep].append(time)
		acc_map[ep].append(cumulative_time)

	#if epoch == 2:
	#	print(acc_this_run)
	#	print(time_this_run)
		

print(len(acc_map.keys()))
out_fname = indir + '/acc.csv'
with open(out_fname, 'w+') as of:
	of.write("Round, Epoch completed, Top-1 Acc(%), Time for epoch(s), Cumulative Time(s)\n")
	for ep, val in acc_map.items():
		if len(val) != 4:
			break
		print(val[0], ep, val[1], val[2], val[3])
		of.write(str(val[0]) + ',' + str(ep) + ',' + str(val[1]) + ',' +  str(val[2]) + ',' +  str(val[3]) + '\n')
		


