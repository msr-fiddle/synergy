import csv
import argparse
import sys
import os

FILE = sys.argv[1]
if not os.path.exists(FILE):
    print("Incorrect file")
    sys.exit(1)

NUM_GPU = 8

base = os.path.basename(FILE)
dirpath = os.path.dirname(FILE)
out_file = dirpath + "/parsed_" +  base
mydict = {}
with open(FILE, mode='r') as infile:
    reader = csv.reader(infile)
    count = 0
    for row in reader:
        count += 1
        mydict[count] = row[1].replace("%", "")
    print(mydict[count])
        #writer = csv.writer(outfile)
        #mydict = {rows[0]:rows[1] for rows in reader}

    print(len(mydict.keys()))
    i = 0
    gpu_map = {}
    skipped =  False
    for k,v in mydict.items():
        if i == 0 and skipped is False:
            skipped = True
        else:
            gpu = i % NUM_GPU
            if gpu not in gpu_map:
                gpu_map[gpu] = []
            gpu_map[gpu].append(v)
            i += 1

    res = []
    for gpus in gpu_map.keys():
        res.append(gpu_map[gpus])

    print(len(res))
    for l in res:
        print(len(l))

    zipped = zip(res)
    with open(out_file, mode='w') as outfile:
        wr = csv.writer(outfile, dialect='excel')
        for row in zip(*res):
            #print(len(row))
            wr.writerow(row)
