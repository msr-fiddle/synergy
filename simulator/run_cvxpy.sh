#!/bin/bash

if [ "$#" -ne 6 ]; then
    echo "./run <ilp1> <ilp2> <phase2> <num-servers> <multi> <out-dir>"
    exit 1
fi

ILP1=$1
ILP2=$2
PHASE2=$3
NUM_SERVER=$4
MULTI=$5
OUT_DIR=$6

#lp_solver=('GUROBI' 'GLPK_MI')
lp_solver=('GUROBI')
#lp_solver=('ECOS' 'ECOS_BB' 'GLPK' 'GLPK_MI' 'GUROBI' 'OSQP' 'SCS')
#ilp_solver=('GUROBI')
ilp_solver=('GUROBI')
#ilp_solver=('GLPK_MI')

#seed_list=(3)
#seed_list=(6 71 8 91 100)
seed_list=(1 21 3 41 5)
EXTRA=""

if [ "$ILP1" -eq "1" ]; then
    EXTRA="${EXTRA} --ilp1"
    if [ "$PHASE2" -eq "1" ]; then
    	solver_list=("${lp_solver[@]}")
    else
    	solver_list=("${ilp_solver[@]}")
    fi
else
    solver_list=("${lp_solver[@]}")
fi

if [ "$PHASE2" -eq "0" ]; then
    EXTRA="${EXTRA} --phase1"
fi

if [ "$MULTI" -eq "1" ]; then
    EXTRA="${EXTRA} --multi"
fi

if [ "$ILP2" -eq "1" ]; then
    EXTRA="${EXTRA} --ilp2"
fi

for solver in ${solver_list[*]}; do
    for seed in ${seed_list[*]}; do
      out_dir="${OUT_DIR}/${solver}/${NUM_SERVER}"
      mkdir -p $out_dir
      out_file="${out_dir}/log_${seed}.log"
      echo $solver $out_file
      #lp_solver="GLPK_MI"
      if [ "$PHASE2" -eq "0" ]; then
          python cvxpy_test.py --debug --seed $seed --num_servers $NUM_SERVER --solver $solver $EXTRA 2>&1 | tee $out_file
      elif [ "$ILP1" -eq "0" ]; then
          python cvxpy_test.py --debug --seed $seed --num_servers $NUM_SERVER --solver $solver --solver2 $solver $EXTRA  2>&1 | tee $out_file 
      else
          python cvxpy_test.py --debug --seed $seed --num_servers $NUM_SERVER --solver $solver  --solver2 $solver $EXTRA 2>&1 | tee $out_file
      fi
      record_file="${out_dir}/record_${seed}.log"
      mv opt_record.log $record_file
    done
done
