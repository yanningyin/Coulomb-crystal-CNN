#!/bin/bash

# ----------------Format to generate series of images-----------------#
# ./runMultiSim.sh $1 $2 $3 $4 $5 $6, where
# $1: num_ions_start; $2: num_ions_end; $3: num_ions_increment
# $4: temperature_start; $5: temperature_end; $6: temperature_increment
# --------------------------END---------------------------------------#

# ---Arguments for running PaulTrapSim.exe---#
#  PROGRAM_NAME      numIons    targetT
# ./PaulTrapSim.exe   150        10
# -----------------END-----------------------#

rm -rf *.hist
rm -rf *.info
# Parse the input arguments
NUMIONS_START=$1
NUMIONS_END=$2
NUMIONS_INCREMENT=$3

T_START=$4
T_END=$5
T_INCREMENT=$6

#load matlab module
module load matlab/matlab-R2021b
SIM_DIR="numIons_$1_$2_$3_targetT_$4_$5_$6/"
mkdir ${SIM_DIR}
cp PaulTrapSim.exe hist2img.m ${SIM_DIR}
cd ${SIM_DIR}

# Loop over ion numbers and heating factors
for ((i=$NUMIONS_START;i<=$NUMIONS_END;i+=$NUMIONS_INCREMENT));do
    for ((j=$T_START;j<=$T_END;j+=$T_INCREMENT));do
	      echo "-----------------------doing numIons = ${i}, targetT = ${j}----------------------------"
        ./PaulTrapSim.exe $i   $j
        matlab -nodisplay -nosplash -nodesktop -r "run('hist2img.m'); exit;"
        rm -rf *.hist
    done
done

rm -rf PaulTrapSim.exe hist2img.m
cd ..
