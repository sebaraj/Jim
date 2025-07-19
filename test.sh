#!/bin/bash
# test.sh

now=`date +"%Y%m%d%H%M%S"`
LOG="test-${now}.log"

if [ -v HPCSDK_RELEASE ]; then
    echo "Running with NVIDIA HPC SDK"

    if [ ! -v CUDA_HOME ] || [ ! -d ${CUDA_HOME} ]; then
        export CUDA_HOME=$(nvc++ -cuda -printcudaversion |& grep "CUDA Path" | awk -F '=' '{print $2}')
        echo "Setting CUDA_HOME=${CUDA_HOME}"
    fi 

    if [ ! -v NCCL_HOME ] || [ ! -d ${NCCL_HOME} ]; then
        export NCCL_HOME=$(dirname `echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nccl | grep -v sharp`)
        echo "Setting NCCL_HOME=${NCCL_HOME}"
    fi 

    if [ ! -v NVSHMEM_HOME ] || [ ! -d ${NVSHMEM_HOME} ]; then
        export NVSHMEM_HOME=$(dirname `echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvshmem`)
        echo "Setting NVSHMEM_HOME=${NVSHMEM_HOME}"
    fi
fi

if [ -e ${LOG} ]; then
  echo "ERROR log file ${LOG} already exists."
  exit 1
fi

CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )

errors=0

for entry in `ls -1`; do
    if [ -f ${entry}/Makefile ] ; then
        if [ "run" == "$1" ] ; then
            NUM_GPUS=`nvidia-smi -L | wc -l`
            for (( NP=1; NP<=${NUM_GPUS}; NP++ )) ; do
                export NP
                export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NP}]}
                CMD="make -C ${entry} $1"
                ${CMD} >> ${LOG} 2>&1
                if [ $? -ne 0 ]; then
                    echo "ERROR with ${CMD} (NP = ${NP}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) see ${LOG} for details."
                    errors=1
                    break
                fi
            done
        else
            CMD="make -C ${entry} $1"
            ${CMD} >> ${LOG} 2>&1
            if [ $? -ne 0 ]; then
                echo "ERROR with ${CMD} see ${LOG} for details."
                errors=1
                break
            fi
        fi
    fi
done

if [ ${errors} -eq 0 ]; then
    echo "Passed."
    exit 0
else
    exit 1
fi
