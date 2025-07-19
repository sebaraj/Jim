#!/bin/bash
# bench.sh

NREP=5
NXNY="20480"

CPUID=0-55
FIRST_CORE=0
MAX_NUM_GPUS=8
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" )
MPI_CPU_BINDING_OPT=("--bind-to" "core" "--map-by" "core")

IFS=$'\n'
function find_best () {
    declare -a RESULTS
    for ((i=0; i<$NREP; i++)); do
        RESULTS+=($("$@"))
    done
    printf '%s\n' "${RESULTS[@]}" | sort -k8 -b -t',' | head -1
    unset RESULTS
}

#Single GPU
if true; then
    echo "type, nx, ny, iter_max, nccheck, runtime"
    export CUDA_VISIBLE_DEVICES="0"
    for (( nx=1024; nx <= 20*1024; nx+=1024 )); do
        find_best taskset -c ${CPUID} ./memcpy_single/jacobi -csv -nx $nx -ny $nx
    done
fi

if true; then
    echo "type, nx, ny, iter_max, nccheck, runtime"
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./memcpy_single/jacobi -csv -nx ${NXNY} -ny ${NXNY}
fi

echo "type, nx, ny, iter_max, nccheck, num_devices, p2p, runtime, runtime_serial"

# multi threaded copy without thread pinning
if true; then

    export OMP_PROC_BIND=FALSE
    unset OMP_PLACES
    
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best ./memcpy_multi/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

export OMP_PROC_BIND=TRUE

#multi threaded copy
if true; then

    NEXT_CORE=${FIRST_CORE}
    OMP_PLACES="{$((NEXT_CORE))}"
    NEXT_CORE=$((NEXT_CORE+1))
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        if (( NUM_GPUS > 1 )); then
            OMP_PLACES="${OMP_PLACES},{$((NEXT_CORE))}"
            NEXT_CORE=$((NEXT_CORE+1))
        fi
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        export OMP_PLACES
        find_best ./memcpy_multi/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#multi threaded copy overlap
if true; then

    NEXT_CORE=${FIRST_CORE}
    OMP_PLACES="{$((NEXT_CORE))}"
    NEXT_CORE=$((NEXT_CORE+1))
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        if (( NUM_GPUS > 1 )); then
            OMP_PLACES="${OMP_PLACES},{$((NEXT_CORE))}"
            NEXT_CORE=$((NEXT_CORE+1))
        fi
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        export OMP_PLACES
        find_best ./memcpy_multi_overlap/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#mpi 
if true; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES "${MPI_CPU_BINDING_OPT[@]}" ./mpi/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#mpi overlap
if true; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES "${MPI_CPU_BINDING_OPT[@]}" ./mpi_overlap/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#nccl
if true; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES "${MPI_CPU_BINDING_OPT[@]}" ./nccl/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#nccl overlap
if true; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES "${MPI_CPU_BINDING_OPT[@]}" ./nccl_overlap/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi


