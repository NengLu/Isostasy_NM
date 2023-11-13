#!/bin/bash
#
# This is a PBS job script. Run with "qsub run_pbs.sh"
#
#PBS -N Zhong2000NM
#PBS -q long
#PBS -l nodes=node02:kaiju:ppn=32 
#PBS -o qsub.stdout
#PBS -e qsub.stderr
#PBS -l walltime=48:00:00
#PBS -m n

# Redirect this job script's outputs

exec > job1.Zhong2000NMCase1  2> job2.Zhong2000NMCase1

# Set an ID for MPI daemon. Mandatory. For when multiple MPI jobs are running on the same node.

export MPD_CON_EXT=$PBS_JOBID
export OMPI_JOBID=$PBS_JOBID

# Run job
echo `date --rfc-3339=ns`": Job started"
#module load uw_stack/py399_petsc3161_openmpi_nodebug underworld/2.12-dev_openmpi_no_debug openmpi/4.1.2 torque
#module load uw_stack/py399_petsc3161_openmpi_nodebug underworld/2.13.0 openmpi/4.1.2 torque
module load underworld/2.13.0
mpiexec -n 32 python3 Zhong2000NMCase1.py > Zhong2000NMCase1.log

echo `date --rfc-3339=ns`": Job finished"

exit 0
