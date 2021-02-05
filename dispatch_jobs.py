"""
python dispatch_jobs.py
"""
from subprocess import call
num_processors = 1
SLURM = f"""#!/bin/bash
#SBATCH -J single_site
#SBATCH -p jila nistq
#SBATCH -q standard
#SBATCH -n {num_processors}
#SBATCH -N 1
#SBATCH --mem=7.5G
#SBATCH -t 01-00:0:00
#SBATCH --tmp=1GB

# Request email about this job: options include BEGIN, END, FAIL, ALL or NONE
# Additionally, TIME_LIMIT_50, 80 or 90 will send email when the job reaches
# 50%, 80% or 90% of the job walltime limit
# Multiple options can be separated with a comma
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

# Lines after this will be executed as if typed on a command line.
# This script is executed from the directory you submitted your job from;
# unlike the old cluster, there is no need for "cd $PBS_O_WORKDIR"

# You should make sure the appropriate environment module is loaded
# for the software you want to use: this is the "module load" command.
# Replace matlab with the package you'll use.

# module load matlab
module load julia
# The following example runs a MATLAB program stored in example.m
# Replace this with commands to run your job. 
"""

cmd = f"julia ~/repos/singlesite_for_cluster/job.jl"
print("Dispatching...")
with open('scratch.txt', 'w') as filehandle:
    filehandle.write(SLURM + "\n" + cmd)
call('sbatch scratch.txt', shell=True)
