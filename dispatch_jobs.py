"""
python dispatch_jobs.py
"""
import numpy as np
from subprocess import call
num_processors = 1
SLURM = f"""#!/bin/bash
#SBATCH -J single_site
#SBATCH -p jila
#SBATCH -q standard
#SBATCH -n {num_processors}
#SBATCH -N 1
#SBATCH --mem=32G
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
# module load julia
# The following example runs a MATLAB program stored in example.m
# Replace this with commands to run your job. 
"""

cmd = f"~/julia-1.5.3/bin/julia ~/repos/singlesite/elliptical_job.jl"
print("Dispatching...")
interionic_spacing = .1
up_modifier = np.sqrt(3)/2 * interionic_spacing
over_modifer = 1/2 * interionic_spacing
points_inside_circle = []
digits = 2


def gen_points(pt, points_inside_circle, x, y):
    radius = .5
    pt = [round(pt[0], digits), round(pt[1], digits)]
    if pt in points_inside_circle or pt[0]**2 + pt[1]**2 > radius**2:
        return
    else:
        points_inside_circle.append(pt)
        x.append(pt[0])
        y.append(pt[1])
        gen_points([pt[0] + over_modifer, pt[1] - up_modifier], points_inside_circle, x, y)
        gen_points([pt[0] - over_modifer, pt[1] - up_modifier], points_inside_circle, x, y)
        gen_points([pt[0] - over_modifer, pt[1] + up_modifier], points_inside_circle, x, y)
        gen_points([pt[0] + over_modifer, pt[1] + up_modifier], points_inside_circle, x, y)
        gen_points([pt[0] + interionic_spacing, pt[1]], points_inside_circle, x, y)
        gen_points([pt[0] + interionic_spacing, pt[1]], points_inside_circle, x, y)
        gen_points([pt[0] - interionic_spacing, pt[1]], points_inside_circle, x, y)
        return points_inside_circle, x, y

pairs, x, y = gen_points([0, 0], [], [], [])

for i, xx in enumerate(x):
    yy = y[i]
    with open('scratch.txt', 'w') as filehandle:
        filehandle.write(SLURM + "\n" + cmd + f" {xx} {yy}")
    call('sbatch scratch.txt', shell=True)
