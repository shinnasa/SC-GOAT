#!/bin/sh
#SBATCH --partition gpu
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 12
#SBATCH --gpus 1
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-user=s.nakamura.sakai@yale.edu
#SBATCH --mail-type=ALL

lspci | grep -i vga
lscpu | grep "^CPU(s):" | awk '{print $2}'
free -h | grep "Mem:" | awk '{print $2}'