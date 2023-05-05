#! /bin/bash


# python_name="SannikovProblem_noretirement_mercury_robust.py" # 3 dmg
# python_name="SannikovProblem_retirement_mercury_robust.py" # 3 dmg
python_name="SannikovProblem_noretirement_mercury_robust_wider.py" # 3 dmg


xiarr=(1 5 10 50 100 1000 1000000)

count=0

for xi in ${xiarr[@]}; do

    action_name="xi_${xi}"


    mkdir -p ./job-outs/${python_name}/${action_name}/

    if [ -f ./bash/${python_name}/${action_name}/train.sh ]; then
        rm ./bash/${python_name}/${action_name}/train.sh
    fi

    mkdir -p ./bash/${python_name}/${action_name}/

    touch ./bash/${python_name}/${action_name}/train.sh

    tee -a ./bash/${python_name}/${action_name}/train.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=xi_${xi}
#SBATCH --output=./job-outs/${python_name}/${action_name}/train.out
#SBATCH --error=./job-outs/${python_name}/${action_name}/train.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=1-12:00:00
##SBATCH --exclude=mcn53,mcn51,mcn05

####### load modules
module load python/anaconda-2020.11 

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u  /home/bincheng/InequalityEcon/$python_name --xi ${xi}

echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"
# echo ${hXarr[@]}

EOF
    count=$(($count + 1))
    sbatch ./bash/${python_name}/${action_name}/train.sh
done

