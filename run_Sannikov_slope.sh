#! /bin/bash


# python_name="SannikovProblem.py" # 3 dmg
# python_name="SannikovProblem_findX.py" # 3 dmg
# python_name="SannikovProblem_findX_Concave.py" # 3 dmg
# python_name="SannikovProblem_findX_Concave_ODEloss.py" # 3 dmg
# python_name="SannikovProblem_findX_Concave_ODEloss2.py" # 3 dmg
python_name="SannikovProblem_findX_Concave_ODEloss3.py" # 3 dmg


count=0

num_run=10
# num_run=50
ID_num_run=$((num_run - 1))

lowerslope=2
# upperslope=10
upperslope=2.5


# lowerslope=2.4
# # upperslope=10
# upperslope=2.6


num_backup=1
ID_num_backup=$((num_backup - 1))

learningarr=(0.01 0.001 0.0001)
# learningarr=(0.001)

for learning in ${learningarr[@]}; do
    for id_backup in $(seq 0 $ID_num_backup); do
        for id in $(seq 0 $ID_num_run); do

            action_name="sloperange=[${lowerslope},${upperslope}]/learning_${learning}/${id}"


            mkdir -p ./job-outs/${python_name}/${action_name}/

            if [ -f ./bash/${python_name}/${action_name}/train.sh ]; then
                rm ./bash/${python_name}/${action_name}/train.sh
            fi

            mkdir -p ./bash/${python_name}/${action_name}/

            touch ./bash/${python_name}/${action_name}/train.sh

            tee -a ./bash/${python_name}/${action_name}/train.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=id_${id}
#SBATCH --output=./job-outs/${python_name}/${action_name}/train_${id_backup}.out
#SBATCH --error=./job-outs/${python_name}/${action_name}/train_${id_backup}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-12:00:00
##SBATCH --exclude=mcn53,mcn51,mcn05

####### load modules
module load python/anaconda-2021.05

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u  /home/bincheng/InequalityEcon/$python_name --id ${id} --num ${num_run} --lowerslope ${lowerslope} --upperslope ${upperslope} --backup ${id_backup} --learning ${learning}

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
    done
done