#! /bin/bash


python_name="MollProblem_mercury.py" # 3 dmg



num_layers_arr=(3 4 5 6)
nodes_per_layer_arr=(30 40 50 60)

# sampling_stages_arr=(10 5)
# sampling_stages_arr=(20000 50000)
# steps_per_sample_arr=(10 5)

# nSim_interior_arr=(1024 2048)
# nSim_boundary_arr=(128 256)

sampling_stages_arr=(20000)
steps_per_sample_arr=(10)

nSim_interior_arr=(1024)
nSim_boundary_arr=(128)


count=0

for num_layers in ${num_layers_arr[@]}; do
	for nodes_per_layer in ${nodes_per_layer_arr[@]}; do
        for sampling_stages in ${sampling_stages_arr[@]}; do
            for steps_per_sample in ${steps_per_sample_arr[@]}; do
                for nSim_interior in ${nSim_interior_arr[@]}; do  
                    for nSim_boundary in ${nSim_boundary_arr[@]}; do

                        action_name="num_layers_${num_layers}_nodes_per_layer_${nodes_per_layer}_sampling_stages_${sampling_stages}_steps_per_sample_${steps_per_sample}_nSim_interior_${nSim_interior}_nSim_boundary_${nSim_boundary}"


                        mkdir -p ./job-outs/${python_name}/${action_name}/

                        if [ -f ./bash/${python_name}/${action_name}/train.sh ]; then
                            rm ./bash/${python_name}/${action_name}/train.sh
                        fi

                        mkdir -p ./bash/${python_name}/${action_name}/

                        touch ./bash/${python_name}/${action_name}/train.sh

                        tee -a ./bash/${python_name}/${action_name}/train.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=num_layers_${num_layers}_nodes_per_layer_${nodes_per_layer}_sampling_stages_${sampling_stages}_steps_per_sample_${steps_per_sample}_nSim_interior_{nSim_interior}_nSim_boundary_${nSim_boundary}
#SBATCH --output=./job-outs/${python_name}/${action_name}/train.out
#SBATCH --error=./job-outs/${python_name}/${action_name}/train.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=5
#SBATCH --mem=5G
#SBATCH --time=1-00:00:00
##SBATCH --exclude=mcn53,mcn51,mcn05

####### load modules
module load python/anaconda-2020.11 

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u  /home/bincheng/InequalityEcon/$python_name --num_layers ${num_layers} --nodes_per_layer ${nodes_per_layer}  --sampling_stages ${sampling_stages} --steps_per_sample ${steps_per_sample} --nSim_interior ${nSim_interior} --nSim_boundary  ${nSim_boundary} 

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
		done
	done
done
