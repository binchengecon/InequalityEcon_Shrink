#! /bin/bash



python_name="SannikovProblem_findX_Concave_ODEloss4_robust.py" # 3 dmg
# python_name="SannikovProblem_findX_Concave_ODEloss4_ODEloss.py" # 3 dmg


num_layers_FFNN_arr=(2 3 4)
activation_FFNN_arr=("tanh")
num_layers_RNN_arr=(0)
nodes_per_layer_arr=(40 50 60)

# num_layers_FFNN_arr=(4)
# activation_FFNN_arr=("tanh")
# num_layers_RNN_arr=(0)
# nodes_per_layer_arr=(60)


sampling_stages_arr=(100000)
steps_per_sample_arr=(10)

nSim_interior_arr=(1024)
nSim_boundary_arr=(1)

LearningRate_arr=(0.001)


# num_layers_FFNN_arr=(0)
# activation_FFNN_arr=("tanh")
# num_layers_RNN_arr=(0)
# nodes_per_layer_arr=(30)

# sampling_stages_arr=(10)
# steps_per_sample_arr=(10)

# nSim_interior_arr=(1024)
# nSim_boundary_arr=(1)

# LearningRate_arr=(0.01)


slopearr=(2.45812833 2.41668545 2.10360393 1.85228996 1.519999 1.0556965 0.95382563 0.75974465 0.33722682)
xiarr=(10000 100 10 5 2.5 1 0.8 0.5 0.1)




LENGTH_slope=$((${#slopearr[@]} - 1))



for num_layers_FFNN in ${num_layers_FFNN_arr[@]}; do
    for activation_FFNN in ${activation_FFNN_arr[@]}; do
        for num_layers_RNN in ${num_layers_RNN_arr[@]}; do
            for nodes_per_layer in ${nodes_per_layer_arr[@]}; do
                for sampling_stages in ${sampling_stages_arr[@]}; do
                    for steps_per_sample in ${steps_per_sample_arr[@]}; do
                        for nSim_interior in ${nSim_interior_arr[@]}; do  
                            for nSim_boundary in ${nSim_boundary_arr[@]}; do
                                for LearningRate in ${LearningRate_arr[@]}; do
                                    for  id in $(seq 0 $LENGTH_slope); do

                                        action_name="num_layers_FFNN_${num_layers_FFNN}_activation_FFNN_${activation_FFNN}_num_layers_RNN_${num_layers_RNN}_nodes_per_layer_${nodes_per_layer}/sampling_stages_${sampling_stages}_steps_per_sample_${steps_per_sample}/nSim_interior_${nSim_interior}_nSim_boundary_${nSim_boundary}/LearningRate_${LearningRate}/"


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
#SBATCH --output=./job-outs/${python_name}/${action_name}/train_${slopearr[$id]}.out
#SBATCH --error=./job-outs/${python_name}/${action_name}/train_${slopearr[$id]}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-12:00:00

####### load modules
module load python/anaconda-2021.05

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u  /home/bincheng/InequalityEcon/$python_name --slope ${slopearr[$id]}  --uncertainty ${xiarr[$id]} --num_layers_FFNN ${num_layers_FFNN} --activation_FFNN ${activation_FFNN} --num_layers_RNN ${num_layers_RNN} --nodes_per_layer ${nodes_per_layer}  --sampling_stages ${sampling_stages} --steps_per_sample ${steps_per_sample} --nSim_interior ${nSim_interior} --nSim_boundary  ${nSim_boundary}  --LearningRate ${LearningRate}

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
            done
        done
    done
done