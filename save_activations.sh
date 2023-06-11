#!/bin/bash
#SBATCH -o log/%j-save_activations.log
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

# set environment variables
export PATH=$ORDINAL_PROBING_ROOT:$PATH

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_DIR=/home/gridsan/groups/maia_mechint/ordinal_probing/results
export FEATURE_DATASET_DIR=$ORDINAL_PROBING_ROOT/feature_datasets/processed_datasets
export TRANSFORMERS_CACHE=/home/gridsan/wgurnee/mechint/ordinal-probing/models
export HF_DATASETS_CACHE=/home/gridsan/groups/maia_mechint/ordinal_probing/hf_home
export HF_HOME=/home/gridsan/groups/maia_mechint/ordinal_probing/hf_home

sleep 0.1  # wait for paths to update

# activate environment and load modules
source $ORDINAL_PROBING_ROOT/ord/bin/activate


PYTHIA_MODELS=('pythia-70m' 'pythia-160m' 'pythia-410m' 'pythia-1b' 'pythia-1.4b' 'pythia-2.8b' 'pythia-6.9b')

# for model in "${PYTHIA_MODELS[@]}"
# do
#     python activations.py --model $model --feature_dataset pile_data_source

#     python activations.py --model $model --feature_dataset europarl_lang
# done

python activations.py --model pythia-6.9b --feature_dataset pile_data_source --batch_size 16

python activations.py --model pythia-6.9b --feature_dataset europarl_lang --batch_size 16