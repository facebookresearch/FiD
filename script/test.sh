#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --job-name=testqa
#SBATCH --output=/private/home/%u/fid/run_dir/%A
#SBATCH --partition=dev
#SBATCH --mem=400GB
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --comment="efificent qa competition 11/14"
 
export NGPU=8;

port=$(shuf -i 15000-16000 -n 1)
name=$SLURM_JOB_ID
#mp=/checkpoint/gizacard/qacache/31578827_base_100/
mp=pretrained_models/nq_base_dpr/checkpoint/best_dev/
#mp=pretrained_models/nq_large_dpr/
#mp=pretrained_models/trivia_base_dpr/
#mp=pretrained_models/trivia_large_dpr/
#mp=pretrained_models/squad_base_bm25/
#mp=pretrained_models/squad_large_bm25/
#tp=preprocessed_data/squad/squad_bm25_test.json
#tp=preprocessed_data/trivia/trivia_dpr_test.json
#mp=/checkpoint/gizacard/qacache/nq/33190028_base_100/
tp=preprocessed_data/nq/nq_dpr_test.json
#mp=/checkpoint/gizacard/qacache/nq/33193365_base_100
#test_data_path=/checkpoint/fabiopetroni/KILT/retrieved_augmented_datasets/dpr/nq-test-kilt.jsonl
#mp=/checkpoint/gizacard/qacache/eli5/29519738_large_5/
#mp=/checkpoint/gizacard/qacache/eli5/29574194_large_5/
#mp=/checkpoint/gizacard/qacache/eli5/29545053_large_10
#mp=/checkpoint/gizacard/qacache/31479974_base_10/
#tp=~/DPR/contexts/nq_mean_i2_kl/dpr_nq_test.json
#tp=~/DPR/contexts/nqefficientqa_i2_kl/dev.json
#tp=~/DPR/contexts/nqefficientqa_dpr/dev.json
#tp=/checkpoint/fabiopetroni/GENRE/competition/data/nqefficientqa_genre_dev_1.json
#mp=/checkpoint/gizacard/qacache/31578827_base_100
#mp=/checkpoint/gizacard/qacache/31692950_base_100/
#mp=/checkpoint/gizacard/qacache/31689022_base_1 #train genre retriever
#mp=/checkpoint/gizacard/qacache/nq/32138337_base_1/
#tp=/checkpoint/fabiopetroni/GENRE/competition/data/dev_fid_retriever.json
#mp=/checkpoint/gizacard/qacache/genre_nq/32166817_base_1/
#tp=/private/home/gizacard/fid/preprocessed_data/genre/dev_fid_retriever.json
#tp=~egrave/efficientqa/data/kilt/nq-dev-genre.json
#tp=~/DPR/contexts/nq_128/test.json
#tp=~/DPR/contexts/nqefficient_128/dev.json
#tp=~/DPR/contexts/nqefficient_128_compression/dev.json
#tp=genre_data/nq_genre_dev_maxload.json
#tp=~/DPR/contexts/nqefficient_128mean_compression/dev.json
#mp=/checkpoint/gizacard/qacache/31578827_base_100 #nqefficient train dpr
#tp=~/DPR/contexts/nqefficient_128mean64x8_compression/train.json
#tp=~/DPR/contexts/nqefficient_dpr_500/dev.json

#tp=~/DPR/competition_contexts/256mean_lists/dev.json
#mp=/checkpoint/gizacard/qacache/nq/32332744_base_200/
#tp=/private/home/gizacard/DPR/competition_contexts/256mean_overlap/dev.json
#tp=~/DPR/contexts/nqefficient_256mean120p/dev.json
#tp=/checkpoint/fabiopetroni/GENRE/competition/data/nqefficientqa_GAR_dev.json
#tp=~/DPR/competition_contexts/768mean_effcadd1/dev.json
#tp=~/DPR/competition_contexts/at_overlap/dev.json
#tp=/private/home/gizacard/fid/data/nqefficient_gar_v2/dev.json
#mp=/checkpoint/egrave/efficientqa/32640889_large_100/ #quantized model
#mp=/checkpoint/egrave/efficientqa/32605057_large_100/ #quantized model
#tp=~gizacard/DPR/competition_contexts/passages_filter_overlap_list_mean256/dev.json
#tp=/private/home/egrave/efficientqa/data/nqefficient_fol/dev.jsonl
#tp=~/DPR/competition_contexts/passages_filter_overlap_list_largescores_T=0.5/dev_efficientqa.json
#tp=~/DPR/competition_contexts/passages_filter_overlap_list_largescores_T=1.5/dev_efficientqa.json
#tp=~/DPR/competition_contexts/passages_filter_overlap_list_largescores_T=1/dev_efficientqa.json
#mp=/checkpoint/gizacard/qacache/nq/33296713_base_10/
#mp=/checkpoint/gizacard/qacache/nq/33310310_base_10

srun ~/anaconda3/envs/fid/bin/python3 test.py \
        --text_maxlength 250 \
        --model_path $mp \
        --eval_data_path $tp \
        --n_context 10 \
        --per_gpu_batch_size 4 \
        --name $name \
        --checkpoint_dir /checkpoint/gizacard/qacache/test \
	--eval_print_freq 50 \
        --main_port $port \
