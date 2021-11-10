export DATASET=~/Dataset
export RESULT=./result
export TOKENIZERS_PARALLELISM=true

python -u experiments/evaluate.py --dataset $DATASET/t1 --method clust --output $RESULT/t17.clust.json > log.txt