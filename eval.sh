export DATASET=~/Dataset
export RESULT=./result
export TOKENIZERS_PARALLELISM=true

python -u experiments/evaluate.py --dataset $DATASET/t1 --method clust --output /content/drive/MyDrive/results/news-tls-best.json