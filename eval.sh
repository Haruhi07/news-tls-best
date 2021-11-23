export DATASET=./dataset
export RESULT=./result
export TOKENIZERS_PARALLELISM=true

python -u experiments/evaluate.py --dataset $DATASET/Timeline17 --method clust --output /content/drive/MyDrive/results/markov-pegasus-clip7.json