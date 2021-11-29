export DATASET=/content/drive/MyDrive/dataset/crisis
export RESULT=./result
export TOKENIZERS_PARALLELISM=true

python -u experiments/evaluate.py --dataset $DATASET --method clust --output /content/drive/MyDrive/results/crisis/ap-pegasus-xsum-clip4.json