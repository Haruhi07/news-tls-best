export DATASET=/content/drive/MyDrive/dataset/crisis
export RESULT=./result

python -u experiments/evaluate.py --dataset $DATASET --method clust --output /content/drive/MyDrive/results/crisis/markov-pegasus-clip10.json