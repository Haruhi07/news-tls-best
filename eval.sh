export DATASET=/content/drive/MyDrive/dataset
export RESULT=./result

python -u experiments/evaluate.py --dataset $DATASET/crisis --method clust --output /content/drive/MyDrive/crisis/clust-clip4.json