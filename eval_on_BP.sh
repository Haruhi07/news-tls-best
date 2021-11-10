DATASET=/work/hs20307/Dataset
RESULT=./result

python experiments/evaluate.py --dataset $DATASET/t17 --method clust --output $RESULT/t17.clust.json