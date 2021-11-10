DATASET=dataset/Timeline17
HEIDELTIME=venv/lib/python3.8/site-packages/tilse/tools/heideltime
CODE=preprocessing

python $CODE/preprocess_tokenize.py --dataset $DATASET
python $CODE/preprocess_heideltime.py --dataset $DATASET --heideltime $HEIDELTIME
python $CODE/preprocess_spacy.py --dataset $DATASET