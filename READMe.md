# Toxic comment Classification
This repo is a fully run-able
This is use case of LSTMs on sentement analysis. The dataset is a bunch of comments with
6 possible lablels of toxic comment. Any comment can have 0-6 labels.

The model that is being used is a bi-directional LSTM with 2 dense layers.

## USER GUIDE
---------------------
1. Go to the Input directory and run:
```
bash download.sh
```
This will download glove embeddings into the data/glove_embeddings/Embeddings dir

2. Download Data files from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
Then place unziped datafiles into data dir

3. Open up the 'Demo' Notebook. Everything should be runable.

If you are getting a messge that CUDA is out of memory, try reducing the batch-size of the val/test data

## Files

1. datasets.py: this file contains the ToxicDataset Class that will be used in the pytorch dataloaders
2. model.py: this is where the model class is stored
3. run_mod.py: this is the script version of the notebook. Nothing new going on here

All imports outside of this module-specific  methods are in
imports.py file

### Current Results
LSTMs are used here. In particular, a bidirecitonal lstm with 2 dense layers.
After training model for 2 epochs, the accuracy was at 97%. There is a lot of room
to improve /s 

