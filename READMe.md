# Toxic comment Classification
This is use case of LSTMs on sentement analysis. The dataset is a bunch of comments with
6 possible lablels of toxic comment. Any comment can have 0-6 labels.
We will start using a simple 2-layer lstm and then experiment with different model 
architectures.

## USER GUIDE
---------------------
1. Go to the Input directory and run:
```
bash download.sh
```
This will download glove embeddings into the data/glove_embeddings/Embeddings dir

2. Download Data files from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
Then place unziped datafiles into data dir

## Files
There are 4 main files being used.
1. preprocess: contains functions for handling preproceccing of data
2. helper_functions: contain methods for creating dictionaries and reading word
embeddings
3. "_mod" scripts: scripts that hold a keras model
4. run_mod: script used to train/test a model

All imports outside of self-contained methods are in
imports.py file

### Current Results
LSTMs are used here. In particular, a bidirecitonal lstm with 2 dense layers.
After training model for 2 epochs, the accuracy was at 97%. There is a lot of room
to improve. We will experiment with transformers and also different architectures for lstms.


