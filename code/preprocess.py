import pandas as pd
from string import punctuation

dat = pd.read_csv("input/train.csv")

dat['comment_text']  = map(lambda x: x.lower(),dat['comment_text'])

comments =
