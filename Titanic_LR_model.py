import numpy as np 
import pandas as pd

d = pd.read_csv("train.csv")
t = pd.read_csv("test.csv")
ids = t["PassengerId"]

d.head(5)