import numpy as np 
import pandas as pd

d = pd.read_csv("train.csv")
t = pd.read_csv("test.csv")
ids = t["PassengerId"]

d.head(5)

from sklearn import preprocessing

def preprocess(data):
        
    data = data.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis=1)
    
    #numericals to clean
    cols = ["Age", "SibSp", "Fare", "Parch"]
    
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
        
    data.Embarked.fillna("X", inplace=True)
    
    le = preprocessing.LabelEncoder()
    
    cols = ["Sex", "Embarked"]
    
    for col in cols:
        data[col] = le.fit_transform(data[col])
        #print(le.classes_)
    return data

d = preprocess(d)
t = preprocess(t)

d.head(5)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = d["Survived"]
X = d.drop("Survived", axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=38)

res = LogisticRegression(random_state=0, max_iter=500).fit(X_train, y_train)
preds = res.predict(X_val)

from sklearn.metrics import accuracy_score
accuracy_score(y_val, preds)

sm_Preds = res.predict(t)

df = pd.DataFrame({"PassengerId": ids.values, "Survived": sm_Preds})

df.to_csv("submission.csv", index=False)