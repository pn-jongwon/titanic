import pandas as pd
import numpy as np

train = pd.read_csv("~/titanic/dataset/train.csv")
test = pd.read_csv("~/titanic/dataset/test.csv")

###############################################################################

train.label = train["Survived"]
train.data = train.drop("Survived", axis = 1)
merged = train.data.append(test)

################################################################################

train.data["Age"] = train.data["Age"].fillna(merged["Age"].mean())
test["Age"] = test["Age"].fillna(merged["Age"].mean())

merged["Cabin"] = merged["Cabin"].fillna("K")
merged["Cabin"] = merged["Cabin"].apply(lambda x: x[0])

train.data["Embarked"] = train.data["Embarked"].fillna(max(set(merged["Embarked"]), key = list(merged["Embarked"]).count))
test["Embarked"] = test["Embarked"].fillna(max(set(merged["Embarked"]), key = list(merged["Embarked"]).count))

train.data["Fare"] = train.data["Fare"].fillna(merged["Age"].median())
test["Fare"] = test["Fare"].fillna(merged["Age"].median())
