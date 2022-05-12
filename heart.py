import pandas as pd
import numpy as np
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("heart.csv")

x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

rand_clf = RandomForestClassifier(n_estimators=1000, random_state = 35)
rand_clf.fit(x, y)
joblib.dump(rand_clf,"model")

