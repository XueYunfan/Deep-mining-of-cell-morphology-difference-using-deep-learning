import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#load cell morphology discriptors
PATH = 'SMC-DATA.csv'
data = pd.read_csv(PATH)
data = data.values
fiture = data[:,2:177]
target = data[:,177]

scaler = MinMaxScaler()
scaler.fit(fiture)
fiture_scaled = scaler.transform(fiture)

#fit model and calculate feature importance
forest = RandomForestClassifier(
	n_estimators=500, n_jobs=-1, random_state=0, max_depth= 10)
forest.fit(fiture_scaled, target)
importances=forest.feature_importances_

#save feature importances
data = pd.DataFrame(importances)
data.to_csv('SMC feature importances.csv')
