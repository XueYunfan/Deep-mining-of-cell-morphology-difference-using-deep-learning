import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

#load cell morphology discriptors
PATH = 'SMC-DATA.csv'
data = pd.read_csv(PATH)
data = data.values
fiture = data[:,2:177]
target = data[:,177]

pipe = Pipeline([('scaler',MinMaxScaler()),('forest', RandomForestClassifier())])
param_grid = {'forest__max_depth':[5,10,15,20,25],
	'forest__n_estimators':[10,100,300,500,1000,2000,3000,5000]}

#Grid searching with k-fold validation
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid.fit(fiture,target)
results = pd.DataFrame(grid.cv_results_)
results.to_csv('SMC grid searching.csv')
