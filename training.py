"""ici on entraine un modele on utilison l'algorithme random forest regressor"""


import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np


#charger les donnes a partir du fichier CSV

features = pd.read_csv('C:/Users/hp/test1.csv')
features.head(5)
x1=pd.DataFrame(features, columns= ['std','beta','noise'])
y1=pd.DataFrame(features, columns= ['seuil'])

print(x1.shape)
x=x1
y=y1
from sklearn.ensemble import RandomForestRegressor

# create regressor object
print(x1.shape)
header=['std','beta','noise','seuil']
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# fit the regressor with x and y data
regressor.fit(x, y.values.ravel())

#tester le model
test=x1.tail(150)
Y_pred = regressor.predict(test)
print(test)

#afficher le score du model
print(regressor.score(test,y1.tail(150)))

#sauvgarder le model
filename = 'finale_model23.sav'
pickle.dump(regressor, open(filename, 'wb'))
fig = plt.figure(figsize=(25,20))

#affiche une arbre de disition
_ = tree.plot_tree(regressor.estimators_[0]
                   , feature_names=header,filled=True,max_depth=3)
fig.savefig('rf_individualtree1.png')