#Self-Organizing Map

# Importing the libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('SOM_Data_V1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 14, sigma = 1.0, learning_rate = 1.0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 150)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = [ 'o', 's', 'x', '*', 'v']
colors = ['r', 'g', 'y', 'b', 'm']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# Extracting Data Points
mappings = som.win_map(X)
outliers = np.concatenate((mappings[(1,1)], mappings[(1,8)], mappings[(2,2)], mappings[(2,3)], mappings[(3,2)], mappings[(3,3)], mappings[(4,3)], mappings[(5,2)], mappings[(6,5)], mappings[(7,3)], mappings[(7,4)], mappings[(7,6)], mappings[(8,3)], mappings[(8,8)]), axis = 0)
outliers = sc.inverse_transform(outliers)
cluster = np.concatenate((mappings[(9,8)]), axis = 0)
cluster = sc.inverse_transform(cluster)