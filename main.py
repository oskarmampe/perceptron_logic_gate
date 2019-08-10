from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]

plt.scatter([x[0] for x in data], [x[1] for x in data], c=labels)

classifier = Perceptron(max_iter=40)
classifier.fit(data, labels)

x1_values = np.linspace(0, 1, 100)
x2_values = np.linspace(0, 1, 100)

point_grid = list(product(x1_values, x2_values))

score = classifier.score(data, labels)
distances = classifier.decision_function(point_grid)
abs_distances = [abs(x) for x in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))

print(score)
print(distances)

heatmap = plt.pcolormesh(x1_values, x2_values, distances_matrix)
plt.colorbar(heatmap)

plt.show()
