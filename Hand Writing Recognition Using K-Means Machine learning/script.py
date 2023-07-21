import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.DESCR)
print(digits.data)
print(digits.target)
plt.gray() 
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])
model = KMeans(n_clusters = 10, random_state = 42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = [
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.30,5.41,6.10,6.48,1.30,0.00,0.00,0.00,0.15,2.44,1.53,3.97,4.42,0.00,0.00,0.00,0.00,0.00,0.00,1.60,5.72,0.00,0.00,0.00,0.00,0.00,0.00,2.29,5.80,0.00,0.00,0.00,0.00,0.00,1.68,6.86,2.14,0.00,0.00,0.00,0.00,1.07,7.63,6.94,4.50,1.60,0.00,0.00,0.00,0.00,0.46,2.37,3.66,1.98,0.00,0.00],
[0.00,0.00,0.31,2.44,2.75,0.00,0.00,0.00,0.00,1.15,6.48,6.25,7.63,3.74,0.08,0.00,0.15,6.71,2.82,0.00,1.22,6.25,2.21,0.00,1.52,6.10,0.00,0.00,0.00,4.57,2.29,0.00,2.29,5.34,0.00,0.00,0.00,5.87,1.91,0.00,0.46,6.56,4.57,3.05,5.11,6.18,0.30,0.00,0.00,0.92,3.89,4.58,3.66,0.30,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.38,2.29,1.45,0.54,0.00,0.00,0.00,0.00,1.30,5.42,6.48,7.17,7.63,3.59,0.00,0.00,0.00,0.00,0.00,1.98,6.86,0.76,0.00,0.00,0.30,4.42,4.20,6.48,4.65,0.61,0.00,0.00,0.15,2.90,5.49,6.79,5.34,1.83,0.00,0.00,0.00,0.38,6.79,1.60,0.00,0.00,0.00,0.00,0.00,3.36,4.58,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.84,5.72,6.10,6.10,5.64,2.90,0.00,0.00,0.23,1.98,1.53,2.07,7.40,2.59,0.00,0.00,0.84,5.26,5.34,6.25,6.41,1.53,0.00,0.00,0.23,2.22,4.20,6.71,4.19,2.44,0.00,0.00,0.00,0.46,6.86,2.44,0.00,0.00,0.00,0.00,0.00,0.84,3.81,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
]

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')











