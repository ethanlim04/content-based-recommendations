#https://www.youtube.com/watch?v=9siFuMMHNIA
import numpy as np
matrix = np.array([[5, 5, 0, 0], [5, None, None, 0], [None, 4, 0, None], [0, 0, 5, 4], [0, 0, 5, None]])

n_u = 4 #4 users
n_m = 5 #5 movies

#movie_features = np.array([[0.9, 0], [1.0, 0.01], [0.99, 0], [0.1, 1.0], [0, 0.9]])
movie_features = np.array([[1, 0.9, 0], [1, 1.0, 0.01], [1, 0.99, 0], [1, 0.1, 1.0], [1, 0, 0.9]], dtype=float)

n = 2 #number of features

theta = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=float)

#variables
def r(matrix, i, j):
  if(matrix[i][j] != None):
    return 1
  else:
    return 0

#Combine
matrix = np.array([[5, 5, 0, 0], [5, None, None, 0], [None, 4, 0, None], [0, 0, 5, 4], [0, 0, 5, None]])

n_u = 4 #4 users
n_m = 5 #5 movies

#movie_features = np.array([[0.9, 0], [1.0, 0.01], [0.99, 0], [0.1, 1.0], [0, 0.9]])
movie_features = np.array([[1, 0.9, 0], [1, 1.0, 0.01], [1, 0.99, 0], [1, 0.1, 1.0], [1, 0, 0.9]], dtype=float)

n = 2 #number of features

theta = np.array([[2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [2.5, 2.5, 2.5]], dtype=float)

print(theta[0].transpose().dot(movie_features[2]))  #prediction 1 (real value: 5)

def totalLoss(matrix, theta, movie_features, n_m, n_u):
  total = 0
  for i in range(n_m):
    for j in range(n_u):
      if(r(matrix, i, j) == 1):
        total += (theta[j].transpose().dot(movie_features[i]) - matrix[i][j])**2
  return (total)

def loss(matrix, theta, movie_features, n_m, j):
  total = 0
  for i in range(n_m):
    if(r(matrix, i, j) == 1):
      total += (theta[j].transpose().dot(movie_features[i]) - matrix[i][j])**2
  return (total)


def gradient_descent(j, matrix, theta, movie_features, n_m, n_u, k):
  total = 0
  for i in range(n_m):
    if(r(matrix, i, j) == 1):
      total += (theta[j].transpose().dot(movie_features[i]) - matrix[i][j])*movie_features[i][k]
  return total
learn_rate = 0.0001


print(theta[0])

tempOut = 0.00
print("loss before: ", totalLoss(matrix, theta, movie_features, n_m, n_u))
for j in range(n_u):
  breakOut = False
  t = 0
  while(breakOut == False and t < 500000):
    theta[j][0] = float(theta[j][0]) - float(learn_rate * gradient_descent(j, matrix, theta, movie_features, n_m, n_u, 0))
    theta[j][1] = float(theta[j][1]) - float(learn_rate * gradient_descent(j, matrix, theta, movie_features, n_m, n_u, 1))
    theta[j][2] = float(theta[j][2]) - float(learn_rate * gradient_descent(j, matrix, theta, movie_features, n_m, n_u, 2))

    if(t == 0):
      minVal = loss(matrix, theta, movie_features, n_m, j)
    else:
      currentVal = loss(matrix, theta, movie_features, n_m, j)
      if(currentVal > minVal and t>1000):
        print("j:", j, "t:", t)
        print("\tmin:", minVal, "current:", currentVal)
        breakOut = True
      else:
        minVal = currentVal
    t += 1
  print(j, theta[j], minVal)

print("loss after: ", totalLoss(matrix, theta, movie_features, n_m, n_u))  #89.69


print("\n\n\n")
for i in range(5):
  print(matrix[i])
print("\n\n\n")

for i in range(n_m):
  for j in range(n_u):
    if(matrix[i][j] == None):
      matrix[i][j] = round(theta[j].transpose().dot(movie_features[i]), 2)
for i in range(5):
  print(matrix[i])