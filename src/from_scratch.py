import numpy as np
#temp = [[0, [25.0, 0, 0, 0], [25.0, 0], [0, 0, 25.0], [25.0, 0, 0]], [20.0, [20.0, 0, 0, 0], [20.0, 0], [0, 20.0, 0], [0, 20.0, 0]], [0, [0, 0, 0, 25.0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]], [0, [0, 0, 25.0, 0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]], [0, [25.0, 0, 0, 0], [25.0, 0], [0, 0, 25.0], [25.0, 0, 0]], [20.0, [20.0, 0, 0, 0], [20.0, 0], [0, 0, 20.0], [20.0, 0, 0]], [0, [25.0, 0, 0, 0], [25.0, 0], [0, 0, 25.0], [25.0, 0, 0]], [20.0, [0, 0, 0, 20.0], [20.0, 0], [0, 20.0, 0], [0, 0, 20.0]], [0, [25.0, 0, 0, 0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]], [0, [25.0, 0, 0, 0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]], [0, [25.0, 0, 0, 0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]], [0, [0, 25.0, 0, 0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]], [0, [0, 0, 25.0, 0], [0, 25.0], [25.0, 0, 0], [0, 25.0, 0]]]
temp = [ [0, [0.25, 0, 0, 0], [0.25, 0], [0, 0, 0.25], [0.25, 0, 0]],
         [0.2, [0.2, 0, 0, 0], [0.2, 0], [0, 0.2, 0], [0, 0.2, 0]],
         [0, [0, 0, 0, 0.25], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]],
         [0, [0, 0, 0.25, 0], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]],
         [0, [0.25, 0, 0, 0], [0.25, 0], [0, 0, 0.25], [0.25, 0, 0]],
         [0.2, [0.2, 0, 0, 0], [0.2, 0], [0, 0, 0.2], [0.2, 0, 0]],
         [0, [0.25, 0, 0, 0], [0.25, 0], [0, 0, 0.25], [0.25, 0, 0]],
         [0.2, [0, 0, 0, 0.2], [0.2, 0], [0, 0.2, 0], [0, 0, 0.2]],
         [0, [0.25, 0, 0, 0], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]],
         [0, [0.25, 0, 0, 0], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]],
         [0, [0.25, 0, 0, 0], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]],
         [0, [0, 0.25, 0, 0], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]],
         [0, [0, 0, 0.25, 0], [0, 0.25], [0.25, 0, 0], [0, 0.25, 0]]]
for i in range(len(temp)):
  for q in range(4):
    for j in range(len(temp[i])):
      if(type(temp[i][j]) == list):
        ttemp = temp[i][:j]
        for k in range(len(temp[i][j])):
          ttemp.append(temp[i][j][k])
        temp[i] = ttemp + temp[i][j+1:]
  temp[i].insert(0, 1)
print(temp)
features = np.array(temp, dtype=float)
matrix = np.array(
    [[10, 2], [6, 6], [2, 10], [4, 8], [10, 3], [10, None], [8, 2], [2, 2], [2, 10], [2, 8], [3, 8], [None, None],
     [3, 7]])
expected = np.array(
    [[10, 2], [6, 6], [2, 10], [4, 8], [10, 3], [10, 'low'], [8, 2], [2, 2], [2, 10], [2, 8], [3, 8], ['low', 'high'],
     [3, 7]])

n_u = len(matrix[0])
n_m = len(matrix)

theta = []
for i in range(n_u):
    uTemp = [5 for j in range(len(features) + 1)]
    theta.append(uTemp)

theta = np.array(theta, dtype=float)


def r(matrix, i, j):
    if (matrix[i][j] == None):
        return 0
    else:
        return 1


def totalLoss(matrix, theta, features, n_m, n_u):
    total = 0
    for i in range(n_m):
        for j in range(n_u):
            if (r(matrix, i, j) == 1):
                total += (theta[j].transpose().dot(features[i]) - matrix[i][j]) ** 2
    return total


def loss(matrix, theta, features, n_m, j):
    total = 0
    for i in range(n_m):
        if (r(matrix, i, j) == 1):
            total += (theta[j].transpose().dot(features[i]) - matrix[i][j]) ** 2
    return total


def gradient_descent(j, matrix, theta, features, n_m, n_u, k):
    total = 0
    for i in range(n_m):
        if (r(matrix, i, j) == 1):
            total += (theta[j].transpose().dot(features[i]) - matrix[i][j]) * features[i][k]
    return total


learn_rate = 0.001
print("loss before:", totalLoss(matrix, theta, features, n_m, n_u))

for j in range(n_u):
    breakOut = False
    t = 0
    while (breakOut == False and t < 1000000):
        for k in range(len(theta[0])):
            theta[j][k] = float(theta[j][k]) - float(
                learn_rate * gradient_descent(j, matrix, theta, features, n_m, n_u, k))

        if (t == 0):
            minVal = loss(matrix, theta, features, n_m, j)
        else:
            currentVal = loss(matrix, theta, features, n_m, j)
            if (currentVal > minVal):
                print("j:", j, "t:", t)
                breakOut = True
            else:
                minVal = currentVal
        t += 1
    print("j:", j, "minVal:", minVal)
    print("\t", j, "loss:", currentVal)

print("loss after:", totalLoss(matrix, theta, features, n_m, n_u))

print("\n\n\n")
for i in range(5):
    print(matrix[i])
print("\n\n\n")

for i in range(n_m):
    for j in range(n_u):
        if (matrix[i][j] == None):
            matrix[i][j] = round(theta[j].transpose().dot(features[i]), 2)

for i in range(len(matrix)):
    print(matrix[i], '\t', expected[i])
print('\n\n\n')
