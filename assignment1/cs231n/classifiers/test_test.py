import numpy as np

def predict_labels(dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    #num_train = dists.shape[1]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      pos_k_min = np.argsort(dists[i],axis=None)
      for j in range(k):
        closest_y.append(y_train[pos_k_min[j]])
      
      y_pred[i] = most_common(closest_y)
      
    return y_pred

def most_common(x):
    #x = np.sort(x)
    dict_x = {}
    for i in range(len(x)):
      if x[i] not in dict_x:
        dict_x[x[i]] = 1
      else:
        dict_x[x[i]] += 1
    return max(dict_x, key=dict_x.get)

def compute_distances_no_loops(X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    dists = np.sqrt(np.sum(X**2,axis=1).reshape(num_test,1) + np.sum(X_train**2,axis=1) - 2 * X.dot(X_train.T))
    return dists

X = np.random.randn(5,10)
X_train = np.random.randn(15,10)

dists = compute_distances_no_loops(X)
print(np.sum(X**2,axis=1).reshape(5,1).shape)
print(np.sum(X_train**2,axis=1).shape)
print((np.sum(X**2,axis=1).reshape(5,1) + np.sum(X_train**2,axis=1)).shape)
