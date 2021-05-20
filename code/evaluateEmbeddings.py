from sklearn.utils.linear_assignment_ import linear_assignment
import tensorflow as tf
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def test2NN(model,train_ds,test_ds):
    neigh = KNeighborsClassifier(n_neighbors=1, weights='distance')
    for (batch, (images, labels)) in enumerate(train_ds):
            labels = [1 if y > 4 else 0 for y in tf.reshape(labels, [-1])]
            labels = tf.dtypes.cast(labels, tf.float32) 
            labels = tf.reshape(labels,[-1,1])
            embed_labels,_ = model([images, labels])
            neigh.fit(embed_labels[:, 1:], np.asarray(labels).ravel())
    accuracy_test = 0
    accuracy_train = 0
    den =0
    for (batch, (images, labels)) in enumerate(test_ds):
            den += 1
            labels = [1 if y > 4 else 0 for y in tf.reshape(labels, [-1])]
            labels = tf.dtypes.cast(labels, tf.float32) 
            labels = tf.reshape(labels,[-1,1])
            embed_labels,_ = model([images, labels])
            predicted = neigh.predict(embed_labels[:, 1:])
            acc = accuracy_score(labels, predicted)
            accuracy_test += acc
    accuracy_test = accuracy_test/den
    return accuracy_test
def get_embeddings_labels(model,test_ds):
    Test_Embeddings = []
    for (batch, (images, labels)) in enumerate(test_ds):
        # we need all the labels to calculate clustering acc and nmi
        labels = tf.dtypes.cast(labels, tf.float32) 
        labels = tf.reshape(labels,[-1,1])
        embed_labels,_ = model([images, labels])
        if not len(Test_Embeddings):
            Test_Embeddings = embed_labels
        else:   
            Test_Embeddings = np.append(Test_Embeddings , np.asarray(embed_labels),axis = 0)
    return Test_Embeddings
def clus_acc(ypred, y):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
    
    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
    
    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))
    
    s = np.unique(ypred)
    t = np.unique(y)
    
    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype = np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)
    
    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    # 
    indices = linear_assignment(C)
    row = indices[:][:, 0]
    col = indices[:][:, 1]
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]] )
        count += np.count_nonzero(idx)
    
    return 1.0*count/len(y)

def nmi(ypred, y):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
    
    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
    
    """
#     print (ypred)
#     print (y)
    return normalized_mutual_info_score(y,ypred)