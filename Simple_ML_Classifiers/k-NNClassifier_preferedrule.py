'''
k-NN classifier.
'''

import data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        l2_distances = self.l2_distance(test_point)
        labels = self.train_labels

        arg_euclid_dist = np.argsort(l2_distances)[:k]

        k_number_accumulation = [ 0 for i in range(10)]
        k_number_distances = [ 0.0 for i in range(10)]

        for i in range(k):
            arg = arg_euclid_dist[i]
            clas = int(labels[arg])
            dist = l2_distances[arg]
            k_number_accumulation[clas] = k_number_accumulation[clas] + 1
            k_number_distances[clas] = k_number_distances[clas] + dist

        max_k_number = []
        max_k_v = []
        for i in range(10):
            if k_number_accumulation[i] > 0:
                k_number_distances[i] = k_number_distances[i]/k_number_accumulation[i]

            current_max = np.argmax(k_number_accumulation)
            max_k_number.append(current_max)
            max_k_v.append(k_number_accumulation[current_max])
            k_number_accumulation[current_max] = 0

        number = max_k_number[0]
        if len(max_k_v) > 1 and max_k_v[0] == max_k_v[1]:
            # Compare average distance of the two number classes if one distance is smaller than the other then the following class is the one.
            if k_number_distances[max_k_number[0]] > k_number_distances[max_k_number[1]]:
                number = max_k_number[1]
            elif k_number_distances[max_k_number[0]] == k_number_distances[max_k_number[1]]:
                # If no concensus is brought then reduce k number and repeat process until the tie is broken.
                number = self.query_knn(test_point, k-1)

        return number

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    x = train_data
    y = train_labels
    N = len(y)
    folds = 10

    average_train_losses_per_k = [ 0.0 for i in range(16)]
    for k in k_range:
        losses_per_folds = []
        for i in range(folds):
            range_i = int(i*(N/folds))
            range_i_end = int((i+1)*(N/folds))
            if i == folds-1:
                test_x_fold = x[range_i:]
                test_y_fold = y[range_i:]
            else:
                test_x_fold = x[range_i:range_i_end]
                test_y_fold = y[range_i:range_i_end]
        
            train_x_fold = np.concatenate((x[0:range_i], x[range_i_end:]), axis=0)
            train_y_fold = np.concatenate((y[0:range_i], y[range_i_end:]), axis=0)
            
            knn = KNearestNeighbor(train_x_fold, train_y_fold)
            fold_losses = classification_accuracy(knn, k, test_x_fold, test_y_fold)

            losses_per_folds.append(fold_losses)
            average_for_fold = np.sum(losses_per_folds)/10
        average_train_losses_per_k[k] = average_for_fold

        print("k: {0} with average per fold of : {1}".format(k,average_for_fold))
    plt.plot(average_train_losses_per_k[1:])
    plt.show()
    return average_train_losses_per_k

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''

    false_counter = 0
    for t in range(len(eval_data)):
        if not knn.query_knn(eval_data[t], k) == int(eval_labels[t]):
            false_counter = false_counter + 1
    return (float(len(eval_data)-false_counter))/len(eval_data)
    

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    print("train classification accuracy when k = 1, {0}".format(classification_accuracy(knn, 1, train_data, train_labels)))
    print("train classification accuracy when k = 15, {0}".format(classification_accuracy(knn, 15, train_data, train_labels)))

    print("test classification accuracy when k = 1, {0}".format(classification_accuracy(knn, 1, test_data, test_labels)))
    print("test classification accuracy when k = 15, {0}".format(classification_accuracy(knn, 15, test_data, test_labels)))
    k_s = cross_validation(train_data, train_labels)
    optimal_k = np.argmax(k_s)
    print("optimal k: {0}".format(optimal_k))
    
    knn = KNearestNeighbor(test_data, test_labels)
    print("Test accuracy with optimal {0}".format(classification_accuracy(knn, optimal_k, test_data, test_labels)))

if __name__ == '__main__':
    main()