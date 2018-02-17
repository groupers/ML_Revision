import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)
from numpy import linalg as LA
from tqdm import tqdm
import seaborn as sb

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta

    def update_params(self, params, grad, mu_previous=0):
        # Update parameters using GD with momentum and return
        # the updated parameters
        mu = mu_previous * self.beta - self.lr * grad
        params += mu
        return (params, mu)

def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    mu_previous = 0
    for _ in range(steps):

        current_update = optimizer.update_params(w, func_grad(w), mu_previous)
        mu_previous = current_update[1]
        w = current_update[0]
        w_history.append(w)
        # Optimize and update the history
        
    return w_history

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        partial_sum = np.array([0.0 for w in self.w])
        for i in range(X.shape[0]):
            partial_sum += np.max(1 - y[i] * (np.dot(self.w, X[i])), 0)
        # hinge_loss = 0.5 * LA.norm(self.w)**2 + (self.c/X.shape[0]) * current_sum
        return (self.c/X.shape[0]) * partial_sum

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        sub_gradients = self.w
        sample_size = X.shape[0]
        feature_size = X.shape[1]
        gradients = np.zeros(feature_size)
        for i in range(sample_size):
            for j in range(feature_size):
                if y[i] * np.dot(self.w,X[i]) < 1:
                    part_g = -y[i] * X[i,j]
                    # Bias
                    if feature_size-1 == j:
                        part_g = -y[i]
                else:
                    part_g = 0.0

                gradients[j] = gradients[j] + part_g

        # To remove regularization on the bias.
        self.w[-1] = 0.0
        return self.w + (self.c/X.shape[0]) * gradients

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        all_elems = np.dot(X, self.w)
        signed = []
        for i in all_elems:
            if i <= 0:
                signed.append(-1)
            else:
                signed.append(1)

        return np.array(signed)

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    svm = SVM(penalty, train_data.shape[1])
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)

    for t in tqdm(range(iters)):
        batchs = batch_sampler.get_batch()
        X, y = batchs
        grad = svm.grad(X, y)
        mu_previous = 0
        current_update = optimizer.update_params(svm.w, grad, mu_previous)
        mu_previous = current_update[1]
        svm.w = current_update[0]
    return svm

def plot_weights(w):
    w = w.reshape(28,28)
    plt.imshow(w, cmap='viridis')
    plt.show()

if __name__ == '__main__':
    # for i in [0, 0.9]:
    #     sgdo = GDOptimizer(1.0, i)
    #     history = optimize_test_function(sgdo)
    #     plt.plot(history, label='beta: {0}'.format(i))
    #     plt.title('SGD without/with Momentum')
    # plt.ylabel('Weight range')
    # plt.xlabel('Steps')
    # plt.legend()
    # plt.show()

    train_data, train_targets, test_data, test_targets = load_data()

    for i in [0.0, 0.1]:
        optimizer = GDOptimizer(0.05, i)
        Train_plus_one_data = np.column_stack((train_data, np.ones(np.shape(train_data)[0])))
        Test_plus_one_data = np.column_stack((test_data, np.ones(np.shape(test_data)[0])))
        svm_op = optimize_svm(Train_plus_one_data, train_targets, 1.0, optimizer, 100, 500)

        print('When beta is equal to : {0}'.format(i))
        print('Training accuracy: {0}'.format(np.mean(svm_op.classify(Train_plus_one_data) == train_targets)))
        print('Training Mean hinge loss: {0}'.format(np.mean(svm_op.hinge_loss(Train_plus_one_data, train_targets))))
        print()
        print('Test accuracy: {0}'.format(np.mean(svm_op.classify(Test_plus_one_data) == test_targets)))
        print('Testing Mean hinge loss: {0}'.format(np.mean(svm_op.hinge_loss(Test_plus_one_data, test_targets))))
        svm_op.w = svm_op.w[:-1]
        plot_weights(svm_op.w)
        print()

