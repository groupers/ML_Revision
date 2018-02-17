'''
Conditional Gaussian classifier on MNIST.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
# from tqdm import tqdm
from numpy.linalg import inv, det

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = []
    for k in range(10):
        values = 0
        total_d_equal_k = 0
        for d in range(len(train_data)):
            if(train_labels[d] == k):
                values = values + train_data[d]
                total_d_equal_k = total_d_equal_k + 1
        means.append(values / total_d_equal_k)
    return np.array(means)

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances= []
    means = compute_mean_mles(train_data, train_labels)
    for k in range(10):
        column_row = 0
        total_d_equal_k = 0
        for i, s in enumerate(train_data):
            if k == train_labels[i]:
                sa_mean = (np.array(s) - means[k]).reshape(-1,1)
                column_row = column_row + np.dot(sa_mean, sa_mean.T)
                total_d_equal_k = total_d_equal_k + 1
        covariances.append((column_row/total_d_equal_k))
    return np.array(covariances)

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    covs = []
    for i in range(10):
        cov_diag = np.log(np.diag(covariances[i]).reshape(8,8))
        covs.append(cov_diag)
    all_concat = np.concatenate(covs, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative likelihood:

        log p(x|y, mu, Sigma)
    '''
    probabilities = []

    for d in np.array(digits):
        probability_d = []
        for i in range(10):

            cov = covariances[i] + 0.01 * np.identity(64)

            cov_inv = inv(cov)

            f = -5 * np.log(2 * np.pi)

            s = -0.5 * np.log(det(cov))

            mean = means[i].reshape(-1, 64)
            dv = d.reshape(-1,64)

            dv_mean = dv - mean

            mahalanobis  = np.dot(np.dot(dv_mean, cov_inv),
                dv_mean.T)

            expo = -0.5 * mahalanobis

            # log p(x|y,mu,Sigma) 
            p_x_given_y = f + s + expo 

            probability_d.append(p_x_given_y)

        probabilities.append(probability_d)
    return np.array(probabilities)

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    probabilities = []
    log_gen_likelihood = generative_likelihood(digits, means, covariances)

    p_y = 0.1

    for d in log_gen_likelihood:
        probability_d = []
        for k in range(10):
            
            # likelihood and prior
            probability_d.append(np.exp(d[k]) * p_y)

        probabilities.append(d + np.log(p_y) - np.log(np.sum(probability_d)))

    return np.array(probabilities)

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )
    
    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    all_cond_likelihood = conditional_likelihood(digits, means, covariances)
    assign_to_correct_label = []
    for i in range(len(digits)):
        label = int(labels[i])
        assign_to_correct_label.append(all_cond_likelihood[i, label])
    return np.mean(assign_to_correct_label)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    prob = []
    for cl in cond_likelihood:
        prob.append(np.argmax(cl))

    return prob

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    print("Train data: {0} average conditional likelihood".format(avg))

    avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print("Test data: {0} average conditional likelihood".format(avg))

    classified = classify_data(train_data, means, covariances)
    err_rate = np.count_nonzero(train_labels - classified)
    print("Train accuracy: {0}".format(float((len(classified)-err_rate))/len(classified)))

    classified = classify_data(test_data, means, covariances)
    err_rate = np.count_nonzero(test_labels - classified)
    print("Test accuracy: {0}".format(float((len(classified)-err_rate))/len(classified)))

    # Evaluation
    plot_cov_diagonal(covariances)

if __name__ == '__main__':
    main()