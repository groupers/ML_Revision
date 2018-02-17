'''
Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))

    # Using regularization instead of prior
    extra_labels = []
    for i in range(10):
        extra_labels.append(i)
        extra_labels.append(i)
    new_train_labels = np.append(train_labels, extra_labels)
    extra_data = [np.array([1 for i in range(64)]).reshape(1,-1),
                  np.array([0 for i in range(64)]).reshape(1,-1)]*10
    join_data = [train_data] +  extra_data
    new_train_data = np.concatenate(join_data, 0)

    # Computation
    for k in range(10):
        total = 0
        for i in range(len(new_train_data)):
            if new_train_labels[i] == k:
                total = total + 1
                eta[k] = eta[k] + new_train_data[i]
        eta[k] = eta[k] / total

    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    to_concat_imgs = []
    for i in range(10):
        img_i = class_images[i].reshape(8,8)
        to_concat_imgs.append(img_i)
    plt.imshow(np.concatenate(to_concat_imgs, 1), cmap="gray")
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    # Bernouilli == binomial(1) distribution (solution proposed on stackoverflow)
    generated = [np.random.binomial(1, eta[k]) for k in range(10)] 
    return generated

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    gen_likelihood = []
    for i in range(bin_digits.shape[0]):
        ith_ks = []
        for k in range(10):
            accumulation = []
            for r in range(bin_digits.shape[1]):
                if bin_digits[i][r] != 1:
                    accumulation.append(np.log(1-eta[k,r]))
                else:
                    accumulation.append(np.log(eta[k,r]))
            ith_ks.append(np.sum(accumulation))
        gen_likelihood.append(ith_ks)
    return np.array(gen_likelihood)


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_likelihood = generative_likelihood(bin_digits, eta)
    cond_likelihood = []
    for i in range(bin_digits.shape[0]):
        accumulation = []
        p_y = 0.1
        for k in range(10):
            accumulation.append(np.exp(gen_likelihood[i,k]) * p_y)
        logged_prob = np.log(np.sum(np.array(accumulation)))
        cond_likelihood.append(gen_likelihood[i] + np.log(p_y) - logged_prob)
    return np.array(cond_likelihood)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    all_cond_likelihood = conditional_likelihood(bin_digits, eta)

    assign_to_correct_label = []
    for i in range(np.array(bin_digits).shape[0]):
        label = int(labels[i])
        assign_to_correct_label.append(all_cond_likelihood[i, label])

    return np.mean(assign_to_correct_label)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''

    cond_likelihood = conditional_likelihood(bin_digits, eta)
    prob = []
    for cl in cond_likelihood:
        prob.append(np.argmax(cl))
    return prob

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    plot_images(np.array(generate_new_data(eta)))

    avg = avg_conditional_likelihood(train_data, train_labels, eta)
    print("Train data: {0} average log conditional likelihood".format(avg))

    avg = avg_conditional_likelihood(test_data, test_labels, eta)
    print("Test data: {0} average log conditional likelihood".format(avg))

    classified = classify_data(train_data, eta)
    err_rate = np.count_nonzero(train_labels - classified)
    print("Train accuracy: {0}".format(float((len(classified)-err_rate))/len(classified)))

    classified = classify_data(test_data, eta)
    err_rate = np.count_nonzero(test_labels - classified)
    print("Test accuracy: {0}".format(float((len(classified)-err_rate))/len(classified)))

if __name__ == '__main__':
    main()
