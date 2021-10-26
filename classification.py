import numpy as np
import csv
import process_data
import random
import heapq
import matplotlib.pyplot as plt
import os
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
# Add three files :
path_image_light = "C:/Users/Administrator/Documents/Project_I/light/"
path_image_normal = "C:/Users/Administrator/Documents/Project_I/normal/"
path_image_dark = "C:/Users/Administrator/Documents/Project_I/dark/"

# Convert array list to array numpy.
data = np.asarray(process_data.open_file('max'))

# Train is 70 percent size data and test is 30 percent size data.
X_train, X_test, Y_train, Y_test = train_test_split(data[:, :3], data[:, 3], test_size=0.3, random_state=1)
X_train, X_test = np.array(X_train, dtype=int), np.array(X_test, dtype=int)
X_train, X_test = X_train/255, X_test/255


init_cluster = X_train[np.random.choice(X_train.shape[0], len(process_data.colors))]


# Feature image while environment
def feature_environment(choose_environment, string='max'):
    """
    :param choose_environment:
    :param string:
    :return:
    """
    if choose_environment == 'light':
        choose_environment = path_image_light
    elif choose_environment == 'normal':
        choose_environment = path_image_normal
    else:
        choose_environment = path_image_dark
    variables_img = []
    for image in os.listdir(choose_environment):
        img = process_data.Image(choose_environment + image) if string == 'max' else process_data.Image_Average(choose_environment + image)
        variables_img.append([str(img.blue), str(img.green), str(img.red), str(img.name_color)])

    variables_img = np.asarray(variables_img)
    features = np.array(variables_img[:, :3], dtype=int)
    colors = variables_img[:, 3]
    return features, colors


def draw(accuracy_dark, accuracy_normal, accuracy_light):
    plt.plot([i for i in range(1, 11)], accuracy_dark, 'g*-', label='environment_dark')
    plt.plot([i for i in range(1, 11)], accuracy_normal, 'b*-', label='environment_normal')
    plt.plot([i for i in range(1, 11)], accuracy_light, 'r*-', label='environment_light')

    plt.title("Project I")
    plt.xlabel('K nearest neighbors')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()


# Code algorithm K-nearest-neighbor.
class Knearestneighbor:
    # Create init K - nearest - neighbors.
    def __init__(self, k_nearest=1, norm=2, random_image=False):
        self.k_nearest = k_nearest
        self.norm = norm

    # Understand x is : test data.
    def predict_X_test(self):
        # If k_nearest == 1 => Only find distance min and end.
        if self.k_nearest == 1:
            predict = []
            for test in X_test:
                list = process_data.norm(X_train, test, self.norm)
                predict.append(Y_train[list.index(min(list))])
            # Accuracy Y_test and predict.
            # print("Accuracy of 1NN: %.2f %%" % (100 * accuracy_score(Y_test, predict)))
            return 100*accuracy_score(Y_test, predict)
        elif 1 < self.k_nearest <= X_train.shape[0]:
            predict = []
            for test in X_test:
                list_color = []
                list = process_data.norm(X_train, test, self.norm)
                for iteration in range(self.k_nearest):
                    list_color.append(Y_train[list.index(min(list))])
                    list[list.index(min(list))] = max(list)
                predict.append(max(list_color, key=list_color.count))
                del list_color
            # print("Accuracy of {0}NN: {1} %%".format(self.k_nearest, (100 * accuracy_score(Y_test, predict))))
            return 100*accuracy_score(Y_test, predict)
        else:
            raise Exception('K-nearest not exceed {}.'.format(self.k_nearest))

    def predict_real_data(self, img_feature):
        if self.k_nearest == 1:
            list = process_data.norm(X_train, np.array(img_feature))
            return Y_train[list.index(min(list))]

    # Use specific realtime.
    def draw(self):
        pass


# Code algorithm K-means.
class Kmeans:

    # Create init K - means.
    def __init__(self, k_cluster=len(process_data.colors), norm=2):
        # Count K cluster include : red , green , blue , violet , black , orange , yellow , white.
        self.K_cluster = len(process_data.colors)
        # Point cluster init random.
        self.cluster = dict(zip(process_data.colors, init_cluster))
        self.norm = norm

    # Implement algorithms K - means.
    def algorithms_kmeans(self):
        """
          Step 1 : init k cluster -> Finished.
          Step 2 : Calculator distance between two data points with three parameters :
          + Blue channels.
          + Green channels.
          + Red channels.
        """
        present_color = []
        for train in X_train:
            list = process_data.norm(init_cluster, train, self.norm)
            present_color.append(process_data.colors[list.index(min(list))])

        new_color =[]
        while True:
            cluster_new = process_data.new_cluster(present_color, X_train)

            for train in X_train:
                list = process_data.norm(cluster_new, train, self.norm)
                new_color.append(process_data.colors[list.index(min(list))])

            if new_color == present_color:
                self.cluster = cluster_new
                break
            else:
                present_color = new_color

    def draw(self):
        pass

class SoftmaxRegression:
    
    def __init__(self, n_class = 4, iterations = 100, learning_rate = .001, tol = 1e-10):
        self.n_class = n_class
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.tol = tol
        self.weights = np.random.randn(X_train.shape[1] + 1, len(process_data.colors))

    # Compute loss function 
    def cost(self):
        # Value cost when compute.
        Y_coding = process_data.one_hot_coding(Y_train)
        A = process_data.softmax(self.Xtrain.dot(self.weights))
        return -np.sum(Y_coding*np.log(A))

    # Compute gradient of loss function
    def gradient(self):
        pass

    def train_data(self):
        """
            Train algorithms softmax regression.
                + Step 1: 
                + Step 2:
                + Step 3:
        """
        # columns four have value 1, specific add bias in problem 
        Xtrain = np.hstack((X_train, np.ones((len(X_train), 1))/255))
        random_id = np.random.permutation(len(Xtrain))
        count = 0
        check_w_after = 5
        xi = Xtrain[0, :].reshape(1, Xtrain.shape[1])
        while count < self.iterations:
            for i in random_id:
                xi = Xtrain[i, :].reshape(1, Xtrain.shape[1])
                yi = process_data.encoding([Y_train[i]])
                ai = process_data.softmax(xi.dot(self.weights))
                weights = self.weights
                self.weights = self.weights + self.learning_rate*(xi.T.dot((yi - ai)))
                count += 1
                # Stopping criteria.
                if count%check_w_after == 0:
                    if np.linalg.norm(self.weights - weights) < self.tol:
                        # Exit loop when not conditional [np.linalg.norm(self.weights - weights) < self.tol]
                        break
            
        print(count)

    def save_weights(self):
        return self.weights

    def predict(self):
        predict = []
        Xtest = np.hstack((X_test, np.ones((len(X_test), 1))/255))
        Probability_predict = process_data.softmax(Xtest.dot(self.weights))
        for i in range(len(Probability_predict)):
            list_predict_color = []
            for j in range(len(process_data.colors)):
                list_predict_color.append(1 if Probability_predict[i][j] == max(Probability_predict[i]) else 0)
            predict.append(list_predict_color)
            del list_predict_color
        predict = np.array(predict)
        cols_rows = process_data.decoding(predict)[1]
        Y_predict = []
        for index in cols_rows:
            Y_predict.append(process_data.colors[index])
        Y_predict = np.asarray(Y_predict)
        element_predict = [True if Y_predict[index] == Y_test[index] else False for index in range(len(Y_test))]
        accuracy = element_predict.count(True) / len(Y_test)
        print(accuracy)
        

if __name__ == "__main__":
    Xreal = [37, 37, 0, 1] 
    object = SoftmaxRegression()
    object.train_data()
    object.predict()  
    Xreal = np.array(Xreal).reshape(1, len(Xreal))/255
    ai = process_data.softmax(Xreal.dot(object.save_weights()))
    print(ai)