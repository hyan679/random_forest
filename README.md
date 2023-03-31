# 1 Introduction

This code comes from assignment 1 of COMP5318: Machine Learning and Data Mining in USYD.

I hope this code can be used as evidence to evaluate my coding ability and data analysis ability, if necessary.

I am required not to use scikit-learn and must write the core code of machine learning by ourselves.

And the accuracy is 75% which is the same as random forest class from scikit-learn

# 2. Data Set

The data set of this assignment is Fashion MNIST, which can be downloaded from https://github.com/zalandoresearch/fashion-mnist

## 2.1 Context

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Contributors intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

## 2.2 Content

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. 

## 2.3 Labels

Each training and test example is assigned to one of the following labels:

|Label|Description|
|----|----|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag| 
|9|Ankle boot|

## 2.4 Dataset Detail

Datasets obtained from different sources may have slightly different structures or values.

In the data set provided by the lecture, each image has been flattened from 2D to 1D, which means each image has become a vector of 1*784. 

Each pixel-value is between 0 and 1. 

The train data includes 30,000 images, takes up 90MB, so the shape of train data is (30000, 784)

The train label inclueds 30,000 labels related to train data, so the the shape of label data is (30000, 1)

The test data and test label share the same patten wth train data set, whose shape is (2000, 784) and (2000, 1)

# 3 Goal

I need to generate Python code to complete the classification algorithm and the higher the accuracy, the better

I am required not to use scikit-learn and must write the core code of machine learning by ourselves

# 4 Algorithm Theory: Random Forest

Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems.

## 4.1 Decision trees

Since the random forest model is made up of multiple decision trees, it would be helpful to start by describing the decision tree algorithm briefly. Decision trees start with a basic question, such as, “Should I surf?” From there, you can ask a series of questions to determine an answer, such as, “Is it a long period swell?” or “Is the wind blowing offshore?”. These questions make up the decision nodes in the tree, acting as a means to split the data. Each question helps an individual to arrive at a final decision, which would be denoted by the leaf node. Observations that fit the criteria will follow the “Yes” branch and those that don’t will follow the alternate path.  Decision trees seek to find the best split to subset the data, and they are typically trained through the Classification and Regression Tree (CART) algorithm. Metrics, such as Gini impurity, information gain, or mean square error (MSE), can be used to evaluate the quality of the split.  

This decision tree is an example of a classification problem, where the class labels are "surf" and "don't surf."

While decision trees are common supervised learning algorithms, they can be prone to problems, such as bias and overfitting. However, when multiple decision trees form an ensemble in the random forest algorithm, they predict more accurate results, particularly when the individual trees are uncorrelated with each other.

## 4.2 Ensemble methods

The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness to create an uncorrelated forest of decision trees. Feature randomness, also known as feature bagging or “the random subspace method”, generates a random subset of features, which ensures low correlation among decision trees. This is a key difference between decision trees and random forests. While decision trees consider all the possible feature splits, random forests only select a subset of those features.

# 5 Investigation

See investigate.ipynb for details

# 6 implementation

See random_forest.ipynb for details

# 7 Technical Environments
I run these 2 files in AWS EC2 Instance type, whose type is t2.micro with 1 vCPU and 1 GiB Memory 