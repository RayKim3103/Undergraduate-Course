import numpy as np 
import pandas as pd
import random
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


def sign(x):
    return 2 * (x >= 0) - 1

class SimpleRandomForest:
    """Simple RandomForest class"""
    
    def __init__(self, num_estimators, max_depth=None, max_features=None):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int. The number of decision trees
                max_depth    : int. 
                classifiers  : list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[0]) is the dictionary denoted as classifier.
                             The classifier has key 'feature_indices' and 'classifier'. The values are the indices of features 
                             for that classifier and the Decsion Tree classifier.
                             
        Args:
            
        Returns:
        
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.max_depth = max_depth
        self.classifiers = []
        
        
    def fit(self, X, y):
        """
        Description:
            Build classifiers from the training set.
               
        Args:
            X: (N, d) numpy array. Training/testing samples.
            y: (N,) numpy array. Ground truth classes.
            
        Returns:
            
            
        """
        self.X = X
        self.y = y

        self.bagging()
        
        
    def bagging(self):
        """
        Description:
            Build simple randomforest classifier. Follow the procedures described in .ipynb file.
            Bagging - Each tree is grown using a bootstrap sample of training data

            Use DecisionTreeClassifier(max_depth=self.max_depth) as a decision tree.
            # step 1. Create random vectors
            # step 2. Use random vector to build multiple decision trees
            # step 3. Combine decision trees
        
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###
        for _ in range(self.num_estimator):
            X_bootstrapped, y_bootstrapped = self.sample_random_vectors()
            feature_indices = self.sample_random_features()
            X_subset = X_bootstrapped[:, feature_indices]

            classifier = DecisionTreeClassifier(max_depth=self.max_depth)
            classifier.fit(X_subset, y_bootstrapped)

            self.classifiers.append({
                'classifier': classifier,
                'feature_indices': feature_indices
            })
        # raise NotImplementedError 
        #################
        
    
    def sample_random_vectors(self):
        """
        Description:
            Select N random vectors with replacement for learning each decision tree.
            You may need to use 'np.random' or 'random' library to select arbitrary vectors.
            Note that N is row numbers of self.X.
        
        Args:
            
        Returns:
            X_bootstrapped: (N, d) numpy array.
            y_bootstrapped: (N,) numpy array.
            
        """
        ### CODE HERE ###
        indices = np.random.choice(self.X.shape[0], size=self.X.shape[0], replace=True)
        X_bootstrapped = self.X[indices]
        y_bootstrapped = self.y[indices]
        # raise NotImplementedError
        #################
        return X_bootstrapped, y_bootstrapped
    
    
    def sample_random_features(self):
        """
        Description:
            Select random features for learning each decision tree.
            You may need to use 'np.random' or 'random' library to select arbitrary features.
            Make sure that m is a random integer value, and selected m features are also random. 
            Note that d is column numbers of self.X.
            
            !! NOTICE that seleting random features DO NOT allow replacement !!
        
        Args:
            
        Returns:
            random_incides: (m,) numpy array. m (<=d and >=1) is an integer value.
            
            
        """
        ### CODE HERE ###
        m = np.random.randint(1, self.X.shape[1] + 1)
        random_indices = np.random.choice(self.X.shape[1], size=m, replace=False)
        # raise NotImplementedError
        #################
        return random_indices
        
        
    def predict(self, X):
        """
        Description:
            Predict the target variables (RandomForest's final prediction). Use the results of voting among attribute classifiers. (i.e. use voting method)
            
            The dictionary {key: value} is composed,
                classifier : {'feature_indices': (indices),
                              'classifier' : (DecisionTreeClassifier)}
               
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of RandomForest classifier. Output values are of 1 or -1.
            
        """
        
        ### CODE HERE ###
        predictions = np.array([
            classifier['classifier'].predict(X[:, classifier['feature_indices']])
            for classifier in self.classifiers
        ])
        pred = sign(np.sum(predictions, axis=0))
        # raise NotImplementedError
        #################
        return pred
    
    
    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of simple randomforest classifier's decision.
            
        """
        ### CODE HERE ###
        logits = np.zeros((X.shape[0]))
        for classifier in self.classifiers:
            feature_indices = classifier['feature_indices']
            clf = classifier['classifier']
            X_sub = X[:, feature_indices]
            logits += clf.predict(X_sub) # prediction 계산 후 누적
            
        proba_pos = 1 / (1 + np.exp(-logits))
        proba_neg = 1 - proba_pos 
        proba = np.vstack((proba_neg, proba_pos)).T
        # raise NotImplementedError
        #################
        return proba

        
        
def compute_each_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies of each classifier.
        
        Args:
            classifier_list: list of dictionary. RandomForest classifiers with feature indices.
            X_train: (N, d) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', d) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
            
    """
    acc_train = []
    acc_test = []

    for i in range(len(classifier_list)):
    
        ### CODE HERE ###
        classifier = classifier_list[i]
        feature_indices = classifier['feature_indices']
        clf = classifier['classifier']

        # Predict on training data
        y_train_pred = clf.predict(X_train[:, feature_indices])
        train_accuracy = np.sum(y_train_pred == y_train) / len(y_train)
        acc_train.append(train_accuracy)

        # Predict on test data
        y_test_pred = clf.predict(X_test[:, feature_indices])
        test_accuracy = np.sum(y_test_pred == y_test) / len(y_test)
        acc_test.append(test_accuracy)
        # raise NotImplementedError
        #################
            
    return acc_train, acc_test        
        
        
    
class AdaBoost:
    """AdaBoost class"""
    
    def __init__(self, num_estimators):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int. The number of decision stumps
                error_history: list. List of weighted error history.
                classifiers: list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[0]) is the dictionary denoted as classifier.
                             The classifier has key 'coefficient' and 'classifier'. The values are the coefficient 
                             for that classifier and the Decsion stump classifier.
        
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.error_history = []
        self.classifiers = []
        
    
    def fit(self, X, y):
        """
        Description:
            Build classifiers from the training set.
               
        Args:
            X: (N, d) numpy array. Training/testing samples.
            y: (N,) numpy array. Ground truth classes.
            
        Returns:
            
            
        """
        self.X = X
        self.y = y
        # initialize the data weight
        ### CODE HERE ###
        self.data_weight = np.ones(y.shape) / y.shape[0] #len(y)
        # print(f'Weight: {self.data_weight}\n')
        # raise NotImplementedError
        #################
        assert self.data_weight.shape == self.y.shape
        self.boosting()
        
    
    def boosting(self):
        """
        Description:
            Boosting - Build adaboost classifier. Follow the procedures described in .ipynb file.
            Use 'DecisionTreeClassifier(max_depth=1)' as a decision stump.
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###
        for i in range(self.num_estimator):
            if i == 0:
                # 첫 번째 iteration에서는 1/N으로 학습
                X_sampled = self.X
                y_sampled = self.y
            else:
                # 이후 iteration에서는 weighted sampling
                indices = np.random.choice(len(self.X), size=len(self.X), replace=True, p=self.data_weight)
                X_sampled, y_sampled = self.X[indices], self.y[indices]
            # Train a decision stump using weighted sampling with replacement
            classifier = DecisionTreeClassifier(max_depth=1)
            classifier.fit(X_sampled, y_sampled)

            # Predict on the training data
            pred = classifier.predict(self.X)
            # print(f'Prediction: {pred}\n')

            # Calculate the weighted error rate
            error = np.sum(self.data_weight * (pred != self.y)) / np.sum(self.data_weight)

            # Compute the classifier coefficient
            coefficient = self.compute_classifier_coefficient(error)
            # coefficient = 0.5 * np.log((1 - error) / error)
            self.error_history.append(error)

            # Save the weak classifier and its coefficient
            self.classifiers.append({
                'classifier': classifier,
                'coefficient': coefficient
            })

            # Update weights
            self.data_weight = self.update_weight(pred, coefficient)    # self.data_weight = self.data_weight * np.exp(-coefficient * self.y * pred)
            
            # Normalize weights
            self.data_weight = self.normalize_weight()                  # self.data_weight = self.data_weight / np.sum(self.data_weight)
        # raise NotImplementedError
        #################
    
    
    def compute_classifier_coefficient(self, weighted_error):
        """
        Description:
            Compute the coefficient for classifier
        
        Args:
            weighted_error: numpy float. Weighted error for the classifier.
            
        Returns:
            coefficient: numpy float. Coefficient for classifier.
            
        """
        ### CODE HERE ###
        coefficient = 0.5 * np.log((1 - weighted_error) / (weighted_error)) #  + 1e-10
        # raise NotImplementedError
        #################
        return coefficient
        
        
    def update_weight(self, pred, coefficient):
        """
        Description:
            Update the data weight. 
        
        Args:
            pred: (N, ) numpy array. Prediction of the weak classifier in one step.
            coefficient: numpy float. Coefficient for classifier.
            
        Returns:
            weight: (N, ) numpy array. Updated data weight.
            
        """
        ### CODE HERE ###
        weight = self.data_weight * np.exp(-coefficient * self.y * pred)
        # raise NotImplementedError
        #################
        return weight
        
        
    def normalize_weight(self):
        """
        Description:
            Normalize the data weight
        
        Args:
            
        Returns:
            weight: (N, ) numpy array. Normalized data weight.
            
        """
        ### CODE HERE ###
        weight = self.data_weight / np.sum(self.data_weight)
        # raise NotImplementedError
        #################
        return weight
    
    
    def predict(self, X):
        """
        Description:
            Predict the target variables (Adaboosts' final prediction). Use the attribute classifiers.
            
            Note that item of classifiers list should be a dictionary like below
                self.classfiers[0] : classifier,  (dict)
                
            The dictionary {key: value} is composed,
                classifier : {'coefficient': (coefficient value),
                              'classifier' : (decision stump classifier)}
        
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of adaboost classifier. Output values are of 1 or -1.
            
        """
        ### CODE HERE ###
        predictions = np.zeros(X.shape[0])
        
        for classifier in self.classifiers:
            coefficient = classifier['coefficient']
            clf = classifier['classifier']
            predictions += coefficient * clf.predict(X)
        pred = sign(predictions)
        # print(f'Predictions: {predictions}\n')
        # print(f'Prediction: {pred}\n')
        # raise NotImplementedError
        #################
        return pred
    
    
    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class using sigmoid function. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, d) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of adaboost classifier's decision.
            
        """
        ### CODE HERE ###
        logits = np.zeros(X.shape[0])
        for classifier in self.classifiers:
            coefficient = classifier['coefficient']
            clf = classifier['classifier']
            logits += coefficient * clf.predict(X)
        proba_pos = 1 / (1 + np.exp(-logits))
        proba_neg = 1 - proba_pos
        proba = np.vstack((proba_neg, proba_pos)).T
        # print(proba)
        # raise NotImplementedError
        #################
        return proba
        
        
    
def compute_staged_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies over stages.
        
        Args:
            classifier_list: list of dictionary. Adaboost classifiers with coefficients.
            X_train: (N, d) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', d) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
                i.e, acc_train[40] =  $\hat{\mathbf{y}}=\text{sign} \left( \sum_{t=1}^{40} \hat{w_t} f_t(\mathbf{x}) \right)$
            
    """
    acc_train = []
    acc_test = []

    for i in range(1, len(classifier_list) + 1):
    
        ### CODE HERE ###
        # Compute weighted sum of predictions for classifiers up to stage i
        train_sum = np.zeros(X_train.shape[0])
        test_sum = np.zeros(X_test.shape[0])

        for classifier in classifier_list[:i]:
            coefficient = classifier['coefficient']
            clf = classifier['classifier']

            train_sum += coefficient * clf.predict(X_train)
            test_sum += coefficient * clf.predict(X_test)

        # Compute final predictions by taking the sign
        y_train_pred = sign(train_sum)
        y_test_pred = sign(test_sum)

        # Calculate accuracy
        acc_train.append(np.sum(y_train_pred == y_train) / len(y_train))
        acc_test.append(np.sum(y_test_pred == y_test) / len(y_test))
        # raise NotImplementedError
        #################
            
    return acc_train, acc_test
    
    

def get_precision(y_pred, y_true):
    """
        Description:
            Compute the precision of the classifier.
        
        Args:
            y_pred: (N, ) numpy array. Prediction of the classifier.
            y_true: (N, ) numpy array. Ground truth classes.
        Returns:
            precision: float. Precision of the classifier.
            
    """
    ### CODE HERE ###
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positive = np.sum(y_pred == 1)
    precision = true_positive / (predicted_positive)
    # raise NotImplementedError
    #################
    return precision

def get_recall(y_pred, y_true):
    """
        Description:
            Compute the recall of the classifier.
        
        Args:
            y_pred: (N, ) numpy array. Prediction of the classifier.
            y_true: (N, ) numpy array. Ground truth classes.
        Returns:
            recall: float. Recall of the classifier.
            
    """
    ### CODE HERE ###
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    actual_positive = np.sum(y_true == 1)
    recall = true_positive / (actual_positive)
    # raise NotImplementedError
    #################
    return recall

def get_f1_score(y_pred, y_true):
    """
        Description:
            Compute the F1 score of the classifier.
        
        Args:
            y_pred: (N, ) numpy array. Prediction of the classifier.
            y_true: (N, ) numpy array. Ground truth classes.
        Returns:
            f1_score: float. F1 score of the classifier.
            
    """
    ### CODE HERE ###
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # raise NotImplementedError
    #################
    return f1_score


def plot_precision_recall_curve(y_proba, y_true):
    """
    Description:
        Compute and plot the precision-recall curve of the classifier.

    Args:
        y_proba: (N, 2) numpy array. Probability scores for each class (column 0 for class -1, column 1 for class 1).
        y_true: (N,) numpy array. Ground truth classes.

    Returns:
        precision_list: list. Precision of the classifier at each threshold.
        recall_list: list. Recall of the classifier at each threshold.
    """

    # Extract probabilities for the positive class (class 1)
    prob_pos = y_proba[:, 1]
    
    ### CODE HERE ###
    thresholds = np.sort(np.unique(prob_pos))
    precision_list = []
    recall_list = []

    for threshold in thresholds:
        y_pred = (prob_pos >= threshold).astype(int) * 2 - 1  # Convert to -1 or 1
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == -1))
        fn = np.sum((y_pred == -1) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
    # raise NotImplementedError
    #################
    return precision_list, recall_list

