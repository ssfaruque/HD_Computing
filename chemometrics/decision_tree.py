# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv(
'https://raw.githubusercontent.com/ssfaruque/HD_Computing/master/chemometrics/datasets/DTreeSets/'+
'noisySets/DT_noisy_005_'+
#'noisySets/DT_noisy_01_'+
#'noisySets/DT_noisy_015_'+
#'noisySets/DT_noisy_02_'+
#'noisySets/DT_noisy_03_'+
#'DNA_Anodisc.csv',
#'DNA_ECOLI.csv',
#'DNA_inLiquidDNA.csv',
#'Full_Set.csv',
#'Yeast_inLiquidHK.csv',
'Yeast_inLiquidLive.csv',
    sep= ',', header = None)

    # Printing the dataswet shape
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    print ("Dataset: ",balance_data.head())
    return balance_data

# Function to split the dataset
def splitdataset(balance_data):

    # Seperating the target variable
    X = balance_data.values[:, 1:1608] #min = 1, max = 1868
    Y = balance_data.values[:, 0]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.25, random_state = 51, shuffle = True, stratify = None)

    return X, Y, X_train, X_test, y_train, y_test

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 51,max_depth=100, min_samples_leaf = 1)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def train_using_entropy(X_train, X_test, y_train):

    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 51,
            max_depth = 100, min_samples_leaf = 1)

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy


# Function to make predictions
def prediction(X_test, clf_object):

    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))

    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

    print("Report : \n", classification_report(y_test, y_pred))

# Driver code
def main():

    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    # Operational Phase
    print("Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i in range(0,len(y_test)):
        predicted = y_pred_gini[i]
        label = y_test[i]

        if predicted == label:
            if predicted == 0 or predicted == 2:
                TN += 1
            else:
                TP += 1
        else:
            if predicted == 0:
                if label == 2:
                    TN += 1
                else:
                    FN += 1
            elif predicted == 2:
                if label == 0:
                    TN += 1
                else:
                    FN += 1
            elif predicted == 5:
                if label == 0 or label == 2:
                    FP += 1
                else:
                    TP += 1
            elif predicted == 10:
                if label == 0 or label == 2:
                    FP += 1
                else:
                    TN += 1
            elif predicted == 15:
                if label == 0 or label == 2:
                    FP += 1
                else:
                    TP += 1
    f1_score = 2 * TP / (2 * TP + FP + FN)

    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
    report = classification_report(y_test, y_pred_entropy, output_dict = True)
    print("Test F1: {}".format(round(f1_score,2)))#report["weighted avg"]["f1-score"]))


# Calling main function
if __name__=="__main__":
    main()
