# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv(
'https://raw.githubusercontent.com/ssfaruque/HD_Computing/master/chemometrics/datasets/DTreeSets/'+
#
#select only one from below
#'noisySets/DT_noisy_005_'+
#'noisySets/DT_noisy_01_'+
#'noisySets/DT_noisy_015_'+
#'noisySets/DT_noisy_02_'+
#'noisySets/DT_noisy_03_'+
#'noisySets/DT_multiplicative_075_'
#'noisySets/DT_multiplicative_090_'
#'noisySets/DT_multiplicative_110_'
#'noisySets/DT_multiplicative_125_'
#'noisySets/DT_additive_025_'
#'noisySets/DT_additive_050_'
#'noisySets/DT_additive_100_'
#
#select only one from below
#'DNA_Anodisc.csv',
#'DNA_ECOLI.csv',
'DNA_inLiquidDNA.csv',
#'Full_Set.csv',
#'Yeast_inLiquidHK.csv',
#'Yeast_inLiquidLive.csv',
    sep= ',', header = None)

    balance_data = np.array(balance_data)
    np.random.shuffle(balance_data)
    threshold = 0.0875
    for i in range(len(balance_data)):
        for j in range(1, len(balance_data[0])):
         if float(balance_data[i][j]) < threshold:
             balance_data[i][j] = 0
    return balance_data

def retrieve_indices_of_label(balance_data, label):
    indices = []
    for i in range(0, len(balance_data[:, 0])):
        if int(balance_data[i][0]) == label:
            indices.append(i)
    return indices

def update_train_test_sets(balance_data):
    num_files_per_category = 5
    ppm0  = retrieve_indices_of_label(balance_data, 0)
    ppm2  = retrieve_indices_of_label(balance_data, 2)
    ppm5  = retrieve_indices_of_label(balance_data, 5)
    ppm10 = retrieve_indices_of_label(balance_data, 10)
    ppm15 = retrieve_indices_of_label(balance_data, 15)

    ppm0_indices  = ppm0[ 0 : num_files_per_category]
    ppm2_indices  = ppm2[ 0 : num_files_per_category]
    ppm5_indices  = ppm5[ 0 : num_files_per_category]
    ppm10_indices = ppm10[0 : num_files_per_category]
    ppm15_indices = ppm15[0 : num_files_per_category]

    ppm0_samples  = np.copy(balance_data[ppm0_indices])
    ppm2_samples  = np.copy(balance_data[ppm2_indices])
    ppm5_samples  = np.copy(balance_data[ppm5_indices])
    ppm10_samples = np.copy(balance_data[ppm10_indices])
    ppm15_samples = np.copy(balance_data[ppm15_indices])

    trainset = np.concatenate((ppm0_samples, ppm2_samples, ppm5_samples, ppm10_samples, ppm15_samples))
    testset = np.delete(balance_data, (ppm0_indices + ppm2_indices + ppm5_indices + ppm10_indices + ppm15_indices), axis=0)

    return trainset, testset

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

    predicted_gini_accuracy = []
    predicted_gini_f1 = []
    predicted_entropy_accuracy = []
    predicted_entropy_f1 = []

    for i in range(10):
        # Building Phase
        data = importdata()
        #X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        trainset, testset = update_train_test_sets(data)
        clf_gini = train_using_gini(trainset[:, 1:1608], testset[:, 1:1608], trainset[:, 0])
        clf_entropy = train_using_entropy(trainset[:, 1:1608], testset[:,1:1608], trainset[:, 0])

        # Operational Phase
        print("Results Using Gini Index:")

        # Prediction using gini
        y_pred_gini = prediction(testset[:, 1:1608], clf_gini)
        cal_accuracy(testset[:, 0], y_pred_gini)
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for i in range(0,len(testset[:,0])):
            predicted = y_pred_gini[i]
            label = testset[i][0]

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
        print("F1-score: {}\n".format(round(f1_score,2)))
        predicted_gini_accuracy.append(accuracy_score(testset[:, 0], y_pred_gini)*100)
        predicted_gini_f1.append(f1_score)

        print("Results Using Entropy:")
        # Prediction using entropy
        y_pred_entropy = prediction(testset[:, 1:1608], clf_entropy)
        cal_accuracy(testset[:, 0], y_pred_entropy)
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for i in range(0,len(testset[:,0])):
            predicted = y_pred_entropy[i]
            label = testset[i][0]

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
        F1_Score = 2 * TP / (2 * TP + FP + FN)
        print("F1-score: {}\n".format(round(F1_Score,2)))
        predicted_entropy_accuracy.append(accuracy_score(testset[:, 0], y_pred_entropy)*100)
        predicted_entropy_f1.append(F1_Score)

    Average_Gini_Acc = statistics.mean(predicted_gini_accuracy)
    Average_Gini_F1 = statistics.mean(predicted_gini_f1)
    Average_Entropy_Acc = statistics.mean(predicted_entropy_accuracy)
    Average_Entropy_F1 = statistics.mean(predicted_entropy_f1)
    print("Average Gini Accuracy: {}".format(round(Average_Gini_Acc, 2)))
    print("Average Gini F1: {}\n".format(round(Average_Gini_F1, 2)))
    print("Average Entropy Accuracy: {}".format(round(Average_Entropy_Acc, 2)))
    print("Average Entropy F1: {}\n".format(round(Average_Entropy_F1, 2)))

# Calling main function
if __name__=="__main__":
    main()
