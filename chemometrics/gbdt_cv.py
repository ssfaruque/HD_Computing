# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
import statistics

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv(
'https://raw.githubusercontent.com/ssfaruque/HD_Computing/master/chemometrics/datasets/DTreeSets/'+
#'noisySets/DT_noisy_005_'+
#'noisySets/DT_noisy_01_'+
#'noisySets/DT_noisy_015_'+
#'noisySets/DT_noisy_02_'+
#'noisySets/DT_noisy_03_'+
#'DNA_Anodisc.csv',
#'DNA_ECOLI.csv',
'DNA_inLiquidDNA.csv',
#'Full_Set.csv',
#'Yeast_inLiquidHK.csv',
#'Yeast_inLiquidLive.csv',
    sep= ',', header = None)

    # Printing the dataswet shape
    #print ("Dataset Length: ", len(balance_data))
    #print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    #print ("Dataset: ",balance_data.head())
    balance_data = np.array(balance_data)
    np.random.shuffle(balance_data)
    threshold = 0.0875
    for i in range(len(balance_data)):
        for j in range(1, len(balance_data[0])):
         if float(balance_data[i][j]) < threshold:
             balance_data[i][j] = 0
    return balance_data

def update_train_test_sets(balance_data, training_indices, testing_indices):
    trainset = balance_data[training_indices]
    testset = balance_data[testing_indices]
    return trainset, testset

# Function to split the dataset
def splitdataset(balance_data):

    # Seperating the target variable
    X = balance_data.values[:, 1:1868] #min = 1, max = 1868
    Y = balance_data.values[:, 0]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 51, shuffle = True, stratify = None)

    return X, Y, X_train, X_test, y_train, y_test

# Function to train Gradient Boosted Decision Tree
def GradientBoost(X_train, X_test, y_train):

    model = CatBoostClassifier(iterations=25, depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):

    #print("Confusion Matrix: ",
    #    confusion_matrix(y_test, y_pred))

    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

    print("Report : \n",
    classification_report(y_test, y_pred))

# Driver code
def main():

    predicted_accuracy = []
    predicted_f1 = []
    num_splits = 2

    for i in range(10):
        print("RUN {}".format(i+1))
        # Building Phase
        # X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        data = importdata()
        kf = KFold(n_splits=num_splits)
        split_num = 1
        accuracies = []
        f1s = []

        for training_indices, testing_indices in kf.split(data):
            print("Split {}/{}".format(split_num,num_splits))
            trainset, testset = update_train_test_sets(data, training_indices, testing_indices)
            #clf_entropy = train_using_entropy(trainset[:, 1:1868], testset[:, 1:1868], trainset[:, 0])

            # Operational Phase
            y_pred = GradientBoost(trainset[:, 1:1868], testset[:, 1:1868], trainset[:, 0])
            #y_pred_entropy = prediction(testset[:, 1:1868], clf_entropy)
            # cal_accuracy(testset[:, 0], y_pred_entropy)
            report = classification_report(testset[:, 0], y_pred, output_dict = True)
            F1 = report["weighted avg"]["f1-score"]
            Accuracy = accuracy_score(testset[:, 0],y_pred)*100
            accuracies.append(Accuracy)
            f1s.append(F1)
            split_num+=1

        Average_Acc = statistics.mean(accuracies)
        Average_F1 = statistics.mean(f1s)
        print("Average Accuracy: {}".format(round(Average_Acc, 2)))
        print("Average F1: {}\n".format(round(Average_F1, 2)))
        predicted_accuracy.append(Average_Acc)
        predicted_f1.append(Average_F1)

    print("\nPredicted Accuracy: {}".format(round(statistics.mean(predicted_accuracy), 2)))
    print("Predicted F1: {}".format(round(statistics.mean(predicted_f1), 2)))

# Calling main function
if __name__=="__main__":
    main()
