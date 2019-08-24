# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# Function importing Dataset
def importdata():
    balance_data = pd.read_csv(
'https://raw.githubusercontent.com/ssfaruque/HD_Computing/master/chemometrics/datasets/DTreeSets/'+
#'noisySets/DT_noisy_01_'+
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
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    # Printing the dataset obseravtions
    print ("Dataset: ",balance_data.head())
    return balance_data

# Function to split the dataset
def splitdataset(balance_data):

    # Seperating the target variable
    X = balance_data.values[:, 1:1868] #min = 1, max = 1868
    Y = balance_data.values[:, 0]

    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 51, shuffle = True, stratify = None)

    return X, Y, X_train, X_test, y_train, y_test

def GradientBoost(X_train, X_test, y_train):

    model = CatBoostClassifier(iterations=10, depth=5)
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

    #Training Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    #Testing Phase
    y_pred = GradientBoost(X_train, X_test, y_train)
    cal_accuracy(y_test, y_pred)


# Calling main function
if __name__=="__main__":
    main()
