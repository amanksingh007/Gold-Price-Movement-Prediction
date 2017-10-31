
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model;
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
#from sklearn import metrices

def modify(dataset):
	temp=[]
	temp.append(1);
	print("kkkkkkkk")
	target=["Gold"]
	for i in range(1,len(target)):
		if target[i]-target[i-1] >0:
			temp.append(1)
		else:
			temp.append(0)
	for i in range(len(temp)):
		print (temp[i])
	dataset['GOld']=temp
	#dataset.add(axis='columns', level=GD, fill_value=temp)
	return dataset;

def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model
 
 
def model_accuracy(trained_model, features, targets):
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score



dataset=pd.read_csv("GOLD_FINAL.csv")
#print(list(df.columns.values))
dataset["Gold"].fillna((dataset["Gold"].mean()), inplace=True)
#print(dataset.info())
#print(dataset.shape)
#print(dataset["Gold"][5])
dataset=modify(dataset)
print(list(dataset.columns.values))
print(dataset.head(10))


training_features=['Silver','Copper']
target=['Gold']

train_x, test_x, train_y, test_y = train_test_split(dataset[training_features], dataset[target], train_size=0.7)
print ("train_x size :: ", train_x.shape)
print ("train_y size :: ", train_y.shape)
print ("test_x size :: ", test_x.shape)
print ("test_y size :: ", test_y.shape)


# Training Logistic regression model
trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    
train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)
 
    # Testing the logistic regression model
test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)
 
print ("Train Accuracy :: ", train_accuracy)
print ("Test Accuracy :: ", test_accuracy)
