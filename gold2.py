import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model;
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
	
def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model
 
 
def model_accuracy(trained_model, features, targets):
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score

dataset=pd.read_csv("GOLD_.csv")


dataset["Gold"].fillna(dataset["Gold"].mean(),inplace=True)
dataset["oil"].fillna(dataset["oil"].mean(),inplace=True)
dataset["Heng_sang"].fillna(dataset["Heng_sang"].mean(),inplace=True)
dataset["USD"].fillna(dataset["USD"].mean(),inplace=True)
dataset["N225"].fillna(dataset["N225"].mean(),inplace=True)
#for name in ["Silver","Copper","oil","Heng_sang","USD","N225","GBP_USD","Gold"]:
	#print("LEVEL:",dataset[name].dropna().mean())

#print(list(dataset.columns.values))
#dataset.info()

training_features=["Silver","Copper","oil","Heng_sang","USD","N225","GBP_USD"]
target=["digital"]
trainX=dataset[["Silver","Copper","oil","Heng_sang","USD","N225","GBP_USD"]].values
trainY=dataset["digital"].values

trainX, testX, trainY, testY = train_test_split(dataset[training_features], dataset[target], train_size=0.8)
'''
print ("train_x size :: ", trainX.shape)
print ("train_y size :: ", trainY.shape)
print ("test_x size :: ", testX.shape)
print ("test_y size :: ", testY.shape)
'''
#print ("edu_target_frequencies :: ", feature_target_frequency_relation(dataset, [training_features[3], target]))
 
#for feature in training_feat    ures:
	#feature_target_frequencies = feature_target_frequency_relation(dataset, [feature, target])
	#feature_target_histogram(feature_target_frequencies, feature)
 
# Training Logistic regression model
for i in range(1):
	trained_logistic_regression_model = train_logistic_regression(trainX, trainY)
	train_accuracy = model_accuracy(trained_logistic_regression_model, trainX, trainY)
# Testing the logistic regression model
	test_accuracy = model_accuracy(trained_logistic_regression_model, testX, testY)
	#print ("Train Accuracy :: ", train_accuracy)
	print ("Test Accuracy :: ", test_accuracy)
 








'''
clf=LogisticRegression();
clf_=clf.fit(trainX,trainY);

train_accuracy = model_accuracy(clf_, trainX, trainY)
 
print ("Train Accuracy :: ", train_accuracy)







temp=[0]*len(trainY)
for i in range(len(trainY)):
	if trainY[i]>=trainY[i-1]:
		temp[i]=1
	else:
		temp[i]=0
		
for i in range(len(trainY)):
	#print(temp[i],trainY[i])
	trainY[i]=temp[i]
#for i in range(len(trainY)):
	#print(trainY[i])
'''