import pandas as pd
import numpy as np
import seaborn as sns
import zipfile
import io
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import metrics

''' Error function '''
def MSE(theta, X, y):
    m = len(X)
    predict = X.dot(theta)
    return (1/(2*m)) * np.sum(np.square(predict - y))

''' Update Weight '''
def updateWeight(theta, X, y, lr, v):
    m = len(X)
    predict = X.dot(theta)
    error = predict - y
    
    gradient = error.dot(X)/m
    rate = lr/ (np.sqrt(v) + .00000001) # add small number so it never divides by 0
    return theta - rate * gradient

''' Learning Rate '''
def RMSprop(v, theta, X, y):
    m = len(X)
    predict = X.dot(theta)
    error = predict - y
    gradient = error.dot(X)/m
    
    return v * .9 + gradient * gradient * .1    

''' Main '''
# Get datasets
url = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip')
zf = zipfile.ZipFile(io.BytesIO(url.content))
csv = zf.open('student-mat.csv')
data = pd.read_csv(csv, sep=';')

pd.set_option('display.max_columns', None)

# check for na values
#print(data.isnull().sum())

# hot encode some fields
internet = pd.get_dummies(data['internet'], drop_first=True)
internet.rename(columns={'yes':'internet'}, inplace=True)

Pstatus = pd.get_dummies(data['Pstatus'], drop_first=True)
Pstatus.rename(columns={'T':'Pstatus'}, inplace=True)

schoolsup = pd.get_dummies(data['schoolsup'], drop_first=True)
schoolsup.rename(columns={'yes':'schoolsup'}, inplace=True)

higherEd = pd.get_dummies(data['higher'], drop_first=True)
higherEd.rename(columns={'yes':'higher'}, inplace=True)

extraClasses = pd.get_dummies(data['paid'], drop_first=True)
extraClasses.rename(columns={'yes':'extraClass'}, inplace=True)

famsup = pd.get_dummies(data['famsup'], drop_first=True)
famsup.rename(columns={'yes':'famsup'}, inplace=True)

activities = pd.get_dummies(data['activities'], drop_first=True)
activities.rename(columns={'yes':'activities'}, inplace=True)

romantic = pd.get_dummies(data['romantic'], drop_first=True)
romantic.rename(columns={'yes':'romantic'}, inplace=True)

data = data.drop(columns =['internet', 'Pstatus', 'schoolsup', 'higher', 'paid', 'famsup', 'activities', 'romantic'])

data = pd.concat([internet, Pstatus, higherEd, schoolsup, extraClasses, famsup, activities, romantic, data], axis=1)

# remove outliers
data = data[data['G3'] > 0]

# add intercept for linear regression
data['intercept'] = 1

# check head
#print(data.head())

# get highest correlated fields to G3
corr = data.corr().abs()['G3']
#print(corr.sort_values())

# graphs
'''
plt.hist(data['G3'], bins=10)
plt.xlabel('Grade')
plt.ylabel('Students')
plt.show()
'''

# prepare X and y
X = data[['G1', 'G2', 'Medu', 'Fedu', 'schoolsup', 'absences', 'goout', 'Dalc', 'extraClass', 'age', 'Pstatus', 'freetime', 'intercept']]
y = data['G3']

# make lmplot
#sns.pairplot(data)
#plt.show()

# make a heat map
#plt.figure(figsize=(20, 20))
#sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm')
#plt.show()

# min-max normalize
X = (X - X.min()) / (X.max()-X.min())
X['intercept'] = 1

# train
epoch = 1000
errHist = []
coefHist = []
lr = .02
theta = np.ones(len(X.columns))
v = 0

# split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state = 69)
for i in range(epoch):
    # rmsprop
    v = RMSprop(v, theta, X_train, y_train)
    
    # gradient descent
    theta = updateWeight(theta, X_train, y_train, lr, v)
    
    # end if error is low
    error = MSE(theta, X_train, y_train)
    if error < .5:
        i = epoch + 1
    
    # keep track of info for graphs
    errHist.append(error)
    coefHist.append(theta)
    
predictions = X_test.dot(theta)

print('R2:\t', metrics.r2_score(y_test, predictions))
print('MAE:\t', metrics.mean_absolute_error(y_test, predictions))
print('MSE:\t', metrics.mean_squared_error(y_test, predictions))
print('RMSE:\t', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print()
print([theta])

# error vs iterations
plt.scatter(range(epoch), errHist)
plt.xlabel('Iterations')
plt.ylabel('Error (MSE/2)')
plt.show()

# scatter plot actual vs prediction
plt.scatter(y_test, predictions)
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.show()

# dist plot percentage vs error
sns.distplot((y_test-predictions), bins=50)
plt.show()

# weights vs iterations
plt.plot(range(epoch), coefHist)
plt.xlabel('Iterations')
plt.ylabel('Weights')
plt.legend(X.columns)
plt.show()