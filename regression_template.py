# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

'''
Directory where the data file is present.
'''
DIR = 'D:\Github\Data\BigMartSales-III'

# Importing the dataset
dataset_train = pd.read_csv(DIR+'/Train_UWu5bXk.csv')
dataset_test = pd.read_csv(DIR+'/Test_u94Q5KV.csv')


# step1 combine train and test data 
#       Add one more column to the combined data to identify train and test data.
#       Perform all the cleaning and data processing operation on combined data.
dataset_train['source'] = 'train'
dataset_test['source'] = 'test'
combined_data = pd.concat([dataset_train,dataset_test], ignore_index=True)

# Get the shape of the data.
combined_data.shape
combined_data.apply(lambda x: sum(x.isnull()))

# Get categorical column
categorical_column = [x for x in combined_data.dtypes.index if combined_data.dtypes[x] == 'object']
print(categorical_column)

'''
    Apply missing value imputation, feature engineering and explaratory data analysis here.
'''


'''
    Separate train and test data 
    Remove Item_Outlet_Sales from test data
'''
train = combined_data.loc[combined_data['source']=="train"]
test = combined_data.loc[combined_data['source']=="test"]

'''
    Find out the target column, IDColumns, Submission columns and predictors.
    Submission columns are used for writing to output CSV file.
    Predictors are the list of features which are used for training model.
'''
target = ''
IDcol = []
submissionCols = []

predictors = [x for x in train.columns if x not in [target]+IDcol]

X = train[predictors]
Y = train[target]


# Method to calculate Mean Square Error.
def calculate_mse(Y_pred, Y_actual):
    # calculate MSE
    mse = np.mean((Y_pred-Y_actual)**2)
    return mse
    
# Plot the residual graph
def plot_residual_graph(Y_pred, Y_actual):
    # Plot the graph and check the data pattern.
    # residual plot
    x_plot = plt.scatter(Y_pred, (Y_pred - Y_actual), c='b')
    plt.hlines(y=0, xmin= -1000, xmax=5000)
    plt.title('Residual plot')

# split the data into train and test data. Cross validation.
X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

'''
    Method to evaludate model performance.
    It prints the accuracy and Standard deviation based on cross validation.
'''
def evaluate_model_performance(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print("Mean Square Error: ",calculate_mse(y_predict, y_test))
    accurracy = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
    print("Accurracy Mean: ", accurracy.mean())
    print("Accurracy Std : ", accurracy.std())
    
