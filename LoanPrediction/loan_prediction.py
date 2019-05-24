# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


"""
Combine train and test data 
Add one more column to the combined data to identify train and test data.
"""
def combine_datasets(train, test):
    dataset_train['source'] = 'train'
    dataset_test['source'] = 'test'
    combined_data = pd.concat([dataset_train,dataset_test], ignore_index=True)
    return combined_data;

"""
Print the summary of the given dataframe.
This method prints following summary details of the dataframe:
    1: shape
    2: Null values per column
"""
def print_dataset_summary(dataset):
    print("\n\n<------ Summary for data --->")
    print("\tShape of train data",dataset.shape)

    print("\tPrinting null values per column :")    
    print(combined_data.apply(lambda x: sum(x.isnull())))    

"""
Calculate mean square error.
"""
def calculate_mse(Y_pred, Y_actual):
    # calculate MSE
    mse = np.mean((Y_pred-Y_actual)**2)
    return mse
    

def plot_residual_graph(Y_pred, Y_actual):
    # Plot the graph and check the data pattern.
    # residual plot
    x_plot = plt.scatter(Y_pred, (Y_pred - Y_actual), c='b')
    plt.hlines(y=0, xmin= -1000, xmax=5000)
    plt.title('Residual plot')

def unique_val_categorical_col(categorical_columns):
    for column in categorical_columns:
        print("<--------- Column name: ",column," ----------->")
        print(combined_data[column].value_counts())
        

def plot_categorical_features(df, categorical_columns):
   print("Size of list: ",len(categorical_columns))
   for column in categorical_columns:
       df[column].value_counts().plot(kind="bar",title=column)
       plt.show()

# Importing the dataset
dataset_train = pd.read_csv('data/train_av.csv')
dataset_test = pd.read_csv('data/test_av.csv')
    
combined_data = combine_datasets(dataset_train, dataset_test)
print_dataset_summary(dataset_train)    
print_dataset_summary(dataset_test)    
print_dataset_summary(combined_data)    

#get categorical column
categorical_column = [x for x in combined_data.dtypes.index if combined_data.dtypes[x] == 'object']
print(categorical_column)

head = combined_data.head()

unique_val_categorical_col(categorical_column)
plot_categorical_features(combined_data, categorical_column)

# Missing value imputation.

combined_data["Gender"].fillna("Male",inplace=True)
combined_data["Married"].fillna("Yes",inplace=True)
combined_data["Credit_History"].fillna(1,inplace=True)
combined_data["Credit_History"].fillna(1,inplace=True)
combined_data['Dependents'] = combined_data['Dependents'].map({'3+':'3', '1' : '1', '0' : '0', '2' : '2'})


combined_data['Dependents'].fillna
combined_data["Credit_History"].value_counts()
combined_data["Dependents"].isnull().sum()


















# step 10
#       separate train and test data 
#       Remove Item_Outlet_Sales from test data

train = combined_data.loc[combined_data['source']=="train"]
test = combined_data.loc[combined_data['source']=="test"]

train.drop(['source'],axis=1,inplace=True)

# target variable name.
target = '' 
IDcol = []
submissionCols = []

predictors = [x for x in train.columns if x not in [target]+IDcol]

X = train[predictors]
Y = train[target]


#split the data into train and test data. Cross validation.
X_train, X_test, Y_train , Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

'''
# Uncomment this code to use LinearRegression.
# Predict Values using LinearRegression Model
linear_regression = LinearRegression()
linear_regression.fit(X_train,Y_train)
Y_predict = linear_regression.predict(X_test)

print(calculate_mse(Y_predict, Y_test))

# calculate R-Squared Adjusted.
score = linear_regression.score(X_test,Y_test)
print(score)

# plot the graph.
plot_residual_graph(Y_predict, Y_test)

test[target] = linear_regression.predict(test[predictors])

# Linear model processing end here...
'''

'''
# Uncomment this code for using Polynomial Regression.

# Predict the values using Polynomial regression.
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_train_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_train_poly, Y_train)
polynomial_regression = LinearRegression()
polynomial_regression.fit(X_train_poly, Y_train)

X_test_poly = poly_reg.fit_transform(X_test)
Y_predict_poly = polynomial_regression.predict(X_test_poly)

# calculate MSE
print(calculate_mse(Y_predict_poly, Y_test))

# calculate R-Squared Adjusted.
score = polynomial_regression.score(X_test_poly,Y_test)
print(score)

plot_residual_graph(Y_predict_poly, Y_test)

# predict for test data.
test[target] = polynomial_regression.predict(poly_reg.fit_transform(test[predictors]))

# checking magnitude of coefficient.
coeff = polynomial_regression.coef_
print(max(coeff))
print(min(coeff))
print(sum(coeff)/len(coeff))

# Evaluating model performance using k-fold cross validation.
poly_regression_accuracies = cross_val_score(estimator = polynomial_regression, X = X_train_poly, 
                             y = Y_train, cv = 10)

print(poly_regression_accuracies.mean())
print(poly_regression_accuracies.std())

# Polynomical regression processing ends here.
'''

'''
# Uncomment this code to use RidgeRegression.

# Training the model using Ridge Regression.
ridge_regression = Ridge(normalize = True)
ridge_regression.fit(X_train, Y_train)
Y_pred_ridge = ridge_regression.predict(X_test)
print(calculate_mse(Y_pred_ridge, Y_test))
ridge_regression.score(X_test, Y_test)

ridge_coeff = ridge_regression.coef_
print(max(ridge_coeff))
print(min(ridge_coeff))
print(sum(ridge_coeff)/len(ridge_coeff))

# Evaluting model performance using k-folde cross validation.
ridge_accuracies = cross_val_score(estimator = ridge_regression, X = X_train, 
                             y = Y_train, cv = 10)
print(ridge_accuracies.mean())
print(ridge_accuracies.std())
# Applying Grid Search to find the best model and the best parameters
alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
fit_interceptOptions = ([True, False])
solverOptions = (['svd', 'cholesky', 'sparse_cg', 'sag'])

parameters = dict(alpha=alphas, fit_intercept=fit_interceptOptions, solver=solverOptions)

grid_search = GridSearchCV(estimator = ridge_regression,
                           param_grid = parameters)

grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

test[target] = grid_search.predict(test[predictors])
# Ridge regression ends here..
'''

# Method to evaludate model performance
def evaluate_model_performance(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print("Mean Square Error: ",calculate_mse(y_predict, y_test))
    accurracy = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
    print("Accurracy Mean: ", accurracy.mean())
    print("Accurracy Std : ", accurracy.std())

# Training model using XGBoost.
xgbRegressor = XGBRegressor()
evaluate_model_performance(xgbRegressor, X_train, Y_train, X_test, Y_test)
'''
Performance of above model :
    Mean Square Error:  1186173.7950376957
    Accurracy Mean:  0.594592170829
    Accurracy Std :  0.019906704365
'''

test[target] = xgbRegressor.predict(test[predictors])

submission = test[IDcol]
submission[target] = test[target]
submission.to_csv(DIR+"/ridge_regression.csv", index=False)