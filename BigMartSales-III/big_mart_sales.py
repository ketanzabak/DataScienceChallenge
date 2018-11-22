# Multiple Linear Regression
# Big Mart Sales III 

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

combined_data.shape
combined_data.apply(lambda x: sum(x.isnull()))

#get categorical column
categorical_column = [x for x in combined_data.dtypes.index if combined_data.dtypes[x] == 'object']
print(categorical_column)
categorical_column = [x for x in categorical_column if x not in ['Item_Identifier','Outlet_Identifier','source']]


# step 2
#       Calculate mean item weight based on the item_type and replace missing values by mean.
item_avg_weight = combined_data.pivot_table(values='Item_Weight', index='Item_Identifier', dropna = True)    
miss_bool = combined_data['Item_Weight'].isnull() 
combined_data.loc[miss_bool,'Item_Weight'] = combined_data.loc[miss_bool,'Item_Identifier'].apply(lambda x : item_avg_weight.loc[x])

# step 3
#       For Column Item_fat_Content LF = Low Fat & reg = regular (make all the rows consistent) 
combined_data['Item_Fat_Content'] = combined_data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat': 'Low Fat'})

# step 4
#       Item_visibility cannot be zero. Change it.
#       Consider following parameters while calculating visibility
#           - Item_Identifier
item_visibility_mean = combined_data.pivot_table(values='Item_Visibility', index='Item_Identifier', dropna = True)    
miss_bool = (combined_data['Item_Visibility'] == 0)
combined_data.loc[miss_bool,'Item_Visibility'] = combined_data.loc[miss_bool,'Item_Identifier'].apply(lambda x : item_visibility_mean.loc[x])
sum(combined_data['Item_Visibility'] == 0)

combined_data['Item_Visibility_MeanRatio'] = combined_data.apply(lambda x: x['Item_Visibility']/item_visibility_mean.loc[x['Item_Identifier']], axis=1)

# step 5
#       For Year_Of_Establishment add new variable NoOfYears. And drop Year_Of_Establishment.
combined_data['Outlet_Years'] = 2018 - combined_data['Outlet_Establishment_Year']

# step 6 
#       Find missing values for outlet size by mode.
outlet_size_mode = combined_data.dropna(subset=["Outlet_Size"]).pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
miss_bool = combined_data['Outlet_Size'].isnull() 
sum(miss_bool)
combined_data.loc[miss_bool,'Outlet_Size'] = combined_data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
sum(combined_data['Outlet_Size'].isnull())

# step 7 
#       Item Type handling.
combined_data['Item_Type_Combined'] = combined_data['Item_Identifier'].apply(lambda x: x[0:2])
combined_data['Item_Type_Combined'] = combined_data['Item_Type_Combined'].map({'FD':'Food', 'DR' : 'Drinks', 'NC' : 'Non Consumable'})

# step 8 
#       Item fat content are not required for the non-consumable food
non_consumable_bool = combined_data['Item_Type_Combined']=="Non Consumable"
combined_data.loc[non_consumable_bool,'Item_Fat_Content'] = 'Non Edible'
combined_data['Item_Fat_Content'].value_counts()

# Remove unnecessart columns
combined_data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
# label encoder
le = LabelEncoder()

# step 9
#       label encoder for Item_Fat_Content (Non-Editable<Low < Regular) 
#                        Outlet Size (small < medium)
combined_data['Item_Fat_Content'].value_counts()
combined_data['Item_Fat_Content_Encoded'] = combined_data['Item_Fat_Content'].map({'Non Edible':1, 'Low Fat' : 2, 'Regular' : 3})

combined_data['Outlet_Size'].value_counts()
combined_data['Outlet_Size_Encoded'] = combined_data['Outlet_Size'].map({'Small':1, 'Medium' : 2, 'High' : 3})

combined_data.drop(['Item_Fat_Content','Outlet_Size'],axis=1, inplace = True)
#       One hot encoder for (Outlet location type, Outlet type, ItemTypeCombined)

combined_data = pd.get_dummies(combined_data, columns=['Outlet_Location_Type','Outlet_Type','Item_Type_Combined'])

# step 10
#       separate train and test data 
#       Remove Item_Outlet_Sales from test data

train = combined_data.loc[combined_data['source']=="train"]
test = combined_data.loc[combined_data['source']=="test"]

test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
submissionCols = ['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']

predictors = [x for x in train.columns if x not in [target]+IDcol]

X = train[predictors]
Y = train[target]

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


submission = test[IDcol]
submission[target] = test[target]
submission.to_csv(DIR+"/ridge_regression.csv", index=False)
