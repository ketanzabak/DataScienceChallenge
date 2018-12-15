# Solution for Titanic: Machine Learning from Disaster 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
DIR = 'D:\Github\Data\Titanic'
df_train = pd.read_csv(DIR+'/train.csv');
df_test = pd.read_csv(DIR+'/test.csv');

#combine data for cleaning.
# handle null values.
df_train['source'] = 'train'
df_test['source'] = 'test'
combined_data = pd.concat([df_train,df_test], ignore_index=True)

combined_data.shape
combined_data.apply(lambda x: sum(x.isnull()))

# calculate avg age based on Sex.
avg_age = combined_data.pivot_table(values='Age', index='Sex', dropna = True)    
miss_bool = combined_data['Age'].isnull() 
combined_data.loc[miss_bool,'Age'] = combined_data.loc[miss_bool,'Sex'].apply(lambda x : avg_age.loc[x])
combined_data.apply(lambda x: sum(x.isnull()))

combined_data = combined_data.drop(['Cabin'],axis=1)

combined_data['Embarked'].value_counts()
combined_data['Embarked'].fillna(combined_data['Embarked'].mode()[0], inplace=True)

combined_data['Pclass'].value_counts()

avg_price_per_class = combined_data.pivot_table(values='Fare', index='Pclass', dropna = True)    
miss_bool = combined_data['Fare'].isnull() 
combined_data.loc[miss_bool,'Fare'] = combined_data.loc[miss_bool,'Pclass'].apply(lambda x : avg_price_per_class.loc[x])

combined_data.apply(lambda x: sum(x.isnull()))

# Feature Engineering
combined_data['TotalFamilyMember'] = combined_data['Parch'] + combined_data['SibSp'];

def sub_string(name):
    return name[name.find(',')+1: name.find('.')]

combined_data['Title'] =combined_data['Name'].map(lambda x: sub_string(x))
combined_data['Title'].value_counts()

def replaceTitle(x):
    title = x['Title'].strip()
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:
        return 'Mr'
    elif title in ['Countess', 'Mme','Lady','Dona','the Countess']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title 

combined_data['Title']=combined_data.apply(replaceTitle, axis=1)
combined_data['Title'].value_counts()

# Drop columns which are not required for processing.
combined_data = combined_data.drop(['Name'],axis=1)
combined_data = combined_data.drop(['Ticket'],axis=1)


# Encode the categorical columns
columns_to_encode = ['Embarked','Sex','Title']
combined_data = pd.get_dummies(combined_data, columns=columns_to_encode)

# Separate train and test data.
train_data = combined_data[combined_data['source'] == 'train']
test_data = combined_data[combined_data['source'] == 'test']

test_data.drop(['Survived','source'],axis=1,inplace=True)
train_data.drop(['source'],axis=1,inplace=True)

target = 'Survived'
IDcol = ['PassengerId']
submissionCols = ['PassengerId','Survived']

predictors = [x for x in train_data.columns if x not in [target]+IDcol]

X = train_data[predictors]
Y = train_data[target]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_data2 = sc.transform(test_data[predictors])

# Fitting classifier to the Training set

# Decision Tree classifier
'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
'''

'''
Confusion Matrix form classification model.
116	23
32	52
'''

# KNN Classifier
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
'''

'''
Confusion Matrix form classification model.
116	23
20	64
'''
# Random forest classifier.
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
'''
127	12
23	61
'''

# Predicting the Test set result
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

test_data[target] = classifier.predict(sc.transform(test_data[predictors]))

submission = test_data[IDcol]
submission[target] = test_data[target].astype(int)
submission[target].astype(int)
submission.to_csv(DIR+"/random_forest_classifier.csv", index=False)