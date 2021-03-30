# Load in our libraries
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# Aseguremonos de que las imágenes aparezcan insertadas en iPython Notebook
# %matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

# usual ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
##xgBoost
from xgboost import XGBClassifier
#lightgbm
import lightgbm as lgb

pd.options.display.max_columns = None



print('')
print('******************************')
print('LOAD DATA')
print('******************************')
# Load in the train and test datasets
# dir_train = '../input/titanic/test.csv'
# dir_test = '../input/titanic/train.csv'
dir_train = 'titanic_train.csv'
dir_test =  'titanic_test.csv'
train = pd.read_csv(dir_train)
test =  pd.read_csv(dir_test)
print('FINISH...')

trainprue=train


print('')
print('******************************')
print('SHOW DATA')
print('******************************')

print('TRAIN') 
print(train.shape) 
# print(train.columns)   
# print(train.describe())  
# print(train.info())
# print(train.head(3))
print('')
print('COUNT')
print(train.groupby('Survived')['PassengerId'].count())
print(train.groupby('Pclass')['PassengerId'].count())
print(train.groupby('Sex')['PassengerId'].count())
print(train.groupby('Age')['PassengerId'].count())
print(train.groupby('SibSp')['PassengerId'].count())
print(train.groupby('Parch')['PassengerId'].count())
print(train.groupby('Ticket')['PassengerId'].count())
print(train.groupby('Fare')['PassengerId'].count())
print(train.groupby('Cabin')['PassengerId'].count())
print(train.groupby('Embarked')['PassengerId'].count())
print('')
print('ALL NULL COUNT')
print(train.isna().sum())

print('')
print('TEST')  
print(test.shape)
# print(test.columns)   
# print(test.describe())
# print(test.info())
# print(test.head(3))
print('')
print('COUNT')
print(test.groupby('Pclass')['PassengerId'].count())
print(test.groupby('Sex')['PassengerId'].count())
print(test.groupby('Age')['PassengerId'].count())
print(test.groupby('SibSp')['PassengerId'].count())
print(test.groupby('Parch')['PassengerId'].count())
print(test.groupby('Ticket')['PassengerId'].count())
print(test.groupby('Fare')['PassengerId'].count())
print(test.groupby('Cabin')['PassengerId'].count())
print(test.groupby('Embarked')['PassengerId'].count())
print('')
print('ALL NULL COUNT')
print(test.isna().sum())

 # 0   PassengerId  891 non-null    int64  
 # 1   Survived     891 non-null    int64  
 # 2   Pclass       891 non-null    int64  
 # 3   Name         891 non-null    object 
 # 4   Sex          891 non-null    object 
 # 5   Age          714 non-null    float64
 # 6   SibSp        891 non-null    int64  
 # 7   Parch        891 non-null    int64  
 # 8   Ticket       891 non-null    object 
 # 9   Fare         891 non-null    float64
 # 10  Cabin        204 non-null    object 
 # 11  Embarked     889 non-null    object 
print('FINISH...')



print('')
print('******************************')
print('PLOT')
print('******************************')

# Histograma
train.hist(figsize=(20,16))


# heatmap
plt.figure(figsize=(15,15))
sns.heatmap(train.corr(), annot=True, cmap='Oranges', fmt='.2f')


# barh y pie
labels = ['No Sobrevive', 'Sobrevive'] 
explode=(0.1,0)
cores= ['#009ACD', '#ADD8E6']

survived_count = pd.cut(x=train.Survived, bins=2, labels=labels, include_lowest=True).value_counts() 
survived_perc = (pd.value_counts(pd.cut(x=train.Survived, bins=2,labels= labels, include_lowest=True),normalize=True) * 100).round(1) 
quant_sobrevi = pd.DataFrame({'survived_count':survived_count, 'Taxa de Supervivientes':survived_perc}) 
percentages = list(quant_sobrevi['Taxa de Supervivientes'])

fig, (axis1, axis2) = plt.subplots(1,2, figsize=(20,8))
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor']= 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.size']=15

x=quant_sobrevi.index
y=quant_sobrevi['survived_count']

axis1.barh(x, y, color=cores, label=labels)
axis1.set_title('Supervivientes', fontsize=20, pad=20)
axis1.set_xlabel('Numero de pasajeros', fontsize=15)

axis2.pie(percentages, explode=explode, labels=labels, colors = cores, autopct='%1.0f%%', shadow=True, startangle=0, pctdistance=0.5,labeldistance=1.1)
axis2.set_title('Taxa de Supervivientes', fontsize=20, pad=20)


# barplot
quant_sex = pd.DataFrame(train.Sex.value_counts())
quant_sex['Supervivientes por Sexo'] = train.groupby('Sex')['Survived'].sum()

quant_emb = pd.DataFrame(train.Embarked.value_counts()) 
quant_emb['Supervivientes por Embarque'] = train.groupby('Embarked')['Survived'].sum()

fig, (axis1, axis2) = plt.subplots(1,2, figsize=(20,8))

plt.figure(figsize=(10,8))
ax = sns.barplot(x=quant_sex.index, y='Sex', data=quant_sex, ax=axis1, palette='Greys_r' )
ax.set_title('Supervivientes por Sexo', fontsize=20,pad=20,)
ax.set_xlabel('Sexo', fontsize=15)
ax.set_ylabel('Quantidade', fontsize=15)
plt.tight_layout()

plt.figure(figsize=(10,8))
ax = sns.barplot(x=quant_emb.index, y='Embarked', data=quant_emb, ax=axis2, palette='Greys_r' )
ax.set_title('Supervivientes por Embarque', fontsize=20,pad=20,)
ax.set_xlabel('Embarque', fontsize=15)
ax.set_ylabel('Quantidade', fontsize=15)
plt.tight_layout()



# barplot
train_1 = train.copy()
train_1.Age.fillna(train_1.Age.median(), inplace=True)  
train_1.Fare.fillna(train_1.Fare.median(), inplace=True)  

# histplot
plt.figure(figsize=(10,8))
ax = sns.histplot(data=train_1, x='Age', hue='Survived', palette='Blues')
ax.set_title('Distribucion de pasageros por edades', fontsize=20, pad=15)
ax.set_xlabel('edades', fontsize=15)
ax.set_ylabel('cantidad', fontsize=15)



# histplot
plt.figure(figsize=(10,8))
ax = sns.histplot(data=train_1, x='Fare', hue='Survived', palette='Blues')
ax.set_title('Distribucion de pasageros por Fare', fontsize=20, pad=15)
ax.set_xlabel('Fare', fontsize=15)
ax.set_ylabel('cantidad', fontsize=15)



# classes = [0,16,32,48,64,81]
# classes = [0,12,18,60,81]
classes = [0,13,27,30,55,81]
labels = ['infantiles','Adolescentes',' Adultos', 'Mayores', 'max']

frequencia_idade = pd.value_counts(pd.cut(x=train_1.Age, bins=classes, labels= labels, include_lowest=True))
freq_perc = pd.DataFrame({'Frequencia':frequencia_idade})

plt.figure(figsize=(10,8))
ax = sns.barplot(x=freq_perc.index, y='Frequencia', data=freq_perc, palette='Greens')
ax.set_title('Cantidad de personas por grupo de edad', fontsize=20, pad=20)
ax.set_ylabel('Frequencia', fontsize=15)
ax.set_xlabel('Grupo edad', fontsize=15)


plt.figure(figsize=(15,8))
sns.kdeplot(train["Age"][train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(train["Age"][train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
plt.show()


plt.figure(figsize=(15,8))
sns.kdeplot(train["Fare"][train.Survived == 1], color="#e74c3c", shade=True)
sns.kdeplot(train["Fare"][train.Survived == 0], color="#3498db", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)
plt.show()


# plt.figure(figsize=(25,8))
# avg_survival_byage = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
# g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")


print('FINISH...')




print('')
print('******************************')
print('PREPARE DATA')
print('******************************')
# Remove all NULLS
# Age column
age_avg = (train['Age'].mean() + test['Age'].mean()) / 2
age_std = (train['Age'].std()  + test['Age'].std())  / 2
age_null_count = train['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
train['Age'][np.isnan(train['Age'])] = age_null_random_list
age_null_count = test['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
test['Age'][np.isnan(test['Age'])] = age_null_random_list

# Cabin column
train['Cabin'].fillna(np.random.randint(1, 500), inplace=True)
test[ 'Cabin'].fillna(np.random.randint(1, 500), inplace=True)

# Fare column
fare_avg = (train['Fare'].mean() + test['Fare'].mean()) / 2
train['Fare'].fillna(fare_avg, inplace=True)
test[ 'Fare'].fillna(fare_avg, inplace=True)

# age_avg = (train['Fare'].mean() + test['Fare'].mean()) / 2
# age_std = (train['Fare'].std()  + test['Fare'].std())  / 2
# age_null_count = train['Fare'].isnull().sum()
# age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
# train['Fare'][np.isnan(train['Fare'])] = age_null_random_list
# age_null_count = test['Fare'].isnull().sum()
# age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
# test['Fare'][np.isnan(test['Fare'])] = age_null_random_list



# Embarked column
train['Embarked'].fillna('S', inplace=True)
test[ 'Embarked'].fillna('S', inplace=True)

# Prepare data

# Survived
# 0    549
# 1    342

# Pclass
# 1    216
# 2    184
# 3    491

# Sex
# female    314
# male      577
train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test[ 'Sex'] = test[ 'Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Age
# 0.42     1
# 0.67     1
#         ..
# 74.00    1
# 80.00    1
train['Age'] = train['Age'].astype(int)
train.loc[ train['Age'] <= 16, 'Age'] 					     = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age']                         = 4 
test['Age'] = test['Age'].astype(int)
test.loc[  test['Age']  <= 16, 'Age'] 					     = 0
test.loc[ (test['Age']  > 16) & (test['Age']  <= 32), 'Age'] = 1
test.loc[ (test['Age']  > 32) & (test['Age']  <= 48), 'Age'] = 2
test.loc[ (test['Age']  > 48) & (test['Age']  <= 64), 'Age'] = 3
test.loc[  test['Age']  > 64, 'Age']                         = 4

# SibSp
# 0    608
# 1    209
# 2     28
# 3     16
# 4     18
# 5      5
# 8      7

# Parch
# 0    678
# 1    118
# 2     80
# 3      5
# 4      4
# 5      5
# 6      1
# Create new feature FamilySize as a combination of SibSp and Parch
train['FamilySize'] = train['SibSp'] + train['Parch']
test[ 'FamilySize'] = test[ 'SibSp'] + test[ 'Parch']
# Create new feature IsAlone from FamilySize
# train['IsAlone'] = np.where(train['FamilySize'] > 0, 0, 1)
# test[ 'IsAlone'] = np.where(test[ 'FamilySize'] > 0, 0, 1)

# Ticket
# 110152         3
# 110413         3
#               ..
# W/C 14208      1
# WE/P 5735      2
train['Ticket_type'] = train['Ticket'].apply(lambda x: x[0:3])
train['Ticket_type'] = train['Ticket_type'].astype('category')
train['Ticket_type'] = train['Ticket_type'].cat.codes
test['Ticket_type']  = test['Ticket'].apply(lambda x: x[0:3])
test['Ticket_type']  = test['Ticket_type'].astype('category')
test['Ticket_type']  = test['Ticket_type'].cat.codes

# Fare
# 0.0000      15
# 4.0125       1
#             ..
# 263.0000     4
# 512.3292     3
train.loc[ train['Fare'] <= 7.91, 'Fare'] 						       = 0
train.loc[(train['Fare'] >  7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] >  14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[ train['Fare'] >  31, 'Fare'] 						       = 3
train['Fare'] = train['Fare'].astype(int)
test.loc[ test['Fare']   <= 7.91, 'Fare']                              = 0
test.loc[(test['Fare']   >  7.91) & (test['Fare'] <= 14.454), 'Fare']  = 1
test.loc[(test['Fare']   >  14.454) & (test['Fare'] <= 31), 'Fare']    = 2
test.loc[ test['Fare']   >  31, 'Fare'] 							   = 3
test['Fare'] = test['Fare'].astype(int)

# Cabin
# A10    1
# A14    1
#       ..
# G6     4
# T      1
# # Feature that tells whether a passenger had a cabin on the Titanic
# train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
# test[ 'Has_Cabin'] = test[ 'Cabin'].apply( lambda x: 0 if type(x) == float else 1)

# Embarked
# C    168
# Q     77
# S    644
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test[ 'Embarked'] = test[ 'Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



print('train')
print(train.head(3))
print('test')
print(test.head(3))

print('FINISH...')



print('')
print('******************************')
print('SHOW DATA FINISH')
print('******************************')
print('Count')
print(train.groupby(['Pclass',      'Survived'])['PassengerId'].count())
print(train.groupby(['Sex',         'Survived'])['PassengerId'].count())
print(train.groupby(['Age',         'Survived'])['PassengerId'].count())
print(train.groupby(['SibSp',       'Survived'])['PassengerId'].count())
print(train.groupby(['Parch',       'Survived'])['PassengerId'].count())
print(train.groupby(['FamilySize',  'Survived'])['PassengerId'].count())
# print(train.groupby(['IsAlone',     'Survived'])['PassengerId'].count())
print(train.groupby(['Ticket_type', 'Survived'])['PassengerId'].count())
print(train.groupby(['Fare',        'Survived'])['PassengerId'].count())
# print(train.groupby(['Has_Cabin',   'Survived'])['PassengerId'].count())
print(train.groupby(['Embarked',    'Survived'])['PassengerId'].count())

print('FINISH...')



print('')
print('******************************')
print('DELETE COLUMN')
print('******************************')

# print('ALL NULL COUNT')
# print(train.isna().sum())


# Store our passenger ID for easy access
PassengerId = test['PassengerId']

# Feature selection
drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
train = train.drop(drop_elements, axis = 1)
test  = test.drop( drop_elements, axis = 1)

# # Feature selection
# drop_elements = ['FamilySize']
# train = train.drop(drop_elements, axis = 1)
# test  = test.drop( drop_elements, axis = 1)

# # Feature selection
# drop_elements = ['IsAlone']
# train = train.drop(drop_elements, axis = 1)
# test  = test.drop( drop_elements, axis = 1)

# # Feature selection
# drop_elements = ['Ticket_type']
# train = train.drop(drop_elements, axis = 1)
# test  = test.drop( drop_elements, axis = 1)

# # Feature selection
# drop_elements = ['Has_Cabin']
# train = train.drop(drop_elements, axis = 1)
# test  = test.drop( drop_elements, axis = 1)

# train = train.drop('Pclass', axis = 1)
# train = train.drop('Sex', axis = 1)
# train = train.drop('Age', axis = 1) 
# train = train.drop('Fare', axis = 1)  
# train = train.drop('Embarked', axis = 1)

# test = test.drop('Pclass', axis = 1)
# test = test.drop('Sex', axis = 1)
# test = test.drop('Age', axis = 1) 
# test = test.drop('Fare', axis = 1)  
# test = test.drop('Embarked', axis = 1)

                   
print('FINISH...')



print('')
print('******************************')
print('VARIABLES X_train Y_train X_test')
print('******************************')
Y_train = train['Survived'].ravel()
X_train = train.drop(['Survived'], axis=1).values # Creates an array of the train data
X_test  = test.values 
print('')
print('X_train')
print(X_train)
print('Y_train')
print(Y_train)
print('X_test')
print(X_test)

print('FINISH...')



print('')
print('******************************')
print('MODELS TRAINNING')
print('******************************')

# KNN - K-nearest-neighbours
model = KNeighborsClassifier(n_neighbors = 3).fit(X_train, Y_train)
scr_knn = round(model.score(X_train, Y_train) * 100, 2)
score_Max = scr_knn
Y_pred_Max = model.predict(X_test)
name_Max = 'KNN - K-nearest-neighbours'

# Support Vector Machines
model = SVC().fit(X_train, Y_train)
scr_svc = round(model.score(X_train, Y_train) * 100, 2)
if (scr_svc > score_Max) :
    score_Max = scr_svc
    Y_pred_Max = model.predict(X_test)
    name_Max = 'Support Vector Machines'    
    
# Logistic Regression
model = LogisticRegression().fit(X_train, Y_train)
scr_log = round(model.score(X_train, Y_train) * 100, 2)
if (scr_log > score_Max) :
    score_Max = scr_log
    Y_pred_Max = model.predict(X_test)
    name_Max = 'Logistic Regression'

# Naive Bayes
model = GaussianNB().fit(X_train, Y_train)
scr_gaussian = round(model.score(X_train, Y_train) * 100, 2)
if (scr_gaussian > score_Max) :
    score_Max = scr_gaussian
    Y_pred_Max = model.predict(X_test)
    name_Max = 'Gaussian Naive Bayes'

# Perceptron
model = Perceptron().fit(X_train, Y_train)
scr_perceptron = round(model.score(X_train, Y_train) * 100, 2)
if (scr_perceptron > score_Max) :
    score_Max = scr_perceptron
    Y_pred_Max = model.predict(X_test)
    name_Max = 'Perceptron'

# Linear SVC
model = LinearSVC().fit(X_train, Y_train)
scr_linear_svc = round(model.score(X_train, Y_train) * 100, 2)
if (scr_linear_svc > score_Max) :
    score_Max = scr_linear_svc
    Y_pred_Max = model.predict(X_test)
    name_Max = 'Linear SVC'

# Stochastic Gradient Descent
model = SGDClassifier().fit(X_train, Y_train)
scr_sgd = round(model.score(X_train, Y_train) * 100, 2)
if (scr_sgd > score_Max) :
    score_Max = scr_sgd
    Y_pred_Max = model.predict(X_test)
    name_Max = 'SGDClassifier'

# Decision Tree
model = DecisionTreeClassifier().fit(X_train, Y_train)
scr_decision_tree = round(model.score(X_train, Y_train) * 100, 2)
if (scr_decision_tree > score_Max) :
    score_Max = scr_decision_tree
    Y_pred_Max = model.predict(X_test)
    name_Max = 'DecisionTreeClassifier'

# Random Forest
model = RandomForestClassifier().fit(X_train, Y_train)
scr_random_forest = round(model.score(X_train, Y_train) * 100, 2)
if (scr_random_forest > score_Max) :
    score_Max = scr_random_forest
    Y_pred_Max = model.predict(X_test)
    name_Max = 'RandomForestClassifier'

# XG Boost
model = XGBClassifier().fit(X_train, Y_train)
scr_xg_boost = round(model.score(X_train, Y_train) * 100, 2)
if (scr_xg_boost > score_Max) :
    score_Max = scr_xg_boost
    Y_pred_Max = model.predict(X_test)
    name_Max = 'XGBClassifier'

# Ada Boosting
model = AdaBoostClassifier().fit(X_train, Y_train)
scr_ada_boost = round(model.score(X_train, Y_train) * 100, 2)
if (scr_ada_boost > score_Max) :
    score_Max = scr_ada_boost
    Y_pred_Max = model.predict(X_test)
    name_Max = 'AdaBoostClassifier'

# Gradient Boosting
model = GradientBoostingClassifier().fit(X_train, Y_train)
scr_gr_boost = round(model.score(X_train, Y_train) * 100, 2)
if (scr_gr_boost > score_Max) :
    score_Max = scr_gr_boost
    Y_pred_Max = model.predict(X_test)
    name_Max = 'GradientBoostingClassifier'

# lightgbm
model = lgb.LGBMRegressor( n_estimators=1000 ).fit(X_train, Y_train)
scr_lgb = round(model.score(X_train, Y_train) * 100, 2)
if (scr_lgb > score_Max) :
    score_Max = scr_lgb
    Y_pred_Max = model.predict(X_test)
    name_Max = 'lightgbm'
     
print('FINISH...')



print('')
print('******************************')
print('MODEL COMPARISON')
print('******************************')
# Model Comparison
models = pd.DataFrame({
    'Model': ['Gradient Boosting', 'Ada Boosting', 'XGBoost', 'Support Vector Machines', 'KNN', 
              'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 
              'Linear SVC', 'Decision Tree', 'lgb'],
    'Score': [scr_gr_boost, scr_ada_boost, scr_xg_boost, scr_svc, scr_knn, 
              scr_log, scr_random_forest, scr_gaussian, scr_perceptron, scr_sgd, 
              scr_linear_svc, scr_decision_tree, scr_lgb]})
print(models.sort_values(by='Score', ascending=True))

print('FINISH...')



print('')
print('******************************')
print('SUBMISSION FILE')
print('******************************')
# Evaluation
# Generating Ground truth
# Submission file
# Finally having trained and fit all our first-level and second-level models, we can now output the predictions into the proper format for submission to the Titanic competition as follows:

print('El mejor modelo es', name_Max, 'con un score', score_Max)
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId, 'Survived': Y_pred_Max })
print('StackingSubmission.head')
print(StackingSubmission.head())

# Generate Submission File 
StackingSubmission.to_csv('prediccion.csv', index=False)

print('FINISH...')
