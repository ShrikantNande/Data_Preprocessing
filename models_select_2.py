from Read import data
from Datatype import num_cols, cat_cols
import pandas as pd
import numpy as np

missing_value=data.isnull().sum()
data=data.dropna()


x= "int64"
df_int=data.select_dtypes([x])
print(df_int.shape)
z= "float64"
df_flt=data.select_dtypes([z])
df_flt=df_flt.round(1)
list_column = data.columns

#Find Catagorical Values
num_cols = data._get_numeric_data().columns#list of numarical variables
print('Numaric Data columns:',len(num_cols))
print('Numaric Data columns:',num_cols)

cat_cols=list(set(list_column) - set(num_cols))#List of categorical variables
print('Categorial Data Columns:',len(cat_cols))
print('Categorial Data Columns:',cat_cols)


#Conert Numarical to Categorical Type
def Read_cat(data):
    col_names=data.columns.tolist()
    #print(col_names)
    cat_var= []
    for i in col_names:  
        y=data[i].unique()
        #print(y)
        if len(y) < 10 :
            print("Column name for numarical data : ", i)
            print("Unique data in columns",y)
            cat_var.append(i)
            #print(cat_columns)
            #print(data[cat_columns].unique())
            #labelencoder_y = LabelEncoder()
            #y = labelencoder_y.fit_transform(cat_columns)
            #print(cat_columns)
            #num_cols.append(cat_cols)
        
    return cat_var

Numaric_to_Categorical = Read_cat(data)

print("Numarical Data Columns length :", len(Numaric_to_Categorical))
print("Numarical Data Columns : ", Numaric_to_Categorical)
#Convert Datatype name to categorical
for i in Numaric_to_Categorical:
    data[i] = pd.Categorical(data[i])

#num_cols = data._get_numeric_data().columns#list of numarical variables
cat_cols=list(set(list_column) - set(Numaric_to_Categorical))#List of categorical variables
print('Categorial Data Columns length:',len(cat_cols))
print('Categorial Data Columns:', cat_cols)

from sklearn.preprocessing import LabelEncoder
Dataframe=[]
col_names=[]
def read_data(data):
    for i in Numaric_to_Categorical:
        if data[i].dtype=='category':
            #print(data[i].shape)
            labelencoder_y = LabelEncoder()
            y = labelencoder_y.fit_transform(data[i])
            #print(cat_columns)
            col_names.append(i)
            Dataframe.append(y)
    return Dataframe
read_data(data)

lst = np.array(Dataframe).tolist()
len(lst)
# Calling DataFrame constructor on list 
df = pd.DataFrame(lst) 
##df=df.T
#df.shape
df_cat=df.T
df_cat.columns = col_names


df_col = pd.concat([df_int, df_flt], axis=1)
df_col.shape
shape=df_col.shape[0]

df_col.index = range(shape)
df_total=pd.concat([df_col, df_cat], axis=1)
data=df_total


                   
                   
print(data.head(5))



#target variable is in last columns
y=data[data.columns[len(data.columns)-1]]
x = data.iloc[:, :-1].values


print("Model List :\n 1) LinearDiscriminantAnalysis = LR \n 2) LogisticRegression = LoR \n 3) KNeighborsClassifier = KNN \n 4) GaussianNB = GNB \n 5) SVM = SVM \n 6) DecisionTreeClassifier = DT")
model_name = input("Select the model : ") 

# import model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# instantiate
LR = LinearRegression()
LoR = LogisticRegression()
KNN = KNeighborsClassifier()
GNB = GaussianNB()
SVM = SVC()
DT = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)


#Read File
y_list=[]
#Read File
def select_model(model_name):
    if model_name=='LR':
        # LinearRegression
        LR = LinearRegression()
        LR.fit(x, y)
        y_pred = LR.predict(X_test)
        print(f'alpha = {LR.intercept_}')
        print(f'betas = {LR.coef_}')
        print('Accuracy of Linear regression on test set: {:.2f}'.format(LR.score(X_test, y_test)))
        y_list.append(y_pred)
        
    elif model_name=='LoR':
        #Logistic_Regression_function()
        LoR = LogisticRegression()
        LoR.fit(X_train, y_train)
        y_pred = LoR.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(LoR.score(X_test, y_test)))
        y_list.append(y_pred)

    elif model_name=='KNN':
        #KNeighborsClassifier
        KNN.fit(X_train, y_train)
        y_pred = KNN.predict(X_test)
        print('Accuracy of KNN on test set: {:.2f}'.format(KNN.score(X_test, y_test)))
        y_list.append(y_pred)

    elif model_name=='GNB':
        #GaussianNB
        GNB.fit(X_train, y_train)
        y_pred = GNB.predict(X_test)
        print('Accuracy of GaussianNB on test set: {:.2f}'.format(GNB.score(X_test, y_test)))
        y_list.append(y_pred)

    elif model_name=='SVM':
        #SVM
        SVM.fit(X_train, y_train)
        y_pred = SVM.predict(X_test)
        print('Accuracy of SVM on test set: {:.2f}'.format(SVM.score(X_test, y_test)))
        y_list.append(y_pred)
        
    elif model_name=='DT':
        #DecisionTreeClassifier()
        DT.fit(X_train, y_train)
        y_pred = DT.predict(X_test)
        print('Accuracy of DecisionTreeClassifier on test set: {:.2f}'.format(DT.score(X_test, y_test)))
        y_list.append(y_pred)

    else:
        print('file not found') 
        
   
select_model(model_name)


