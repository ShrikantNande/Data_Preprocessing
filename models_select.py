from Read import data
from Datatype import num_cols, cat_cols

print("Select input variables by Data types such as \n 1) int64 \n 2) float64 \n 3) object")
x= input("Input_Data_type :")
#print(x)
x=data.select_dtypes([x])

#target variable is in last columns
y=data[data.columns[len(data.columns)-1]]

print("Model List :\n 1) LinearDiscriminantAnalysis = LR \n 2) LogisticRegression = LoR \n 3) KNeighborsClassifier = KNN \n 4) GaussianNB = GNB \n 5) SVM = SVM \n 6) DecisionTreeClassifier = DT")
model_name = input("Select the model : ") 

# import model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
def select_model(model_name):
    if model_name=='LR':
        # LinearRegression
        LR = LhinearRegression()
        #LR.fit(X_train, y_train)
        model = LR.fit(x, y)
        print(f'alpha = {model.intercept_}')
        print(f'betas = {model.coef_}')
        print("Model predictions :",model.predict(x))
        print('Accuracy of Linear regression on test set: {:.2f}'.format(LR.score(X_test, y_test)))

        
    elif model_name=='LoR':
        #Logistic_Regression_function()
        LoR = LogisticRegression()
        LoR.fit(X_train, y_train)
        y_pred = LoR.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(LoR.score(X_test, y_test)))
        
    elif model_name=='KNN':
        #KNeighborsClassifier
        KNN.fit(X_train, y_train)
        y_pred = KNN.predict(X_test)
        print('Accuracy of KNN on test set: {:.2f}'.format(KNN.score(X_test, y_test)))
       
    elif model_name=='GNB':
        #GaussianNB
        GNB.fit(X_train, y_train)
        y_pred = GNB.predict(X_test)
        print('Accuracy of GaussianNB on test set: {:.2f}'.format(GNB.score(X_test, y_test)))
        
    elif model_name=='SVM':
        #SVM
        SVM.fit(X_train, y_train)
        y_pred = SVM.predict(X_test)
        print('Accuracy of SVM on test set: {:.2f}'.format(SVM.score(X_test, y_test)))
        
    elif model_name=='DT':
        #DecisionTreeClassifier()
        DT.fit(X_train, y_train)
        y_pred = DT.predict(X_test)
        print('Accuracy of DecisionTreeClassifier on test set: {:.2f}'.format(DT.score(X_test, y_test)))
        
    else:
        print('file not found') 
        
   
select_model(model_name)

