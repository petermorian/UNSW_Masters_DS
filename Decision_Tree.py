import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import os
from graphviz import Source
from sklearn.tree import export_graphviz

dataframe = pd.read_csv('./data.csv')  # Load data
dataframe.dropna(inplace = True) #remove na's
print(dataframe.head()) #show first five rows of data
for col in dataframe.columns:
    print(col) #show column names

print("======================================")
print("======== STEP 1: ANALYSE DATA ========")
print("======================================")
print("")
print("== Data Types of all variables ==")
print(dataframe.dtypes)
print("===============================")

print("======== Summary Stats for numeric variables ========")
summary_stats1=dataframe.describe().T.rename(columns={"mean": "Mean", "50%": "Median (50%)"}).T
print(summary_stats1)
print("=====================================================")

print("====== Summary Stats for categorgical variables ======")
summary_stats2=dataframe.describe(include=['object', 'bool']).T
print(summary_stats2)
print("======================================================")

print("========= Unique Values =========")
for col in dataframe:
    print(col, ':', dataframe[col].unique())
print("=================================")

print("======= Correlation Matrix =======")
print(dataframe.corr())
print("==================================")

print("==== 5 Interesting Variables ====")
selected_cols = ['Churn', 'MonthlyCharges', 'TotalCharges', 'tenure' , 'Contract'] 
print("The 5 selected variables of interest are", selected_cols)
print("*** Churn ***")
print("Selected since it is the target variable of the model")
print(dataframe.iloc[:,  -1 ].value_counts())
print(dataframe.iloc[:,  -1 ].value_counts(normalize=True))
print("*** MonthlyCharges & TotalCharges ***")
print("Selected since their correlation is not very high.")
print("It was intially assumed that there would be an almost linear relationship between them, but their correlation is only 0.65.")
print("*** Tenure ***")
print("Selected since the standard deviation (24) is very high compared to the mean (32).")
print("Tenure is much more volatile that intially assumed.")
print("*** Contract ***")
print("It is assumed that longer contracts have better churn rates.")
print("Selected since we would like to know the proportion of customers paying monthly (and can thus easily leave) or stay with the company for longer.")
print(dataframe['Contract'].value_counts())
print(dataframe['Contract'].value_counts(normalize=True))

print("")
print("======================================")
print("======== STEP 2: PREPARE DATA ========")
print("======================================")
print("")

print("== Dropped customerID ==")
dataframe.drop(['customerID'],axis=1,inplace=True)
print("========================") #column not needed

print("== Number of Nulls ==")
print(dataframe.isnull().sum())
print("=====================") #no nulls

print("== Represent categorgical variables ==")
# make gender numeric
dataframe['gender']=dataframe['gender'].replace('Female',0)
dataframe['gender']=dataframe['gender'].replace('Male',1)

#make Y/N variables numeric
binary_vars = ['Partner', 'Dependents', 'PhoneService',
                'PaperlessBilling', 'Churn']
for i in binary_vars:
    dataframe[i]=dataframe[i].replace('No',0)
    dataframe[i]=dataframe[i].replace('Yes',1)

#make Y/N internet variables numeric
internet_vars = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in internet_vars: #(assumption: no internet service is the same as "No")
    dataframe[i]=dataframe[i].replace('No',0)
    dataframe[i]=dataframe[i].replace('No internet service',0)
    dataframe[i]=dataframe[i].replace('Yes',1)

# make MultipleLines numeric (assumption: no phone service is the same as "No")
dataframe['MultipleLines']=dataframe['MultipleLines'].replace('No',0)
dataframe['MultipleLines']=dataframe['MultipleLines'].replace('No phone service',0)
dataframe['MultipleLines']=dataframe['MultipleLines'].replace('Yes',1)
# make InternetService numeric (assumption: Fiber optic is best)
dataframe['InternetService']=dataframe['InternetService'].replace('No',0)
dataframe['InternetService']=dataframe['InternetService'].replace('DSL',1)
dataframe['InternetService']=dataframe['InternetService'].replace('Fiber optic',2)
# make Contract numeric (assumption: longer contact is better)
dataframe['Contract']=dataframe['Contract'].replace('Month-to-month',0)
dataframe['Contract']=dataframe['Contract'].replace('One year',1)
dataframe['Contract']=dataframe['Contract'].replace('Two year',2)
# make PaymentMethod numeric (assumption: auto payments are better)
dataframe['PaymentMethod']=dataframe['PaymentMethod'].replace('Mailed check',0)
dataframe['PaymentMethod']=dataframe['PaymentMethod'].replace('Electronic check',1)
dataframe['PaymentMethod']=dataframe['PaymentMethod'].replace('Credit card (automatic)',2)
dataframe['PaymentMethod']=dataframe['PaymentMethod'].replace('Bank transfer (automatic)',3)

#check changes
for col in dataframe:
    print(col, ':', dataframe[col].unique())
print("=====================") #transfrom characters to numerics

print("")
print("==== STEP 3: BUILD DECISION TREE ====")
print("=====================================")
print("")

print("************* ALL VARIABLES *************")
X = dataframe.iloc[:, 0 : -1 ]
X_names_all = X.columns
X_final_all = pd.DataFrame(X).to_numpy( )

Y = dataframe.iloc[:,  -1 ]
Y_dummy = pd.get_dummies(Y, columns=['Churn'], drop_first=True) 
Y_name = Y_dummy.columns
Y_s = pd.DataFrame(Y_dummy).to_numpy( )
Y_final = np.ravel(Y_s)

# split into training and test sets
X_train_all, X_test_all, Y_train_all, Y_test_all = train_test_split(X_final_all, Y_final, random_state=0)

#create best model
best_model_task1_all =  DecisionTreeClassifier(random_state=0,
                                             max_depth=5)

# train model 
DT_fit_all = best_model_task1_all.fit(X_train_all, Y_train_all)

#plot_tree(decision_tree) 
DT_Plot_all = export_text(DT_fit_all, 
                show_weights=True, 
                feature_names= list(X_names_all)) 
print("not printed due to large tree size")
print("")

print("************* SELECTED VARIABLES *************")
print("")
# 4 selected variabiables for decision tree
# based on importance of previous tree & correlation matrix
# the tree with all variables is not used due to overfitting
# 'SeniorCitizen', 'tenure' & 'MonthlyCharges' are important and not well correlated with each other
# 'InternetService' was used since by logic, it appears to be more common nowadays than phone services
selected_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'InternetService'] 

X = dataframe.iloc[:, 0 : -1 ]
X_select = X[np.intersect1d(X.columns, selected_cols)]
X_names = X_select.columns
X_final = pd.DataFrame(X_select).to_numpy( )

Y = dataframe.iloc[:,  -1 ]
Y_dummy = pd.get_dummies(Y, 
                        columns=['Churn'] , 
                        drop_first=True) 
Y_name = Y_dummy.columns
Y_s = pd.DataFrame(Y_dummy).to_numpy( )
Y_final = np.ravel(Y_s)

# split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_final, 
                                                    Y_final, 
                                                    random_state=0)

#create best model
best_model_task1 =  DecisionTreeClassifier(random_state=0,
                                             max_depth=5)

# train model 
DT_fit = best_model_task1.fit(X_train, Y_train)

#plot_tree(decision_tree) 
DT_Plot = export_text(DT_fit, 
                show_weights=True, 
                feature_names= list(X_names)) 
print(DT_Plot)



print("")
print("======= STEP 4: EVALUATE MODEL =======")
print("======================================")
print("")
# create confusion matrix
Y_predicted_all = DT_fit_all.predict(X_test_all)
CM_all = confusion_matrix(Y_test_all, Y_predicted_all)
class_names = ['Churn_No', 'Churn_Yes']
CM_Dataframe_all = pd.DataFrame(CM_all, 
                            index=class_names, 
                            columns=class_names)
Y_predicted = DT_fit.predict(X_test)
CM = confusion_matrix(Y_test, Y_predicted)
class_names = ['Churn_No', 'Churn_Yes']
CM_Dataframe = pd.DataFrame(CM, 
                            index=class_names, 
                            columns=class_names)
print('Confusion Matrix (all):')
print(CM_Dataframe_all)
print('Confusion Matrix:')
print(CM_Dataframe) 

print('')
print('Accuracy Score (all):',accuracy_score(Y_test_all, Y_predicted_all))
print('Accuracy Score:',accuracy_score(Y_test, Y_predicted))

print('')
print('Report (all): ')
print(classification_report(Y_test_all, Y_predicted_all))
print('Report: ')
print(classification_report(Y_test, Y_predicted))

print('')
print('Importance (all): ')
feature_importance_all = pd.DataFrame(best_model_task1_all.feature_importances_,
                                    index = X_names_all,
                                    columns=['importance']).sort_values('importance', ascending=False)
print(feature_importance_all)
print('')
print('Importance: ')
feature_importance = pd.DataFrame(best_model_task1.feature_importances_,
                                    index = X_names,
                                    columns=['importance']).sort_values('importance', ascending=False)
print(feature_importance)

# create Decision Tree chart
dot_data = export_graphviz(
        best_model_task1,
        out_file='tree.dot', 
        feature_names=X_names,
        class_names=class_names,
        rounded=True,
        filled=True)
graph = Source(dot_data)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

#create CM heatmap for all variabes
sns.heatmap(CM_Dataframe_all, 
            annot=True,  
            cmap="BuPu", 
            fmt=".0f")
plt.title("Confusion Matrix (all)")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.savefig('./confusion_matrix_all_task1.png')
plt.show()
plt.close()

#create CM heatmap for selected variabes
sns.heatmap(CM_Dataframe, 
            annot=True,  
            cmap="BuPu", 
            fmt=".0f")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.savefig('./confusion_matrix_task1.png')
plt.show()
plt.close()