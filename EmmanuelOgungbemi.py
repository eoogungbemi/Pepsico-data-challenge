#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from IPython.display import display

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# reading the data

df = pd.read_csv('pepsico.csv')


# In[3]:


# To see the testing dataset profiling
pandas_profiling.ProfileReport(df)


# In[4]:


train = df
train.shape


# In[5]:


train.info()


# Drop columns with over 30% missing data. That is below 524 non-null object.

# In[6]:


# removing the Storage Conditions column
train = train.drop(['Storage Conditions'], axis = 1)


# In[7]:


# removing the Packaging Stabilizer Added column
train = train.drop(['Packaging Stabilizer Added'], axis = 1)


# In[8]:


# removing the Transparent Window in Package column
train = train.drop(['Transparent Window in Package'], axis = 1)


# In[9]:


# removing the Preservative Added column
train = train.drop(['Preservative Added'], axis = 1)


# In[10]:


# removing the Moisture (%) column
train = train.drop(['Moisture (%)'], axis = 1)


# In[11]:


# removing the Residual Oxygen (%) column
train = train.drop(['Residual Oxygen (%)'], axis = 1)


# In[12]:


# removing the Hexanal (ppm) column
train = train.drop(['Hexanal (ppm)'], axis = 1)


# In[13]:


# removing the Study Number column
train = train.drop(['Study Number'], axis = 1)


# In[14]:


# removing the Sample ID column
train = train.drop(['Sample ID'], axis = 1)


# In[15]:


train.head()


# In[16]:


train.info()


# In[17]:


print(train.isnull().sum())


# In[18]:


train['Base Ingredient'].fillna(train['Base Ingredient'].mode()[0], inplace = True)


# In[19]:


print(train.isnull().sum())


# In[20]:


train.columns


# In[21]:


train.corr()


# In[22]:


# getting their shapes
print("Shape of train:", train.shape)


# In[23]:


# Convert to a Classification task
clean_data = train.copy() # New data frame to avoid confusion 
clean_data['Difference From Fresh'] = (clean_data['Difference From Fresh'] > 20) * 1


# In[24]:


y = clean_data['Difference From Fresh']
y.head()


# In[25]:


# removing the Difference From Fresh column
train = train.drop(['Difference From Fresh'], axis = 1)


# In[26]:


train.head()


# In[27]:


X = train.iloc[:,0:]
X.head()


# In[28]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
for col in X:
    # Compare if the dtype is object
    if X[col].dtype=='object':
    # Use LabelEncoder to do the numeric transformation
         X[col]=le.fit_transform(X[col])


# In[29]:


X.head()


# In[30]:


import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[31]:


from sklearn.preprocessing import StandardScaler
X = np.nan_to_num(X)
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[32]:


# splitting x and y into train and validation sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test: ", y_test.shape)


# <h2 id="modeling">Modeling (Logistic Regression with Scikit-learn)</h2>

# Lets build our model using __LogisticRegression__ from Scikit-learn package. This function implements logistic regression and can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You can find extensive information about the pros and cons of these optimizers if you search it in internet.
# 
# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem in machine learning models.
# __C__ parameter indicates __inverse of regularization strength__ which must be a positive float. Smaller values specify stronger regularization. 
# Now lets fit our model with train set:

# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# Now we can predict using our test set:

# In[34]:


predict = LR.predict(X_test)
predict


# In[35]:


predict_prob = LR.predict_proba(X_test)
predict_prob


# <h2 id="evaluation">Evaluation</h2>

# ### jaccard index
# Lets try jaccard index for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the union of two label sets. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

# In[36]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, predict)


# ### confusion matrix
# Another way of looking at accuracy of classifier is to look at __confusion matrix__.

# In[37]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, predict, labels=[1,0]))


# In[38]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predict, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['not_fresh=1','fresh=0'],normalize= False,  title='Confusion matrix')


# The model correctly predicted 117 samples as fresh. 
# It wrongly predicts 33 samples as fresh.
# The model is good only when predicting freshness of samples

# In[39]:


print("Training Accuracy :", LR.score(X_train, y_train))

print("Validation Accuracy :", LR.score(X_test, y_test))

print (classification_report(y_test, predict))


# In[40]:


from sklearn.metrics import log_loss
log_loss(y_test, predict_prob)


# ### Let us use other classifiers to see if we can get a better accuracy

# ### Random Forest Classifer

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
rfc_train_score = rfc.score(X_train, y_train)


print("Training Accuracy :", rfc.score(X_train, y_train))

print("Validation Accuracy :", rfc.score(X_test, y_test))

cm = confusion_matrix(y_test, rfc_pred)
print(cm)

cr = classification_report(y_test, rfc_pred)
print(cr)


# In[42]:


# Visualizing Confusion Matrix using Heatmap
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0)
dtc_train = dtc.fit(X_train, y_train)

dtc_pred = dtc.predict(X_test)

dtc_train_score = dtc.score(X_train, y_train)
dtc_val_score = dtc.score(X_test, y_test)

print("Training Accuracy :", dtc.score(X_train, y_train))

print("Validation Accuracy :", dtc.score(X_test, y_test))

cm = confusion_matrix(y_test, dtc_pred)
print(cm)

cr = classification_report(y_test, dtc_pred)
print(cr)


# In[44]:


#Xg-Boost Classifier

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

xgb = XGBClassifier()
xgb_train = xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_train_score = xgb.score(X_train, y_train)
xgb_val_score = xgb.score(X_test, y_test)

print("Training Accuracy :", xgb.score(X_train, y_train))

print("Validation Accuracy :", xgb.score(X_test, y_test))

cm = confusion_matrix(y_test, xgb_pred)
print(cm)

cr = classification_report(y_test, xgb_pred)
print(cr)


# In[45]:


# Visualizing Confusion Matrix using Heatmap
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[46]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

print("Training Accuracy :", logreg.score(X_train, y_train))

print("Validation Accuracy :", logreg.score(X_test, y_test))

cm = confusion_matrix(y_test, logreg_pred)
print(cm)

cr = classification_report(y_test, logreg_pred)
print(cr)


# In[47]:


# Visualizing Confusion Matrix using Heatmap
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[48]:


# Gaussian Naive Bayes
# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
gaussian_pred = gaussian.predict(X_test)

print("Training Accuracy :", gaussian.score(X_train, y_train))

print("Validation Accuracy :", gaussian.score(X_test, y_test))

cm = confusion_matrix(y_test, gaussian_pred)
print(cm)

cr = classification_report(y_test, gaussian_pred)
print(cr)


# In[49]:


# Support Vector Classifier(SVC)
from sklearn.svm import SVC 

svc = SVC(gamma='auto')
svc = svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print("Training Accuracy :", svc.score(X_train, y_train))

print("Validation Accuracy :", svc.score(X_test, y_test))

cm = confusion_matrix(y_test, svc_pred)
print(cm)

cr = classification_report(y_test, svc_pred)
print(cr)


# In[50]:


# K-Nearest Neighbors
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train) 

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("Training Accuracy :", knn.score(X_train, y_train))

print("Validation Accuracy :", knn.score(X_test, y_test))

cm = confusion_matrix(y_test, knn_pred)
print(cm)

cr = classification_report(y_test, knn_pred)
print(cr)


# In[51]:


# Visualizing Confusion Matrix using Heatmap
# import required modules
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[52]:


# fit an Extra Trees model to the data
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
# display the relative importance of each attribute
print(model.feature_importances_)
#plot feature importance
#This shows the importance of each features in determining the model
from xgboost import plot_importance
from matplotlib import pyplot
plot_importance(xgb_train)
pyplot.show()


# In[53]:


#Xg-Boost Classifier

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

xgb = XGBClassifier()
xgb_train = xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_train_score = xgb.score(X_train, y_train)
xgb_val_score = xgb.score(X_test, y_test)

print("Training Accuracy :", xgb.score(X_train, y_train))

print("Validation Accuracy :", xgb.score(X_test, y_test))

cm = confusion_matrix(y_test, xgb_pred)
print(cm)

cr = classification_report(y_test, xgb_pred)
print(cr)


# In[54]:


# Visualizing Confusion Matrix using Heatmap
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[55]:


# Spot Check Algorithms
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
models = []
models.append(('LR', LogisticRegression(C=0.01, solver='liblinear')))
models.append(('XGB', XGBClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RFC', RandomForestClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models: 
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold,)
    results.append(cv_results)
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)


# In[56]:


### cross validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]

classifiers=['LR','XGB', 'KNN','DTC','NB','SVM','RFC']
models=[LogisticRegression(solver='liblinear', multi_class='ovr'),
        XGBClassifier(),KNeighborsClassifier(),DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0), GaussianNB(),SVC(gamma='auto'),RandomForestClassifier()]

for i in models:
    model = i
    cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
    
models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
models_dataframe


# In[57]:


prediction = rfc.predict(X_test)


# In[58]:


predict_rfc =rfc.predict_proba(X)
predict_rfc # result in array


# In[59]:


prediction =pd.DataFrame(predict_rfc,columns=['Class_1','Class_2'])
print(prediction.head())


# In[62]:


y1=df.iloc[:,:0:1].join(prediction)


# In[63]:


y1 


# In[65]:


Pred_sample = rfc.predict(X)
Pred_sample 


# In[66]:


Pfinal = pd.DataFrame(Pred_sample)


# In[67]:


Pfinal


# In[78]:


Pfinal.to_csv(r'C:\Users\Amy\Downloads\Prediction.csv')

