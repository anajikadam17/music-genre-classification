# --------------
# import packages

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# code starts here
df = pd.read_csv(path)
X = df.drop('label', 1)
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)







# code ends here


# --------------
# box plot on spectral bandwidth
bandwidth = X_train['spectral_bandwidth']
sns.distplot(bandwidth)
plt.show()

# histograme for zero crossing rate
zc_rate = X_train['zero_crossing_rate']
sns.distplot(zc_rate)
plt.show()
# histogram for spectral centroid
centroid = X_train['spectral_centroid']
sns.distplot(centroid)
plt.show()



# --------------
# import the packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# code starts here

# instantiate standard scaler
sc = StandardScaler()

# fit and transform on X_train
sc.fit(X_train)
X_train = sc.transform(X_train)

# transform the standard scaler on X_test
X_test = sc.transform(X_test)

# instantiate logistic regression model
lr=LogisticRegression(random_state=9)

# fit the logistic model on X_trian , y_train
lr.fit(X_train, y_train)

# make predictions 
y_pred_lr = lr.predict(X_test)

# calculate the accuracy for logistic regression
accuracy_lr = accuracy_score(y_test,y_pred_lr)
print(accuracy_lr)


# code ends here


# --------------
#import the packages 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# instantiate standard scaler
svm = SVC()

# fit and transform on X_train
svm.fit(X_train,y_train)
y_pred_sc = svm.predict(X_test)
accuracy_sc = accuracy_score(y_test,y_pred_sc)
print(accuracy_sc)
# instantiate logistic regression model
rf=RandomForestClassifier(random_state=9)

# fit the logistic model on X_trian , y_train
rf.fit(X_train, y_train)

# make predictions 
y_pred_rf = rf.predict(X_test)

# calculate the accuracy for logistic regression
accuracy_rf = accuracy_score(y_test,y_pred_rf)
print(accuracy_rf)


# --------------
# import packages
from sklearn.ensemble import BaggingClassifier

# instantiate baggingclassifier
bagging_clf = BaggingClassifier(lr, n_estimators=50, max_samples=100, 
                                bootstrap=True, random_state=9)

# fit the classifier on X_train,y_train
bagging_clf.fit(X_train, y_train)

# make the prediction
y_pred_bag = bagging_clf.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred_bag)
print(accuracy)


# --------------
# import packages
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

# code starts here
nv = GaussianNB()

# fit the classifier on X_train,y_train
nv.fit(X_train, y_train)

voting_clf_soft = VotingClassifier([('lr', lr), ('rf', rf), ('nv', nv)], voting = 'soft')
voting_clf_soft.fit(X_train, y_train)
# make the prediction
y_pred_soft = voting_clf_soft.predict(X_test)

# calculate the accuracy
accuracy_soft = accuracy_score(y_test, y_pred_soft)
print(accuracy_soft)

# code ends here


