
# coding: utf-8

# ### Titanic: Machine Learning from Disaster is a contest offered by the Kaggle site , where each participant must predict passenger survival or death through different variables such as age, gender or class

#  ## Importing Libraries

# In[2]:

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import re as re


# ## Reading Datasets

# In[3]:

train=pd.read_csv('train.csv')
test= pd.read_csv('test.csv')
full_data=[train,test]


# Key             Definition
# 
# survival	    Survival	0 = No, 1 = Yes
# 
# pclass	        Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# 
# sex	                  Sex	
# 
# Age	             Age in years	
# 
# sibsp	        # of siblings / spouses aboard the Titanic	
# 
# parch	        # of parents / children aboard the Titanic	
# 
# ticket	        Ticket number	
# 
# fare	        Passenger fare	
# 
# cabin	        Cabin number	
# 
# embarked	    Port of Embarkation

# In[4]:

train.head()


# In[5]:

train.info()


# In[6]:

test.info()


# We have null entries in  Age, Cabin and Embarked, we'll have to impute these later

# ## EDA

# In[10]:

train['Age'].hist(bins=70)


# In[12]:

sns.factorplot('Pclass','Survived',data=train)


# ## Creating/Extracting Features

# ### Function for getting Title from name

# In[14]:

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# ### Dropping PassengerId and Name

# In[15]:

train.drop(['PassengerId','Name'],axis=1,inplace=True)


# In[16]:

train.Title.unique()


# ### Categorizing title in 5 categories

# In[17]:

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# ### Imputing missing values of Age with median of respective title

# In[18]:

for dataset in full_data:
    for title in ['Mr','Mrs','Miss','Master','Rare']:
        mask1= dataset['Title']==title
        mask2= dataset['Age'].isnull()
        indices=dataset[mask1][mask2].index
        for i in indices:
            dataset['Age'][i]=(train[train['Title']==title])['Age'].median()


# In[21]:

test.drop(['Name'],axis=1,inplace=True)


# In[22]:

train.info()


# ### Creating new feature is_child for Age less than 15

# In[23]:

for dataset in full_data:
    dataset['is_child']=0
    dataset.loc[dataset['Age']<=14,'is_child']=1


# In[24]:

train.head()


# ### Creating more new features from SibSp and Parch

# In[25]:

for dataset in full_data:
    dataset['family_size']=dataset['SibSp']+dataset['Parch']+1


# In[26]:

for dataset in full_data:
    dataset['is_alone']=0
    dataset.loc[dataset['family_size']==1,'is_alone']=1


# In[27]:

for dataset in full_data:
    dataset.drop(['SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)


# In[28]:

train.head()


# ### Imputing Embarked with median value

# In[29]:

for dataset in full_data:
    dataset['Embarked'].fillna('S',inplace=True)


# ### Changing Sex to nominal and OneHotEncoding(manually) on Embarked

# In[31]:

dict={'male':1,'female':0}
train['Sex']=train['Sex'].map(dict)
test['Sex']=test['Sex'].map(dict)


# In[30]:

a=pd.get_dummies(train[['Embarked','Title']])
b=pd.get_dummies(test[['Embarked','Title']])


# In[33]:

train=train.join(a)
test=test.join(b)


# In[34]:

train.drop(['Embarked','Title'],axis=1,inplace=True)
test.drop(['Embarked','Title'],axis=1,inplace=True)


# In[35]:

train.head()


# ### Categorizing Fare and Age

# In[36]:

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
train['CategoricalAge'] = pd.cut(train['Age'], 5)


# In[37]:

cf=pd.get_dummies(train['CategoricalFare'])
ca=pd.get_dummies(train['CategoricalAge'])
cf.columns=['F1','F2','F3','F4']
ca.columns=['A','B','C','D','E']


# In[38]:

train=train.join(ca)
train=train.join(cf)


# In[31]:

train.drop(['Age','Fare','CategoricalAge','CategoricalFare'],axis=1,inplace=True)


# In[39]:

test['CategoricalFare'] = pd.qcut(test['Fare'], 4)
test['CategoricalAge'] = pd.cut(test['Age'], 5)


# In[40]:

cf=pd.get_dummies(test['CategoricalFare'])
ca=pd.get_dummies(test['CategoricalAge'])
cf.columns=['F1','F2','F3','F4']
ca.columns=['A','B','C','D','E']
test=test.join(ca)
test=test.join(cf)


# In[41]:

test.drop(['Age','Fare','CategoricalAge','CategoricalFare'],axis=1,inplace=True)


# In[42]:

train.head()


# ## Modelling using Random Forest

# In[43]:

X=train[['Pclass','Sex','is_child','family_size','is_alone','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Rare',         'F1','F2','F3','F4','A','B','C','D','E']].values


# In[44]:

y=train['Survived'].values


# In[45]:

print X.shape, y.shape


# In[46]:

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(X,y,test_size=0.2)


# In[47]:

print xtrain.shape,xtest.shape


# In[48]:

from sklearn.ensemble import RandomForestClassifier


# In[49]:

tune={'n_estimators':[6,7,8,9,10,11,12,13,14,15],'max_depth':np.arange(5,15,1),'min_samples_split':[2,3,4]}


# In[50]:

from sklearn.grid_search import GridSearchCV
clf=RandomForestClassifier()
clf1=GridSearchCV(clf,param_grid=tune,cv=10,n_jobs=4)
clf1.fit(xtrain,ytrain)


# In[51]:

clf_new=clf1.best_estimator_


# In[52]:

clf_new.fit(xtrain,ytrain)


# In[53]:

from sklearn.metrics import accuracy_score
print accuracy_score(ytest,clf_new.predict(xtest))
print accuracy_score(ytrain,clf_new.predict(xtrain))


# In[54]:

test.shape


# In[55]:

X_test=test[['Pclass','Sex','is_child','family_size','is_alone','Embarked_C','Embarked_Q','Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Rare',         'F1','F2','F3','F4','A','B','C','D','E']].values


# In[56]:

clf_test=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=6, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=3, min_weight_fraction_leaf=0.0,
            n_estimators=14, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)


# In[57]:

clf_test.fit(X,y)


# In[58]:

y_pred=clf_test.predict(X_test)


# In[59]:

subm=test['PassengerId']
subm=pd.DataFrame(subm)
subm['Survived']=y_pred


# In[60]:

subm.shape


# In[100]:

subm.to_csv('submission.csv',sep=',')


# ### This submission scored 0.80861 LB score

# In[99]:




# In[ ]:



