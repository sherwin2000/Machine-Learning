#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[2]:


df = pd.read_csv(r'D:\ML\homeprices.csv.txt')
df


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')


# In[10]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(df[['area']] ,df.price)


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area , reg.predict(df[['area']]), color='blue')


# In[17]:


reg.predict([[3000]]) # to predict the price of house when area is 3000


# In[19]:


reg.coef_  #coefficient in the equation y=mx+b , i.e, 'm'


# In[20]:


reg.intercept_ #intercept in the equation y=mx+b , i.e, 'b'


# In[22]:


d = pd.read_csv(r'D:\ML\areas.csv.txt')
d


# In[23]:


price = reg.predict(d) # to predict the prices based on area


# In[25]:


d['price'] = price


# In[26]:


d


# In[28]:


d.to_csv("D:\ML\price.csv")


# In[ ]:


### Multivariate linear regression


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[4]:


df = pd.read_csv(r"D:\ML\Multivariate linear regression\homeprices.csv.txt")
df


# In[5]:


import math
median_bedroom = math.floor(df.bedrooms.median())
median_bedroom


# In[6]:


df.bedrooms = df.bedrooms.fillna(median_bedroom)
df


# In[7]:


reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']], df.price) ## first write independent variables in double[[]], then write the target variable


# In[8]:


reg.predict([[3000,4,40]]) ## we took the input as area, bedrooms and age


# In[ ]:


### Joblib


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[15]:


df = pd.read_csv(r'D:\ML\homeprices.csv.txt')
df


# In[16]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(df[['area']] ,df.price)


# In[40]:


from sklearn.externals import joblib


# In[41]:


joblib.dump(reg,'model')


# In[42]:


model = joblib.load('model')


# In[43]:


model.predict([[5000]])


# In[ ]:


# Dummy variable method


# In[44]:


import pandas as pd
import numpy as np


# In[45]:


df = pd.read_csv(r"D:\ML\One Hot Encoding\homeprices.csv.txt")
df


# In[46]:


dummies = pd.get_dummies(df.town)
dummies


# In[47]:


merged = pd.concat([df,dummies],axis='columns')
merged


# In[50]:


final = merged.drop(['town'], axis='columns')
final


# In[57]:





# In[65]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[52]:


x = final.drop('price', axis = 'columns')
x


# In[53]:


y = final.price
y


# In[58]:


model.fit(x,y)


# In[60]:


model.predict([[3400,0,0,1]]) # 3400 sqr ft home in west windsor 1) monroe township	 2)robinsville 3)west windsor


# In[61]:


model.score(x,y) ## to check how much accurate our model is


# In[ ]:


## One Hot Encoding


# In[88]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[89]:


dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle


# In[90]:


x = dfle[['town','area']].values ## 2-d array
x


# In[91]:


y = dfle.price.values
y


# In[92]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')


# In[93]:


x = ct.fit_transform(x)
x


# In[94]:


x = x[:,1:] # to remove the first column
x


# In[95]:


model.fit(x,y)


# In[96]:


model.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor


# In[ ]:


# Training And Testing Available Data


# In[ ]:



# We have a dataset containing prices of used BMW cars.
# We are going to analyze this dataset and build a prediction function that can predict a price
# by taking mileage and age of the car as input. We will use sklearn train_test_split method to 
# split training and testing dataset


# In[111]:


import pandas as pd
df = pd.read_csv(r"D:\ML\Training and Testing\carprices.csv.txt")
df.head()


# In[112]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[113]:


plt.scatter(df['Mileage'],df['Sell Price($)'])
plt.xlabel('Mileage')
plt.ylabel('Selling Price ($)')
plt.show()


# In[114]:


plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
plt.xlabel('Age (in years)')
plt.ylabel('Selling Price ($)')
plt.show()


# In[117]:


x = df[['Mileage','Age(yrs)']]  # independent variables


# In[118]:


y = df['Sell Price($)'] # dependent variable


# In[119]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[120]:


x_train


# In[121]:


x_test


# In[122]:


y_train


# In[123]:


y_test


# In[125]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)


# In[126]:


clf.predict(x_test)


# In[127]:


clf.score(x_test, y_test)


# In[ ]:


## Logistic Regression


# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"D:\ML\Logistic Regression\insurance_data.csv.txt")
df.head()


# In[4]:


plt.scatter(df.age,df.bought_insurance,marker='+',color='red')
plt.show()


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)


# In[7]:


x_test


# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[9]:


model.fit(x_train, y_train)


# In[10]:


x_test


# In[13]:


y_predicted = model.predict(x_test)


# In[15]:


model.predict_proba(x_test)


# In[16]:


model.score(x_test,y_test)


# In[17]:


y_predicted


# In[18]:


x_test


# In[28]:


model.predict([[76]]) # 70 year old person will take the insurance, so answer is 1


# In[ ]:


import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[20]:


def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y


# In[21]:


age = 35
prediction_function(age) ##0.485 is less than 0.5 which means person with 35 age will not buy insurance


# In[23]:


age = 43
prediction_function(age) ## 0.568 is more than 0.5 which means person with 43 will buy the insurance


# In[ ]:


## Logistice Regression (Multiclass)


# In[8]:


from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[9]:


plt.gray() 
for i in range(5):
    plt.matshow(digits.images[i])


# In[10]:


dir(digits)


# In[11]:


digits.data[0]


# In[12]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2)


# In[15]:


model.fit(x_train, y_train)


# In[17]:


model.score(x_test, y_test)


# In[18]:


model.predict(digits.data[0:5])


# In[ ]:





# In[23]:


# Decision tree


# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("D:\ML\Decision Tree\salaries.csv.txt")
df.head()


# In[3]:


inputs = df.drop('salary_more_then_100k',axis='columns')


# In[4]:


target = df['salary_more_then_100k']


# In[5]:


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[6]:


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])


# In[7]:


inputs


# In[8]:


inputs_n = inputs.drop(['company','job','degree'],axis='columns')


# In[9]:


inputs_n


# In[10]:


target


# In[11]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[12]:


model.fit(inputs_n, target, )


# In[13]:


model.score(inputs_n,target)


# In[14]:


model.predict([[2,1,0]])


# In[ ]:


## SVm


# In[10]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[11]:


iris.feature_names


# In[12]:


iris.target_names


# In[13]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[14]:


df['target'] = iris.target
df.head()


# In[15]:


iris.target_names # 0 means setosa, 1 means versicolor, 2 means virginica


# In[16]:


df[df.target==1].head()


# In[17]:


df[df.target==2].head()


# In[18]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[19]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')

plt.show()


# In[22]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')

plt.show()


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x = df.drop(['target','flower_name'], axis='columns')
y = df.target


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[26]:


len(x_train)


# In[27]:


len(x_test)


# In[31]:


from sklearn.svm import SVC
model = SVC(kernel='linear')


# In[32]:


model.fit(x_train, y_train)


# In[33]:


model.score(x_train, y_train)


# In[ ]:


## Random forest Algorithm


# In[80]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[81]:


dir(digits)


# In[82]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[83]:


plt.gray() 
for i in range(4):
    plt.matshow(digits.images[i])


# In[84]:


df = pd.DataFrame(digits.data)
df.head()


# In[85]:


df['target'] = digits.target


# In[86]:


df[0:12]


# In[87]:


x = df.drop('target',axis='columns')
y = df.target


# In[88]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[89]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(x_train, y_train)


# In[90]:


model.score(x_test, y_test)


# In[91]:


y_predicted = model.predict(x_test)


# In[92]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[93]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.show()


# In[1]:


## KFold Cross validation


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)


# In[4]:


lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[5]:


svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)


# In[6]:


rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)


# In[7]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf


# In[8]:


for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)


# In[9]:


from sklearn.model_selection import cross_val_score


# In[10]:


cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), digits.data, digits.target,cv=3)


# In[11]:


cross_val_score(SVC(gamma='auto'), digits.data, digits.target,cv=3)


# In[12]:


cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)


# In[13]:


scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),digits.data, digits.target, cv=10)
np.average(scores1)


# In[1]:


## Clustering With K Means


# In[2]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r"D:\ML\Clustering with K means\income.csv.txt")
df.head()


# In[4]:


plt.scatter(df['Age'],df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')


# In[5]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted


# In[6]:


df['cluster']=y_predicted
df.head()


# In[7]:


km.cluster_centers_


# In[8]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()


# In[9]:


scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


# In[10]:


df.head()


# In[11]:


plt.scatter(df.Age,df['Income($)'])


# In[12]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted


# In[13]:


df['cluster']=y_predicted
df.head()


# In[14]:


km.cluster_centers_


# In[15]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.legend()


# In[16]:


## Elbow Plot


# In[17]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)


# In[18]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()


# In[ ]:





# In[19]:


# Naive Bayes Tutorial Part 1: Predicting survival from titanic crash


# In[20]:


import pandas as pd


# In[21]:


df = pd.read_csv(r"D:\ML\Part 1\titanic.csv.txt")
df.head()


# In[22]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[23]:


inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[24]:


dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)


# In[25]:


inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)


# In[26]:


inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(3)


# In[27]:


inputs.columns[inputs.isna().any()] # to display the column which has "Na" Value


# In[28]:


inputs.Age[:10]


# In[29]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())  # to fill the Na values with mean values
inputs.head()                               


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)


# In[31]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[32]:


model.fit(X_train,y_train)


# In[33]:


model.score(X_test,y_test)


# In[34]:


X_test[0:10]


# In[35]:


y_test[0:10]


# In[36]:


model.predict(X_test[0:10])


# In[37]:


model.predict_proba(X_test[:10])


# In[38]:


from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)


# In[39]:


# Naive Bayes Part 2


# In[40]:


import pandas as pd


# In[42]:


df = pd.read_csv(r"D:\ML\Part 2\spam.csv.txt")
df.head()


# In[43]:


df.groupby('Category').describe()


# In[44]:


df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam)


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]


# In[47]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


# In[48]:


emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)


# In[49]:


X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)


# In[ ]:


# SKlearn Pipeline


# In[50]:


from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])


# In[51]:


clf.fit(X_train, y_train)


# In[52]:


clf.score(X_test,y_test)


# In[53]:


clf.predict(emails)


# In[ ]:




