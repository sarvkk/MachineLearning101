# %%
pip install scikit-learn

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# %%
"""
Data Ingession
"""

# %%
data=pd.read_csv("advertising.csv")
data.head()

# %%
data.describe()

# %%
data.info()

# %%
#check null
data.isnull().sum()

# %%
"""
EDA
"""

# %%
#check for outliers
sns.boxplot(x=data['TV'],showmeans=True)
plt.title("TV Spend Boxplot")
plt.show()

# %%
sns.boxplot(x=data['radio'],showmeans=True)
plt.title("Radio Spend Boxplot")
plt.show()

# %%
sns.boxplot(x=data['newspaper'],showmeans=True)
plt.title("Newspaper Spend Boxplot")
plt.show()

# %%
"""
Here, We can see that there are outliers in the data for newspapers.
"""

# %%
sns.boxplot(x=data['sales'],showmeans=True)
plt.title("Sales Spend Boxplot")
plt.show()

# %%
corr_matrix = data.corr()
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# %%
"""
Feature Engineering
"""

# %%
data['total_spend']=data["TV"]+data["newspaper"]+data['radio']

# %%
data.head()

# %%
data['tv_to_radio_ratio']=data["TV"]/data['radio']
data['newspaper_to_total_spend_ratio']=data["newspaper"]/data['total_spend']

data.head()

# %%
corr_matrix = data.corr()
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# %%
"""
Scaling
"""

# %%
data.columns

# %%
data[data['tv_to_radio_ratio']==np.inf]

# %%
data[data['newspaper_to_total_spend_ratio']==np.inf]

# %%
infinite_mask=(data['tv_to_radio_ratio']==np.inf)
data.loc[infinite_mask,'tv_to_radio_ratio']=0

# %%
data[data['tv_to_radio_ratio']==np.inf]


# %%
"""
Train Test Split
"""

# %%
X_train,X_test,y_train,y_test = train_test_split(data,data['sales'],test_size=.2,random_state=42)

# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# %%
"""
Modeling
"""

# %%
"""
Simple linear regression function
"""

# %%
model=LinearRegression()

# %%
model.fit(X_train, y_train)

# %%
y_pred= model.predict(X_test)

# %%
y_pred

# %%
"""
Evaluation
"""

# %%
mse = mean_squared_error(y_test,y_pred)
accuracy = r2_score(y_test,y_pred)

print("Mean Squared Error:",mse)
print("Accuracy:",accuracy)

# %%
"""
Categorical data is needed for accuracy score so we will use r2 score for now.
"""

# %%
