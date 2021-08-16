# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:54:21 2020

@author: 79456
"""

# Linear Discriminant Analysis (LDA)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
df = pd.read_csv('abalone.csv')
X = df.iloc[:, 1:7].values
y = df.iloc[:, 8].values

df.columns = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']
# To display the top 5 rows
df.head(5)

# To display the bottom 5 rows
df.tail(5)

# Checking the data type
df.dtypes

# Renaming the column names
df = df.rename(columns={'Diameter':'Diam','Sex' : 'Gender','Viscera weight' : 'Viscera_weight','Shell weight' : 'Shell_weight','Shucked weight' : 'Shucked_weight'})
df.head(5)

# Dropping irrelevant columns
df = df.drop(['Viscera_weight','Shucked_weight'],axis=1)
df.head(5)

# Total number of rows and columns
df.shape

# Rows containing duplicate data
duplicate_rows_df = df[df.duplicated()]
print('number of duplicate rows: ', duplicate_rows_df.shape)

# Used to count the number of rows before removing the data
df.count()

#checking for non-null values
df.info()

# Dropping the duplicates 
df = df.drop_duplicates()
df.head(5)
df.count()

# Finding the null values.
print(df.isnull().sum())               

# Dropping the missing values.
df = df.dropna() 
df.count()

# After dropping the values
print(df.isnull().sum())

#Detecting Outliers
sns.boxplot(x=df['Length'])
sns.boxplot(x=df['Diam']) 
sns.boxplot(x=df['Height']) 
sns.boxplot(x=df['Whole weight']) 
sns.boxplot(x=df['Rings']) 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df < (Q1-1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
df.Shell_weight.describe()

######Univariate Analysis
######################

#Categorical Unordered Univariate Analysis:
'''
An unordered variable is a categorical variable that has no defined order.
 
'''
# Let's calculate the percentage of each category.
df.Gender.value_counts(normalize=True)
#plot the bar graph of percentage job categories
df.Gender.value_counts(normalize=True).plot.barh()
plt.show()

#Categorical Ordered Univariate Analysis:
'''
Ordered variables are those variables that have a natural rank of order. 
'''
#calculate the percentage of each education category.
df.Rings.value_counts(normalize=True)

#plot the pie chart of education categories
df.Rings.value_counts(normalize=True).plot.pie()
plt.show()

'''
This is how we analyze univariate categorical analysis. 
If the column or variable is of numerical then we’ll analyze by calculating its mean,
 median, std, etc. 
We can get those values by using the describe function.
'''
df.Shell_weight.describe()




######Bivariate Analysis

'''
a) Numeric-Numeric Analysis:

Analyzing the two numeric variables from a dataset is known as numeric-numeric analysis. 
We can analyze it in three different ways.
'''



# Plotting a Histogram
df.Gender.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title('Number of values /Gender'  )
plt.ylabel('Num of Values')
plt.xlabel('Gender')

df.Rings.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title('Number of values /Rings'  )
plt.ylabel('Num of Values')
plt.xlabel('Rings')

df.Shell_weight.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title('Number of values /Rings'  )
plt.ylabel('Num of Values')
plt.xlabel('Shell_weight')

#plot the pie chart of  categories
df.Rings.value_counts(normalize=True).plot.pie()
plt.show()

#plot the pie chart of  categories
df.Gender.value_counts(normalize=True).plot.pie()
plt.show()



###Scatter Plot

# Plotting a scatter plot
#we see trend line between Rings and shell weight
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['Rings'], df['Shell_weight'])
ax.set_xlabel('Rings')
ax.set_ylabel('Shell_weight')
plt.show()

#plot the scatter plot 
plt.scatter(df.Rings,df.Shell_weight)
plt.show()

#plot the scatter plot 
df.plot.scatter(x="Rings",y="Shell_weight")
plt.show()


#Pair Plot
#plot the pair plot 
sns.pairplot(data = df, vars=['Rings','Shell_weight','Gender'])
plt.show()


# Correlation Matrix
'''
Since we cannot use more than two variables as x-axis and y-axis in Scatter and Pair Plots,
 it is difficult to see the relation between three numerical variables in a single graph. 
In those cases, we’ll use the correlation matrix.
'''
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c,cmap='BrBG',annot=True)
c

#######Numeric - Categorical Analysis
'''
Analyzing the one numeric variable and one categorical variable from a dataset is
 known as numeric-categorical analysis.
 We analyze them mainly using mean, median, and box plots.
'''

#mean value shell weight / rings
df.groupby('Shell_weight')['Rings'].mean()

#median value
df.groupby('Shell_weight')['Rings'].median()

#plot the box plot
sns.boxplot(df.Rings, df.Shell_weight)
plt.show()

sns.boxplot(df.Gender, df.Shell_weight)
plt.show()

#####Categorical — Categorical Analysis

#create  numerical data type
df['Rings'] = np.where(df.Rings >= 10)
df.response_rate.value_counts()

#plot the bar graph  with average value
df.groupby('Rings')['Shell_weight'].mean().plot.bar()
plt.show()

df.groupby('Gender')['Shell_weight'].mean().plot.bar()
plt.show()


#### Multivariate Analysis

result = pd.pivot_table(data=df, index='Rings', columns='Gender',values='Shell_weight')
print(result)

#create heat map of education vs marital vs response_rate
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()



print(df['Shell_weight'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['Shell_weight'], color='g', bins=100, hist_kws={'alpha': 0.4});



###Numerical data distribution

list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()

df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)







              
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()



dataset["Diameter"] = dataset["Diameter"].astype(float)
dataset["Rings"] = dataset["Rings"].astype(int)
height = dataset.iloc[3:20,3]
rings = dataset.iloc[3:20:,8]
sex = dataset.iloc[1:20:,0]

dataset.info(verbose=True)
dataset_droped = dataset.drop('Sex',axis = 1)
dataset_dic = {"Height" : dataset_1,"Rings" : dataset_2}
df = pd.DataFrame (dataset_dic, columns = ['Height','Rings'])
dataset.rename_axis("RINGS", axis=8)




sns.histplot(data=dataset, x=dataset["Diameter"], hue=dataset["Rings"], multiple="stack")
sns.kdeplot(data=dataset,x="Height", y="Rings", multiple="stack")

sns.kdeplot(data = dataset_droped, x="Height", y = "Rings")
sns.kdeplot(data = dataset_droped,x=sex)
sns.lmplot(x = "Sex", y= "Rings", data = dataset)
sns.kdeplot(dataset.Diameter,dataset.Rings,shade=True,shade_lowest=False,cmap='Reds')
sns.kdeplot(dataset.Diameter,dataset.Rings,cmap='Reds')

f, axes  = plt.subplots(1,2)
sns.kdeplot(dataset.Diameter,dataset.Rings,cmap='Reds',ax = axes[0])
sns.kdeplot(dataset.Shell_weight,dataset.Rings,cmap='Reds',ax = axes[1])

for i in dataset_1:
    if i  == i.find("M"):
        print(i)
        
try:
    list1 = map (float, dataset_1) 
    list2 = map(int, dataset_2)
except ValueError:
    print ("Error")

sns.kdeplot(data=dataset,x=list1, hue=list2, multiple="stack")    
       