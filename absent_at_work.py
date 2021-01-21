from fancyimpute import KNN
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



#####Settin the working directory###################################################3

os.chdir("E:\data_sci\decision_trees\absenteesm_at_work.xlsx")

###Loading the data########################################################################

emp_absent = pd.read_excel("data_set.xls",sheet_name = "Absenteeism_at_work")


###Exploratory Data Analysis#####################################################################

emp_absent.shape

##Data types of the variables###########################

emp_absent.dtypes

###Number of unique values present in each variable###############
emp_absent.nunique()



'''Transformation of data types
'''
emp_absent['ID'] = emp_absent['ID'].astype('category')

emp_absent['Reason for absence']=emp_absent['Reason for absence'].replace(0,20)

emp_absent['Reason for absence']=emp_absent['Reason for absence'].astype('category')

emp_absent['Month of absence']=emp_absent['Month of absence'].replace(0,np.nan)

emp_absent['Month of absence']=emp_absent['Month of absence'].astype('category')

emp_absent['Day of the week']=emp_absent['Day of the week'].astype('category')

emp_absent['Seasons']=emp_absent['Seasons'].astype('category')

emp_absent['Disciplinary failure']=emp_absent['Disciplinary failure'].astype('category')

emp_absent['Education']=emp_absent['Education'].astype('category')

emp_absent['Son']=emp_absent['Son'].astype('category')

emp_absent['Social drinker']=emp_absent['Social drinker'].astype('category')

emp_absent['Social smoker']=emp_absent['Social smoker'].astype('category')

emp_absent['Pet']=emp_absent['Pet'].astype('category')


###Make a copy of the dataframe

df = emp_absent.copy()

####From the EDA and problem categorizing the variables in two categories "Continuous" and "Categorical"

continuous_vars= ['Transportation expense','Distance from Residence to Work', 'Service time', 'Age' ,'Work load Average/day ',\
                  'Hit target', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']

categorical_vars = ['ID', 'Reason for absence', 'Month of absence', 'Seasons','Disciplinary failure',\
                    'Education','Son','Social drinker','Social smoker','Pet']

######Missing Value Analysis##################################

###Creating dataframe with number of missing values#####################

missing_val = pd.DataFrame(df.isnull().sum())


###Rename the columns##########################
missing_val = missing_val.rename(columns = {'index':'variables', 0:'Missing_perc'})

###Calculate the percentage of Missing Values##########################

missing_val['Missing_perc'] = (missing_val['Missing_perc']/len(df))*100

###Sort the rows according to decreasing missing percentage####################

missing_val = missing_val.sort_values('Missing_perc',ascending = False).reset_index(drop = True)


'''
Imputing the missing values

'''

###Actual value = 31
#Mean = 26.68
#Median = 25.0

print(df["Body mass index"].iloc[1])

##Create a missing value
df["Body mass index"].iloc[1]= np.nan


###Apply the KNN imputation algorithm
df = pd.DataFrame(KNN(k=3).fit_transform(df),columns = df.columns)
print(df["Body mass index"].iloc[1])

##Round the values of the categorical attributes  #

for i in categorical_vars:
    df.loc[:,i] = df.loc[:,i].round()
    df.loc[:,i] = df.loc[:,i].astype('category')
   
   
##Checking for missing values###################
   
df.isnull().sum()

#################Distribution of data using graphs#################################

##Check the bar graph of categorical data using factorplot

sns.set_style("whitegrid")
sns.factorplot(data=df, x = 'Reason for absence', kind = 'count', size = 4, aspect = 2)
sns.factorplot(data=df, x = 'Seasons', kind = 'count', size = 4, aspect = 2)
sns.factorplot(data=df, x = 'Education', kind = 'count', size = 4, aspect = 2)
sns.factorplot(data=df, x = 'Disciplinary failure', kind = 'count', size = 4, aspect = 2)

###Check the distribution of numerical data using histogram

plt.hist(data= df, x = 'Weight', bins = 'auto', label = 'weight')
plt.xlabel('weight')
plt.title('Weight Distribution')

plt.hist(data= df, x = 'Age', bins = 'auto', label = 'weight')
plt.xlabel('Age')
plt.title('Age Distribution')

fig,axs = plt.subplots(1,2, figsize = (15,5), sharey= True)
axs[0].scatter(data=df, x = 'Age', y = 'Absenteeism time in hours')
axs[1].scatter(data = df, x = 'Weight', y = 'Absenteeism time in hours', color = 'red')
fig.suptitle('Scatter plot for Age and Weight')
plt.xlabel('Age and Weight')
plt.ylabel('Absenteeism in hours')

##Check for outliers in data using boxplot

sns.boxplot(data = df[['Absenteeism time in hours','Body mass index','Height','Weight']])
fig = plt.gcf()
fig.set_size_inches(8,8)

sns.boxplot(data = df[['Hit target','Service time','Age','Transportation expense']])
fig = plt.gcf()
fig.set_size_inches(8,8)

#df = df.rename(columns = {'Work load Average/day ':'Avg_work_load_per_day'})
for i in continuous_vars:
    q75,q25 = np.percentile(df[i],[75,25])##75th and 25th percentile you obtained
    iqr = q75 - q25###Calculating the IQR
    minimum = q25 - (iqr*1.5)
    maximum = q75 + (iqr*1.5)
   
    ##Replace the outliers with NA
    df.loc[df[i] < minimum,i] = np.nan
    df.loc[df[i] > maximum,i] = np.nan
   
    ##Impute the missing values with KNN imputation algorithm
    df = pd.DataFrame(KNN(k = 3).fit_transform(df),columns = df.columns)
    df.isnull().sum()

sns.boxplot(data = df[['Absenteeism time in hours','Body mass index','Height','Weight']])
fig = plt.gcf()
fig.set_size_inches(8,8)

sns.boxplot(data = df[['Hit target','Service time','Age','Transportation expense']])
fig = plt.gcf()
fig.set_size_inches(8,8)

'''
Feature Selection
'''

df_corr = df.loc[:,continuous_vars]

###Check for multicollinearity using correlation graph
#Set the width and height of the plot

f, ax = plt.subplots(figsize = (10,10))

##Generate correlation matrix
corr = df_corr.corr()

##Plot this using heatmap in seaborn
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), \
            square = True, annot = True, ax = ax)
plt.plot()


##Variable Reduction

to_drop= ['Weight']
df = df.drop(to_drop, axis = 1)


##Update the continuous and categorical variables
continuous_vars.remove('Weight')

clean_data = df.copy()


'''
Feature Scaling
'''

##Normality check
for i in continuous_vars:
    if i == 'Absenteeism time in hours':
        continue
    sns.distplot(df[i],bins = 'auto')
    plt.title("Checking distribution for variable "+str(i))
    plt.ylabel("Density")
    plt.show()
   
##Normalization of the continuous variables
for i in continuous_vars:
    if i == 'Absenteeism time in hours':
        continue
    df[i] = (df[i] - df[i].min())/(df[i].max()- df[i].min())
   
'''
Machine Learning Models

'''

##Create dummy variables for factor variables

df = pd.get_dummies(data = df, columns = categorical_vars)

##Copying dataframe

df1 = df

df.columns.get_loc('Absenteeism time in hours')

####Split the data into train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,df.columns != 'Absenteeism time in hours'], \
                                                    df.iloc[:,9], test_size = 0.20,\
                                                    random_state = 1)



###Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

###Build a decision tree regressor

dt_model = DecisionTreeRegressor(random_state=1).fit(X_train,y_train)

##Predict for test cases
dt_predictions = dt_model.predict(X_test)

##Create a dataframe for actual and predicted values

df_dt = pd.DataFrame({'actual':y_test,'pred':dt_predictions})

###define a function to calculate RMSE

def RMSE(y_actual, y_predicted):
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse

###Calculate the RMSE and R-Squared values
print("Root Mean Squared Error:"+ str(RMSE(y_test, dt_predictions)))
print("R^2 Score(Coefficient of determination) = "+str(r2_score(y_test,dt_predictions)))

###Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

###Build random forest using RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=500, random_state=1).fit(X_train,y_train)

##Predict for test cases

rf_predictions = rf_model.predict(X_test)

##Create data frame for actual and predicted values

df_rf = pd.DataFrame({'actual':y_test,'pred':rf_predictions})

###Calculate the RMSE and R-Squared values
print("Root Mean Squared Error:"+ str(RMSE(y_test, rf_predictions)))
print("R^2 Score(Coefficient of determination) = "+str(r2_score(y_test,rf_predictions)))


###Linear Regression

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression().fit(X_train,y_train)

##Predict for test cases

lr_predictions = lr_model.predict(X_test)


##Create data frame for actual and predicted values

df_lr = pd.DataFrame({'actual':y_test,'pred':lr_predictions})

###Calculate the RMSE and R-Squared values
print("Root Mean Squared Error:"+ str(RMSE(y_test, lr_predictions)))
print("R^2 Score(Coefficient of determination) = "+str(r2_score(y_test,lr_predictions)))
