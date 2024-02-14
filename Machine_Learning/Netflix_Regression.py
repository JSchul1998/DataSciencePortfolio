import pandas as pd
import numpy as np
from datetime import datetime, date 
import matplotlib.pyplot as plt

##Machine Learning Imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns

##Import Netflix Data
df = pd.read_csv('Desktop/Netflix_Userbase.csv')

##Remove Plan Duration as all columns are 1 month, so this is redundant 
df = df.drop(['Plan Duration', 'Last Payment Date'],axis=1)

##Create a new column which takes Join Date and calculates Lifetime of Account
def age(born): 
    ##Enter format of date being shown currently
    born = datetime.strptime(born, "%Y-%m-%d").date() 
    today = date.today() 
    return today.year - born.year - ((today.month,  
                                      today.day) < (born.month,  
                                                    born.day)) 

df['Account Lifetime'] = df['Join Date'].apply(age) 
df.pop('Join Date')

##As this dataset contains dates in the future, remove negative account lifetimes
df = df[df['Account Lifetime'] > 0]

##Make new column which calculates the cumulative customer revenue
def Rev(frame):
    return frame[0]*12*frame[1]
# This notation allows function to be applied to arguments from multiple columns of data 
df['Revenue'] = df[['Monthly Revenue','Account Lifetime']].apply(Rev, axis=1)

print(df.head(10))


############We want to first figure out how revenue is divided amongst age groups, 
############gender, country, and devices
gender_group = df.groupby(by="Gender").sum()
#ax = gender_group.plot.bar(y='Revenue')
#plt.show()

country_group = df.groupby(by="Country").sum()
#ax = country_group.plot.bar(y='Revenue')
#plt.show()

ax = sns.pairplot(df[['Revenue','Account Lifetime','Gender','Age']],hue="Gender")
plt.show()

##Calculate the mean, standard deviation, max, and min revenue for those aged 31 and older AND Female
##Note that can specify multiple arguments (AND) with &, | for (OR), ~ for (NOT)
df_new = df[(df['Gender'] == 'Female') & (df['Age'] >= 31)]

print('Mean Revenue', df[(df['Gender'] == 'Female') & (df['Age'] >= 31)]['Revenue'].mean())
print('STD Revenue', df[(df['Gender'] == 'Female') & (df['Age'] >= 31)]['Revenue'].std())
print('Max Revenue', df[(df['Gender'] == 'Female') & (df['Age'] >= 31)]['Revenue'].max())
print('Min Revenue', df[(df['Gender'] == 'Female') & (df['Age'] >= 31)]['Revenue'].min())

##Show a boxplot of revenue 
boxplot = df.boxplot(column=['Revenue'])  
plt.show()

##Clearly, there are outliers. Let's remove them and the associated rows.
def Quartiles(frame,val):
    return frame.quantile(val)
firstQ = Quartiles(df['Revenue'],0.25)
thirdQ = Quartiles(df['Revenue'],0.75)
IQR = thirdQ-firstQ
Outliers = firstQ - 1.5*IQR, thirdQ + 1.5*IQR

##Remove Outliers and plot new pairplot
df = df[(df['Revenue'] < Outliers[1]) & (df['Revenue'] > Outliers[0])]
ax = sns.pairplot(df[['Revenue','Account Lifetime','Gender','Age']],hue="Gender")
plt.show()

##There we go! Now we should be all set to begin the regression analysis
##################We then want to see how much total revenue
###################regresses with these variables

##First, dummify all non-quantitative variables and separare X and y variables
print(df.columns)
X = df.drop(['User ID','Monthly Revenue','Revenue'],axis=1)
y = df.pop('Revenue')
print(y.head())
X = pd.get_dummies(X, columns=['Gender','Country','Subscription Type','Device'])
print(X.head())



##We need to pre-process the predictor data before running through the regression algorithm
pipeline = Pipeline([
    ##This first transform ensures data become normally distributed
    ('yeo_johnson_transform', PowerTransformer(method='yeo-johnson')),
    ##These then standardize X values to balance variable weights
    ('min_max_scaler', MinMaxScaler()),
    ('std_scaler', StandardScaler()),
])

# Apply the pipeline to only quantitative columns of dataframe
X[['Age','Account Lifetime']] = pipeline.fit_transform(X[['Age','Account Lifetime']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)
print(X_test)


##Now make a dictionary for the models we will use for regression (we will use 6 different ones)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Ridge': Ridge(),
    'KNeighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor()
}

mae_scores = {}
mse_scores = {}
r2_scores = {}
best_model = None
best_score = float('-inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae_scores[name] = mean_absolute_error(y_test, y_pred)
    mse_scores[name] = mean_squared_error(y_test, y_pred)
    r2_scores[name] = r2_score(y_test, y_pred)
    
    # Print the model's parameters and metrics
    print(f'Parameters for {name}: {model.get_params()}')
    print(f'{name}: MAE={mae_scores[name]:.2f}, MSE={mse_scores[name]:.2f}, R2={r2_scores[name]:.2f}')
    print()
    
    # Update the best model based on lowest MAE
    if r2_scores[name] > best_score:
        best_model = model
        best_score = r2_scores[name]

print(f'Best Model: {best_model}')



# Plotting results
plt.figure(figsize=(10, 6))

# MAE scores
plt.subplot(1, 3, 1)
plt.barh(list(mae_scores.keys()), list(mae_scores.values()))
plt.xlabel('Mean Absolute Error')
plt.title('Model Comparison - MAE')
# MSE scores
plt.subplot(1, 3, 2)
plt.barh(list(mse_scores.keys()), list(mse_scores.values()))
plt.xlabel('Mean Squared Error')
plt.title('Model Comparison - MSE')
# R2 scores
plt.subplot(1, 3, 3)
plt.barh(list(r2_scores.keys()), list(r2_scores.values()))
plt.xlabel('R-squared')
plt.title('Model Comparison - R-squared')

plt.tight_layout()
plt.show()


# Additional plots for comparing model predictions
plt.figure(figsize=(12, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Scatter plot for predictions vs. actual values
    plt.scatter(y_test, y_pred, label=name, alpha=0.7)

plt.xlabel('Actual Positive Increase')
plt.ylabel('Predicted Positive Increase')
plt.title('Actual vs. Predicted Positive Increase for Different Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
