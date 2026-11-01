### Import packages in the virtual environment
import pandas as pd   ## search the package "pandas"
import statsmodels.api as sm    ## search the package "statsmodels"
import statsmodels.stats.api as sms   ## no need to install package
from statsmodels.compat import lzip  ## no need to install package
import matplotlib.pyplot as plt   ## search the package "matplotlib"
import seaborn as sns   ## search the package "seaborn"
import scipy.stats as scipy   ## search the package "scipy"
from scipy.stats import shapiro   ## no need to install package
import sys    ## no need to install package
import os     ## no need to install package
os.chdir(r"/home/vboxuser/CIS3920/Assignment 1 attached files Feb 28, 2026 402 PM")
### Change the address to your working folder
pd.set_option('display.max_columns', 500)
## Display all columns when printing results
mydata = pd.read_csv("kc_house_data.csv")
## Remember to include “r” first inside the bracket.

file = open('output2.txt','wt')
sys.stdout = file
## Open a text file and save all results into the text file

print(mydata.describe())
## Display summary statistics of the data

df = pd.DataFrame(mydata)
##Transform the dataset into two-dimensional, size-mutable, potentially heterogeneous tabular data

######################## Linear regression with only one independent variable
x = df['AgeOfHouse']
##Define the independent variables in the model
y = df['price']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model1 = sm.OLS(y, x).fit(cov_type='HC0')
##Fit the model using Heteroskedasticity-Robust Standard Errors
results_model1 = model1.summary()
print(results_model1)
##Output the results

df["predicted1"] = model1.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=df, x="price", y="predicted1")
plt.show()
##Plot the predicted value against the actual value of charges.



######################## Linear regression with only multiple independent variables

x = df[['bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'AgeOfHouse', 'sqfthouse']]
##Define the independent variables in the model
y = df['price']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model2 = sm.OLS(y, x).fit(cov_type='HC0')
##Fit the model using Heteroskedasticity-Robust Standard Errors
results_model2 = model2.summary()
print(results_model2)
##Output the results

df["predicted2"] = model2.predict(x)
##Caculate the predicted value of dependent variable based on the model



plt.show()
##Plot the predicted value against the actual value of charges.

##Regression diagnostic I: multicollinearity
corr_matrix = df.corr()
print(corr_matrix)

##Regression diagnostic 2: heteroscedasticity in the error term
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model2.resid, model2.model.exog)
print(lzip(names, test))

##Regression diagnostic 3: autocorrelation
##This is not a time-series dataset. Thus, the concern about autocorrelation is minimal.

##Regression diagnostic 4: model specification errors
##1)The model does not exclude any “core” variables.
##2)The model does not include superfluous variables.
##3)The functional form of the model is suitably chosen.
##4)There are no errors of measurement in the regressand and regressors.
##5)Outliers in the data, if any, are taken into account.

print(mydata.describe())

##6)The probability distribution of the error term is well specified.
residuals = model2.resid
print(shapiro(residuals))

file.close()
##Close the text file.


