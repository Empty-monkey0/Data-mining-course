### Import packages in the virtual environment
import pandas as pd   ## search the package "pandans"
import statsmodels.api as sm    ## search the package "statsmodels"
import statsmodels.stats.api as sms   ## no need to install package
from statsmodels.compat import lzip   ## no need to install package
import matplotlib.pyplot as plt    ## search the package "matplotlib"
import seaborn as sns     ## search the package "seaborn"
import scipy.stats as scipy    ## search the package "scipy"
from scipy.stats import shapiro   ## no need to install package
import sys   ## no need to install package
import openpyxl as xl
import os   ## no need to install package


os.chdir(r"/home/vboxuser/CIS3920/Lab 1.2 attached files Mar 10, 2026 236 PM")
##Change the address to your working folder

pd.set_option('display.max_columns', 500)
## Display all columns when printing results

mydata = pd.read_excel("Group26crashes_clean (1).xlsx", header = 3)
## Remember to include “r” first inside the bracket.

file = open('Projectoutput.txt','wt')
sys.stdout = file
## Open a text file and save all results into the text file

print(mydata.describe())
## Display summary statistics of the data

df = pd.DataFrame(mydata)
##Transform the dataset into two-dimensional, size-mutable, potentially heterogeneous tabular data

######################## Linear regression with only one independent variable
x = df['is_ford']
##Define the independent variables in the model
y = df['fatal']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model1 = sm.Logit(y, x).fit()
##Fit the model
results_model1 = model1.summary()
print(results_model1)
##Output the results



######################## Linear regression with only multiple independent variables

x = df[['is_ford', 'vehicle_age', 'is_suv_truck', 'is_commercial', 'male_driver', 'license_invalid', 'impaired_state', 'illegal_behavior', 'night_crash']]
##Define the independent variables in the model
y = df['fatal']
##Define the dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model2 = sm.Logit(y, x).fit()
##Fit the model
results_model2 = model2.summary()
print(results_model2)
##Output the results


file.close()
##Close the text file.


