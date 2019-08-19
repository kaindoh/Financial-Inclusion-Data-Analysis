# Financial-Inclusion-Data-Analysis
## imported necessary files

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

## loading files and gettting the preview of the data using pandas

## creating hypothesis
Ho: An Individual has a bank account
H1:An individual has no bank account

## cleaning dataset
making the columns to be uniform
Removing duplicates
Removing missing values
Removing outliers

## visualization
I plotted the following
scatter plots
bar plots
pie charts
line graphs

## Creating a model
I created a model using RandomForestclassifier and logistic reggresion

## reduction method
Principle component Analysis
Linear reggresion
Linear Discriminant Analysis

## Conclusion 
The best reduction method was LDA with accuracy of 88.89% predictions
Histogram

