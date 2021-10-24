# REAL STATE HOUSING PRICE PREDICTION

import pandas as pd

housing = pd.read_csv("housing_data.csv")
import  numpy as np

# print(housing.head())  # print top 5 rows

# print(housing.info())  # give the info about out data entry

# print(housing['CHAS'].value_counts())   # it will give the count of all values of any particular category data

# print(housing.describe())   # it will describe all our data

import matplotlib.pyplot as plt

housing.hist(bins = 50,figsize = (20,15))  #histogram for our data representing
# print(plt.show())

# TRAIN TEST SPLITTING --------->
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing , test_size = 0.2,random_state = 42)  #normally use random_state = 42,
# we are using random_state for fixing the shuffled value of our data set
# print("Rows for testing -> ",len(train_set),"\nRows for training -> ",len(test_set))

# if we want to fix the ratio of any particular feautre in train_set and test_set
# then we can use stratified shuffle slit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 2, test_size = 0.2, random_state = 42)  #n_split will be according to no. of which categories we want to set shufflespit
for train_index, test_index in split.split(housing,housing['CHAS'],housing['RAD']):  # here we have added two features so, n_split = 2
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Now, we will store the copy of strat_train_set in housing data
housing = strat_train_set.copy()

# print(strat_test_set['CHAS'].value_counts())
# print(strat_train_set['CHAS'].value_counts())
# print(housing.info())
# print(strat_test_set['RAD'].value_counts())
# print(strat_train_set['RAD'].value_counts())


# Now the ratio of 1 and 0 in both train_set and test_set will be same 
# 0    95
# 1     7
# Name: CHAS, dtype: int64
# 0    376
# 1     28
# Name: CHAS, dtype: int64

# LOOKING FOR A CORRELATION--------------->
# if we want to know the strongly affected features availabla into our dataset

corr_matrix = housing.corr()
rel = corr_matrix['MEDV'].sort_values(ascending=False)
# print(rel)
# MEDV       1.000000
# RM         0.695360     # According to this data RM is the strongly affected feature
# ZN         0.360445     # on which price will have strongly positive correlation
# B          0.333461
# DIS        0.249929
# CHAS       0.175260
# AGE       -0.376955
# RAD       -0.381626
# CRIM      -0.388305
# NOX       -0.427321
# TAX       -0.468536
# INDUS     -0.483725
# PTRATIO   -0.507787
# LSTAT     -0.737663

# if we want to plot the graph of correlation feature-->
from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
# print(scatter_matrix(housing[attributes] , figsize = (12,8)))

# TRY OUT THE ATTRIBUTES COMBINATION------------>
# It means, if want to add new feature by the combination of two or more attributes then -->

housing["TAXRM"] = housing["TAX"]/housing["RM"]
# TAXRM has been added in our housing data
# print(housing.head())
# WE can check the correlation of this feeature
# corr_matrix = housing.corr()
# rel = corr_matrix['MEDV'].sort_values(ascending=False)
# print(rel)

# Plotting graph for TAXRM-->
plot = housing.plot(kind ="scatter" , x = "TAXRM" , y = "MEDV",alpha = 0.8)   # alpha is used for the darkness of point of our graph
# print(plot)

housing = strat_train_set.drop("MEDV" , axis = 1)
housing_labels = strat_train_set["MEDV"].copy()  # here we will separate to housing and housing labels

# MISSING DATA------------>
# To take care of missing attributes, you have three options:\n",
    #     1. Get rid of the missing data points\n",  (rid of data means, removove the data)
    #     2. Get rid of the whole attribute\n",
    #     3. Set the value to some value(0, mean or median)"
    # First two option can affect our prediction so, we will use third option

# option 1-->
a = housing.dropna(subset=["RM"])    # oringinal dataframe will be unchanged
# print(a.shape)  # we can check the shape and size of RM
 
# Option 2-->
b = housing.drop("RM",axis = 1) # Note that there will be no "RM" column
# original housing dataframe will be unchanged
# if we want to change then use housing variable instead of a and b
# print(b)

#  Option 3 -->

median = housing["RM"].median()
c = housing["RM"].fillna(median)    # Note that original dataframe will not be changed
# print(c)

# Now, we will do the third option using sklearn and fit into our original dataframe
# print(housing.info())
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)    # calculate some parameters
x = imputer.transform(housing)   # this is a numpy array and transform data in this

# we will create a new dataframe for transformed dataset in which all missing values will be fit by the median values
housing_tr = pd.DataFrame(x , columns = housing.columns)
# print(housing_tr["RM"].describe())  



# SCIKIT LEARN DESIGN ---------------->

# "Primarily, three types of objects"--->
# "1. Estimators - It estimates some parameter based on a dataset.
    #  Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters\n",
# "2. Transformers - transform method take input and returns output based on the learnings
    # from fit(). It also has a convenience function called fit_transform() which fits and then transforms.\n",
# "3. Predictors - LinearRegression model is an example of predictor.
    #  fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions."

# FEATURE SCALING ----->
# Primarily, two types of feature scaling methods:--->
#     "1. Min-max scaling (Normalization)
#     "    (value - min)/(max - min)",
#     "    Sklearn provides a class called MinMaxScaler for this",
#     "2. Standardization",
#     "    (value - mean)/std",
#     "    Sklearn provides a class called StandardScaler for this

# We will use pipeline for the series of steps-->
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer', SimpleImputer(strategy = 'median')),
    ('std_scalar',StandardScaler())]) # add as many as pipeline you want
      

housing_num_tr = my_pipeline.fit_transform(housing)
# print(housing_num_tr)
# print(housing_num_tr.shape)


# SELECTING A MODEL --->
from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# if we want to use decision tree regressor model
from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()  # if we use decision tree model
# If we want to use randomforestRegressor then,
from sklearn.ensemble import RandomForestRegressor  # here ensemble means combine different regression and use it
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# PREDICTION SOME DATA-->
some_data = housing.iloc[:5]    # taking some data from housing data

some_labels = housing_labels.iloc[:5]  # taking 5 lables

prepared_data = my_pipeline.transform(some_data)

# print(model.predict(prepared_data))  # these are pridicted values
# Now we can compare/check the prediction values
# print(list(some_labels))   # these are  original values

# EVALUATING THE MODEL--->
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)
# print(rmse)  # so, this will be our root mean square error


# USING BETTER EVALUATION TECHNIQUE - CROSS VALIDATION
# how it works --> lets we have data -= 1 2 3 4 5 6 7 8 9 10
# we will find the error by training and testing 1 by 1 
# Use  any one for testing and other for training finding the errors and repeat the process respectively

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model , housing_num_tr , housing_labels, scoring = "neg_mean_squared_error" ,cv = 10)
rmsc_scores = np.sqrt(-scores)  # bcz we will get neg error - will convert it into pos
# print(rmsc_scores)


def print_scores(scores):
    print("Scores : " , scores)
    print("Mean : " , scores.mean())
    print("Standard deviation : " , scores.std())

# print_scores(rmsc_scores)    # This will give the values for decision tree model
# we can use linear regeression and RandomForestRegressor model like this, we will use that one type of model which will give lowest error

# So, now we got different values for mean and standard deviation for every model-->
# For linear regresion-->
    # Mean :  5.033624671156926
    # Standard deviation :  1.0560992597577876
# For Decision tree-->
    # Mean :  4.663916968421566
    # Standard deviation :  1.2175076398663032
# For RandomForestRegressor-->
    # Mean :  3.3148356942800534
    # Standard deviation :  0.6172708981931212

# As we can see randomforestRegressor model work better for this dataset
# So, We can use it for better prediction

# SAVING THE MODEL-->
from joblib import dump , load
dump(model , 'Dragon_Realstate_Model.joblib')
# now, we can load our model anywhere by importing from joblib and make predictions

# TESTING THE MODEL ON TEST DATA-->
x_test = strat_test_set.drop("MEDV",axis = 1)
y_test = strat_test_set["MEDV"].copy()

x_test_prepared = my_pipeline.transform(x_test)
final_prediction = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test , final_prediction)
final_rmse = np.sqrt(final_mse)
# print(final_rmse)
# print(final_prediction)  # these are the predicted values of y_test
# print(final_prediction , list(y_test))   # here, we can check by printing both
