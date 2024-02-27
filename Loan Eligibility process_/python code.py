# 1.Importing libraries
import pandas as pd
import numpy as np
# For visualization 
import seaborn as sns
import matplotlib.pyplot as plt
# For data splitting
from sklearn.model_selection import train_test_split
# For Feature Engineering ( Encoding , Scaling )
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
# For Modeling
from sklearn.linear_model import LinearRegression

# For regression model evaluation
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error



# ----------------------- Training data -----------------------

# 2.Loading the data
data = pd.read_csv("loan_old.csv")
print("\nThe original ( training ) data : ")
print(data.info())



# -------------------------------------
# 3.Data Exploration 

# To print the data with it's dimensions ( no. of rows and columns )
print(data)

# Dropping not important feature
data.drop( "Loan_ID" , axis = 1 , inplace=True)

# Checking for missing values in each column
print("\nMissing values in each column :")
print(data.isnull().sum())

# Checking the type of each feature (categorical or numerical)
print("\nThe data type of each column : " )
print( data.dtypes)

# Checking the numerical features distributions ( whether have the same scale or not ) 
print("\nStatistical analysis for each column:")
print(data.describe())   # We concluded that the data doesn't have the same scale ( range of values ) 

# A pair-plot to show the distribution of data values between numerical columns based on loan status ( 0, 1 )
sns.pairplot(data , hue="Loan_Status" , height=1.5)
plt.show()



# -------------------------------------
# 4. Data Preprocessing 
# Dropping records containing missing values
data.dropna( inplace=True )

print("\nThe data after dropping missing values :")
print(data.info())



# -------------------------------------
# 5. Data Splitting 
x = data.drop(["Loan_Status" , "Max_Loan_Amount" ] , axis = 1 )
LS_y = data["Loan_Status"]
MLA_y = data["Max_Loan_Amount"]

# The data is shuffled and split into training and testing sets
# For Loan_Status prediction ( Binary Classification ) , for Max_Loan_Amount prediction ( Regression )
x_train, x_test, ls_y_train, ls_y_test , mla_y_train , mla_y_test= train_test_split(x, LS_y, MLA_y , test_size=0.2, random_state=42)

# Printing the data and shape ( dimension ) of the training and testing data
print("\nX_train : ")
print(x_train.head())
print(x_train.shape)

print("\nX_test : ")
print(x_test.head())
print(x_test.shape)

print("\nLoan Status y_train : ")
print(ls_y_train.head())
print(ls_y_train.shape)

print("\nLoan Status y_test : ")
print(ls_y_test.head())
print(ls_y_test.shape)

print("\nMax. Loan Amount y_train : ")
print(mla_y_train.head())
print(mla_y_train.shape)

print("\nMax. Loan Amount y_test : ")
print(mla_y_test.head())
print(mla_y_test.shape)



# -------------------------------------
# 6. Data encoding

# Binary ( dictionary ), One Hot Encoding for training categorical features 
binary_dict = { "Male" : 0 ,  "Female" : 1 , "No" : 0 , "Yes": 1 , "Not Graduate" : 0 , "Graduate" : 1}
columns_to_map = ["Gender", "Married", "Education"]
x_train[columns_to_map] = x_train[columns_to_map].replace(binary_dict)

x_train = pd.get_dummies(x_train, columns=["Dependents" , "Property_Area" ], drop_first=True)
print("\nX_train after encoding :" )
print( x_train.info())
print(x_train)

# Binary , One Hot Encoding for testing categorical features 
x_test[columns_to_map] = x_test[columns_to_map].replace(binary_dict)

x_test = pd.get_dummies(x_test, columns=["Dependents" , "Property_Area" ], drop_first=True)
print("\nX_test : \n")
print(x_test.info())
print(x_test)

# Binary Encoding for target feature
loanstatus_dict = {  "N" : 0  , "Y" : 1 }
ls_y_train = ls_y_train.map(loanstatus_dict)
ls_y_test = ls_y_test.map(loanstatus_dict)



# -------------------------------------
# 7. Data scaling
# RobustScaler will be used as it isn't sensitive to outliers ( in case of their existence )
scaler = RobustScaler()
x_train[["Income" , "Coapplicant_Income" , "Loan_Tenor"]] = scaler.fit_transform(x_train[["Income" , "Coapplicant_Income" , "Loan_Tenor"]])
x_test[["Income" , "Coapplicant_Income" , "Loan_Tenor"]] = scaler.transform(x_test[["Income" , "Coapplicant_Income" , "Loan_Tenor"]])



# -------------------------------------
# 8. Modeling
# Linear Regression model to predict the max. loan amount ( continous values )
reg_model = LinearRegression()
# Train the model
reg_model.fit(x_train , mla_y_train)
# Test the model
y_pred = reg_model.predict(x_test)
# Evaluate the model
r2 = r2_score(mla_y_test, y_pred)

mse = mean_squared_error(mla_y_test , y_pred)

print("\nLinear Regression : ")
print(f'Mean Squared Error : {mse}')  # We concluded that ,on average, the squared difference between your model's predictions and the actual values is quite large
print("R-squared (R2) Score:", r2)
print(f'Training score : {reg_model.score(x_train,mla_y_train)} \nTest score : {reg_model.score(x_test, mla_y_test)}')
print("_________________________________________________________________________________________")


# Binary Classification model to predict the loan status ( 0 , 1 )

# Sigmoid activation function --> to make the output of the hypothesis function between 0 and 1.
def sigmoid(z):

    return 1 / (1 + np.exp(-z))

# Logistic regression hypothesis function --> to calculates the predicted probabilities using the sigmoid function.
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

# Compute the logistic regression cost  -> It computes the log-likelihood cost based on the predicted probabilities and the actual labels.
def compute_cost(X, Y, theta):
    m = len(Y)
    h = hypothesis(X, theta)
    cost = -(1 / m) * np.sum(Y * np.log(h) + (1 - Y) * np.log(1 - h))
    return cost

#  Gradient descent optimization for logistic regression --> iteratively updates the parameters (theta) based on the gradient of the cost function.
def gradient_descent(X, Y, theta, learning_rate = 0.01 , num_iterations = 1000):
    m = len(Y)
    costs = []
    for _ in range(num_iterations):
        h = hypothesis(X, theta)
        gradient = np.dot(X.T, (h - Y)) / m
        theta -= learning_rate * gradient
        cost = compute_cost(X, Y, theta)
        costs.append(cost)

    return theta, costs

# Train logistic regression model --> It adds a bias term to the input features, initializes the parameters, and then calls the gradient descent function 
# to optimize the parameters.
def logistic_regression(X, Y, learning_rate=0.01, num_iterations=1000):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros(X.shape[1])
    theta, costs = gradient_descent(X, Y, theta, learning_rate, num_iterations)
    return theta, costs


# A function to predict the accuracy of the logistic regression model

# Predict labels based on logistic regression model
# X : input features , theta : learned parameters  
def Predict(X, theta, threshold=0.5):
    # To calculate the predicted probabilities using the logistic regression model
    probabilities = hypothesis(X, theta)
    # To Converts probabilities to binary predictions based on the threshold
    return (probabilities >= threshold).astype(int)


# Calculate accuracy of logistic regression model
#  y : actual labels
def calculate_accuracy(X, Y, theta, threshold=0.5):
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the bias term
    predictions = Predict(X, theta, threshold)
    # To Compares binary predictions with actual labels and calculates the accuracy as the ratio of correct predictions
    correct_predictions = (predictions == Y).astype(int)
    accuracy = np.mean(correct_predictions)
    return accuracy

# Train the model
theta , costs = logistic_regression(x_train , ls_y_train)
print("\nLearned Parameters (Theta):", theta)
print("\nFinal Cost:", costs[-1])

# Testing and evaluating the model
# Calculate accuracy on the training set
train_accuracy = calculate_accuracy(x_train,ls_y_train, theta)

# Calculate accuracy on the test set
test_accuracy = calculate_accuracy(x_test, ls_y_test, theta)

# Print the accuracies
print("\nLogistic Regression : ")
print("\nTraining Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("_________________________________________________________________________________________")




# ----------------------- Testing data -----------------------

# 2.Loading the data
new_data = pd.read_csv("loan_new.csv")
print("\nNew ( testing ) data : ")
print(new_data.info())



# -------------------------------------
# 3.Data Exploration 

# To print the data with it's dimensions ( no. of rows and columns )
print(new_data)

# Dropping not important feature
new_data.drop( "Loan_ID" , axis = 1 , inplace=True)

# Checking for missing values in each column
print("\nMissing values in each column in the new data :")
print(new_data.isnull().sum())

# Checking the type of each feature (categorical or numerical)
print("\nThe data type of each column in the new data : " )
print( new_data.dtypes)

# Checking the numerical features distributions ( whether have the same scale or not ) 
print("\nStatistical analysis for each column in the new data :")
print(new_data.describe())   # We concluded that the data doesn't have the same scale ( range of values ) 

# A pair-plot to show the distribution of data values between numerical columns based on loan status ( 0, 1 )
sns.pairplot(new_data  , height=1.5)
plt.show()



# -------------------------------------
# 4. Data Preprocessing 
# Dropping records containing missing values
new_data.dropna( inplace=True )
print("\nThe Cleaned new dataset :")
print(new_data.info())



# -------------------------------------
# 5. Data Encoding
new_data[columns_to_map] = new_data[columns_to_map].replace(binary_dict)

new_data = pd.get_dummies(new_data, columns=["Dependents" , "Property_Area" ], drop_first=True)
print("\nNew data after encoding : " )
print(new_data)



# 6. Data scaling
new_data[["Income" , "Coapplicant_Income" , "Loan_Tenor"]] = scaler.transform(new_data[["Income" , "Coapplicant_Income" , "Loan_Tenor"]])



# -------------------------------------
# 7.Modeling
# Predicting the two target variables : 

# Loan Status--> Logistic Regression 

# Add a column of ones for the bias term
new_df = np.c_[np.ones(new_data.shape[0]), new_data]
new_data["Binary_Loan_Status"] = Predict(new_df , theta )
new_data["Loan_Status"] = new_data["Binary_Loan_Status"].apply(lambda x: "Y" if x == 1 else "N")
print("\n New data after predicting Loan Status : \n" , new_data)

# To remove the two columns of the binary predicted variables ( 1st target variable ) to predict the 2nd onr ( continous values )
reg_new_data = new_data.drop(["Binary_Loan_Status" , "Loan_Status"] , axis = 1)

# Max. Loan Amount --> Linear Regression
reg_new_data["Max_Loan_Amount"] = reg_model.predict(reg_new_data)
print("\n New data after predicting Max. Loan Amount : \n" , reg_new_data)