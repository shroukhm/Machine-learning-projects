#      Loan Eligibility process project

## problem statment 

A housing finance company offers interest-free home loans to customers.
When a customer applies for a home loan, the company validates the
customer's eligibility for a loan before making a decision.
Now, the company wants to automate
the customers' eligibility validation
process based on the customers' details
provided while filling the application
form. These details include gender,
education, income, credit history, and
others. The company also wants to have
a predictive model for the maximum
loan amount that an applicant is
authorized to borrow based on his details.
You are required to build a linear regression model and a logistic regression
model for this company to predict loan decisions and amounts based on
some features.

## Business Question

 This is a ML model which helps the housing finance company in loan eligibility validation process by predicting :

1. The loan acceptance status (whether the loan is approved or rejected).
2. Max. loan amount.
#
So, this project is divided into two sub projects:
1. Binary Classification model (Logistic Regression): for loan status prediction. 
2. Regression model (Linear Regression model): for max. loan amount prediction.

## Datasets

There are two attached datasets:
- The first dataset “loan_old.csv” contains 614 records of applicants' data with 10 feature columns in addition to 2 target columns. The features are: the loan application ID, the applicant's gender, marital status, number of dependents, education and income, the co-applicant's income, the number of months until the loan is due, the applicant's credit history check, and the property area. The targets are the maximum loan amount (in thousands) and the loan acceptance status.
- The second dataset “loan_new.csv” contains 367 records of new applicants' data with the 10 feature columns.

Note: These datasets are modified versions of the "Loan Eligibility Dataset".
The original datasets were obtained from Kaggle
