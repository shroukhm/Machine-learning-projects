# 1.Importing libraries

import pandas as pd
import numpy as np
# For visualization 
import matplotlib.pyplot as plt
# For data splitting
from sklearn.model_selection import train_test_split
# To create a Decision Tree model for classification
from sklearn.tree import DecisionTreeClassifier
# To evaluate the accuracy of a classification model
from sklearn.metrics import accuracy_score



# 2.Loading the data

data = pd.read_csv("drug.csv")

print(data.info())



# 3.Data preprocessing

# Handling the missing Values

# Checking the number of null values
data.isnull().sum()
# Handle missing values by dropping them as they represent very small number of data
data.dropna( inplace=True )

# Categorical Encoding

# One Hot Encoding
data_encoded = pd.get_dummies(data, columns=["Sex", "BP", "Cholesterol"], drop_first=True)
# Label Encoding
drug_dict = { "drugA" : 1 ,  "drugB" : 2 , "drugC" : 3 , "drugX": 4 , "drugY" : 5}
data_encoded["Drug"] = data_encoded["Drug"].replace(drug_dict)

print(data_encoded)



# 4. Splitting the data to X (inputs) and Y (output)

# Extract features and target variable
X = data_encoded.drop('Drug', axis=1)
y = data_encoded['Drug']



# 5.Modeling

# Define a function for the experiment ( training , testing , evaluation )
def run_experiment(X_train, X_test, y_train, y_test):
    # Train a decision tree classifier
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(X_train, y_train)

    # Make predictions
    y_pred = dec_tree.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Return the decision tree and accuracy
    return dec_tree, accuracy


# Training and Testing with Fixed Train-Test Split Ratio

best_accuracy = 0
best_model = None
# Different random states values
rs_values = [15, 20 , 26 , 36 , 42]
for i , random_state in enumerate (rs_values , start = 1):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= random_state )

    # Run the experiment
    tree, accuracy = run_experiment(X_train, X_test, y_train, y_test)

    # Print results
    print(f"Experiment {i} --> Test Accuracy : {accuracy:.4f} , Training Size : {len(X_train)} , Testing Size : {len(X_test)} , Tree Size : {tree.tree_.node_count}")

    # To select the model that achieves the highest overall performance
    if accuracy > best_accuracy :
      best_accuracy = accuracy
      best_model = tree


print ("/nBest model : ")
print(f"Test Accuarcy : {best_accuracy:.4f} , Tree Size : {best_model.tree_.node_count}")


# Training and Testing with a Range of Train-Test Split Ratios

split_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
results = {'Training Size': [], 'Mean Accuracy': [], 'Max Accuracy': [], 'Min Accuracy': [], 'Mean Tree Size': [] , 'Max Tree Size' : [] , 'Min Tree Size' : []}

for split_ratio in split_ratios:
    accuracies = []
    tree_sizes = []

    for random_state in rs_values :
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split_ratio, random_state= random_state)

        # Run the experiment
        tree, accuracy = run_experiment(X_train, X_test, y_train, y_test)

        # Append results
        accuracies.append(accuracy)
        tree_sizes.append(tree.tree_.node_count)

    # Calculate statistics
    accuracy_mean = np.mean(accuracies)
    max_accuracy = np.max(accuracies)
    min_accuracy = np.min(accuracies)

    tree_size_mean = np.mean(tree_sizes)
    max_tree_size = np.max(tree_sizes)
    min_tree_size = np.min(tree_sizes)


    # Store results
    results['Training Size'].append(split_ratio)
    results['Mean Accuracy'].append(accuracy_mean)
    results['Max Accuracy'].append(max_accuracy)
    results['Min Accuracy'].append(min_accuracy)
    results['Mean Tree Size'].append(tree_size_mean)
    results['Max Tree Size'].append(max_tree_size)
    results['Min Tree Size'].append(min_tree_size)

# Display results
result_df = pd.DataFrame(results)
print(result_df)



# 6.Plots

plt.figure(figsize=(12, 6))

# Accuracy vs Training Set Size
plt.subplot(1, 2, 1)
plt.plot(result_df['Training Size'], result_df['Mean Accuracy'], marker='o')
plt.title('Mean Accuracy vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Accuracy')

# Tree Size vs Training Set Size
plt.subplot(1, 2, 2)
plt.plot(result_df['Training Size'], result_df['Mean Tree Size'], marker='o')
plt.title('Mean ree Size vs Training Set Size')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Tree Size')

plt.tight_layout()
plt.show()

