# 1.Importing libraries

import pandas as pd
import numpy as np



# 2.Loading the data

data = pd.read_csv("diabetes.csv")

print(data.info())



# 3. Splitting the data to X (inputs) and Y (output)

# random_seed : To initialize the random number generator
# If you use the same random seed, you will get the same sequence of random numbers every time you run your code.
def train_test_split(data, labels, test_size=0.2, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    # Number of samples
    num_samples = len(data)

    # Shuffle indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Calculate the number of samples for the test set
    num_test = int(test_size * num_samples)

    # Split the indices into training and testing sets
    test_indices = indices[:num_test]
    train_indices = indices[num_test:]

    # Split the data and labels
    X_train, X_test = data.iloc[train_indices], data.iloc[test_indices]
    y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]

    return X_train, X_test, y_train, y_test

x = data.drop(["Outcome"], axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_seed=42)



# 4.Data preprocessing

# Normalize each feature column using Min-Max Scaling function
def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())


# Apply Min-Max Scaling on training data and test data
x_train_scaled = x_train.apply(lambda x: min_max_scaling(x))
x_test_scaled = x_test.apply(lambda x: min_max_scaling(x))



# 5.Modeling

# KNN Classifer to predict whether this female is diabetic or not ( 0 , 1 )
def knn_predict(x_train, y_train, x_test, k):

    predictions = []
    for i in range(len(x_test)):

        distances = []

        for j in range(len(x_train)):
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((x_test.iloc[i] - x_train.iloc[j]) ** 2))
            distances.append((distance, y_train.iloc[j]))

        # Sort distances and get k-nearest neighbors
        k_nearest_neighbors = sorted(distances)[:k]

        # Count the votes for each class
        votes = {0: 0, 1: 0}
        for _, label in k_nearest_neighbors:
            votes[label] += 1


        # Check for tie existence
        max_vote = max(votes.values())
        if list(votes.values()).count(max_vote) > 1:
            # If tie exists, apply distance-weighted voting
            # A dictionary To keep track of the weighted votes for each class (0 and 1)
            # during the Distance-Weighted Voting process in the k-Nearest Neighbors (KNN) algorithm
            # The keys are the class labels (0 and 1), and the corresponding values are the initial vote counts
            weighted_votes = {0: 0, 1: 0}
            for distance, label in k_nearest_neighbors:
                # Assign higher weights to closer neighbors
                weight = 1 / (distance + 1)  # Adding 1 to avoid division by zero
                weighted_votes[label] += weight

            # Predict the class with the highest weighted vote
            predicted_class = max(weighted_votes, key=weighted_votes.get)
        else:
            # If no tie, predict the class with the majority of votes
            predicted_class = max(votes, key=votes.get)

        predictions.append(predicted_class)

    return predictions

# Testing the model using different k values
i = 1
accuracy_sum = 0
accuracy_sum_train = 0
accuracy_sum_test = 0
for k in range(3, 7):  # You can adjust the range as needed

    print( f'Iteration {i} : ' )
    i += 1

    print(f'   - k value : {k}')

    # Training set predictions
    predictions_train = knn_predict(x_train_scaled, y_train, x_train_scaled, k)
    correct_train = sum(predictions_train == y_train)     # Number of correctly classified instances
    total_train = len(y_train)
    accuracy_train = correct_train / total_train

    print(f'   - Number of correctly classified train instances : {correct_train}')
    print(f'   - Total number of instances in the train set : {total_train}')
    print(f"   - Training Accuracy for k = {k} : {accuracy_train*100}%\n")

    # Test set predictions
    predictions_test = knn_predict(x_train_scaled, y_train, x_test_scaled, k)
    correct_test = sum(predictions_test == y_test)
    total_test = len(y_test)
    accuracy_test = correct_test / total_test

    print(f'   - Number of correctly classified test instances : {correct_test}')
    print(f'   - Total number of instances in the test set : {total_test}')
    print(f"   - Testing Accuracy for k = {k} : {accuracy_test *100}%\n")

    # To get the sum of all accuracies for average accuracies across all iterations calculation
    accuracy_sum_train += accuracy_train
    accuracy_sum_test += accuracy_test

# Printing the Average accuracies across all iterations
average_accuracy_train = accuracy_sum_train / (i - 1)  # subtract 1 to get the correct number of iterations
average_accuracy_test = accuracy_sum_test / (i - 1)
print(f'\nAverage Training Accuracy across all iterations : {average_accuracy_train*100}%')
print(f'Average Testing Accuracy across all iterations : {average_accuracy_test*100}%\n')