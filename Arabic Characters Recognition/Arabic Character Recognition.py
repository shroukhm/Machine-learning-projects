#####################################  Problem Statement  #####################################
#  - A supervised multi-class classification problem objective is to categorize/classifies images into specific Arabic alphabet letters.
#  - The dataset comprises pixel values representing each image.
#  - To identify the optimal model , three different models : SVM, KNN, and 2 distinct NN architectures are built.



#####################################  1.Importing Libraries  #####################################
# Helper libraries
import pandas as pd
import numpy as np
import seaborn as sns

# For visualization
import matplotlib.pyplot as plt

# For data spliting
from sklearn.model_selection import train_test_split

# For modeling
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
# For preventing overfitting in NN models
from keras.callbacks import EarlyStopping

# For models evaluation
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score



#####################################  2.Loading the data  #####################################

# Load the training data
train_images = pd.read_csv("/content/csvTrainImages 13440x1024.csv", header=None)
train_labels = pd.read_csv("/content/csvTrainLabel 13440x1.csv", header=None)

# Load the testing data
test_images = pd.read_csv("/content/csvTestImages 3360x1024.csv", header=None)
test_labels = pd.read_csv("/content/csvTestLabel 3360x1.csv", header=None)

# Store the classes to use later when plotting the images , since the class names are not included with the dataset.
class_names_arabic = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك',
                          'ل', 'م', 'ن', 'ه', 'و', 'ي']



#####################################  3.Data exploration and preparation  #####################################

print("\n Train images :  \n" , train_images)

print("\n Train labels :  \n" , train_labels)

print("\n Test images :  \n" , test_images)

print("\n Test labels :  \n" , test_labels)



# Cheking for null values 
print( "\n No. of missing values in train images :  \n" , train_images.isnull().sum() )
print( "\n No. of missing values in train labels :  \n" , train_labels.isnull().sum() )
print( "\n No. of missing values in test images :  \n" , test_images.isnull().sum() )
print( "\n No. of missing values in test labels :  \n" , test_labels.isnull().sum() )

# Identify the number of unique classes
print( "\n No. of unique values in train labels :  \n" , len(train_labels[0].unique()) )

# Examine the distribution of samples in each class
print( "\n No. of each class in train labels :  \n" ,train_labels[0].value_counts() )



#####################################  4.Data Scaling  #####################################

# Normalizing each image to range of 0 to 1
# 255 --> the default number of number of pixels
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Another way for scaling ( get max. value in each column and divide the cell value by it )
#train_images_normalized = train_images.div(train_images.max(axis=1), axis=0)


# A function that reconstructs an image from its flattened vector then displayig it

def display_image(image):
    image_matrix = np.reshape(image, (32, 32))  # Reshape to 32x32
    plt.imshow(image_matrix, cmap='gray' )
    plt.show()

# Use this function to visualize some of the images in the test dataset
num_images_to_visualize = 3
for i in range(num_images_to_visualize):
  random_index = np.random.randint(0, len(test_images_normalized)-1)
  display_image(test_images_normalized.iloc[random_index, :].values)



#####################################  5.Modeling  #####################################

########### 5.1.SVM

# Training the model

# First experiment ( SVM )
# Train an SVM model on the training data
SVM_model =SVC(kernel='poly', degree=2, C=1).fit(train_images_normalized, train_labels)


# Test the model
SVM_pred = SVM_model.predict(test_images_normalized)

# Evaluating model performance

SVM_f1_scores = f1_score(test_labels, SVM_pred, average=None)
print('F1 Scores ( for each class ): \n', SVM_f1_scores)

# Calculate and print the average F1 score
average_svm_f1 = sum(SVM_f1_scores) / len(SVM_f1_scores)
print(f"SVM model : \nAverage F1 Score for all classes : {average_svm_f1:.2f}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_labels, SVM_pred)

# Plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_arabic, yticklabels=class_names_arabic)
plt.title('Confusion Matrix')
plt.show()



#####################################  Changing data labels  #####################################

# Change data labels from 1 : 28 to 0 : 27 to be suitable for KNN , NN models as their array indices are 0 based
train_labels = train_labels.apply( lambda x : x - 1 )
test_labels = test_labels.apply( lambda x : x - 1 )



#####################################  Spliting into training , validation datasets  #####################################

# Split the training data into training and validation sets
X_train_images, X_val_images, y_train_labels, y_val_labels = train_test_split(train_images_normalized, train_labels, test_size=0.2,
                                                                              random_state=42)


########### 5.2.KNN

# Training the model

# Second experiment ( KNN )
# Define a range of K values to experiment with
k_values = np.arange(1, 21)

# Experiment with different K values
f1_scores = []

for k in k_values:
    # Training the model on the training dataset part
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_images, y_train_labels)
    # Testing the model predictions on the validation dataset part
    y_val_pred = knn_model.predict(X_val_images)
    f1 = f1_score(y_val_labels, y_val_pred, average='weighted')
    f1_scores.append(f1)

# Plot the average f-1 scores with different K values
plt.plot(k_values, f1_scores, marker='o')
plt.title('Average F-1 Scores with Different K Values')
plt.xlabel('K Value')
plt.ylabel('Average F-1 Score')
plt.show()

# Find the best K value
best_k = k_values[np.argmax(f1_scores)]
print(f"The best K value is : {best_k}")


# Test the model with the best K value
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(train_images_normalized, train_labels)
test_labels_pred = best_model.predict(test_images_normalized)


# Evaluating model performance
knn_f1_scores = f1_score(test_labels, test_labels_pred, average='weighted')
print(f"KNN model : \nAverage F-1 Score on Testing Dataset: {knn_f1_scores}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(test_labels, test_labels_pred)

# Plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_arabic, yticklabels=class_names_arabic)
plt.title('Confusion Matrix')
plt.show()

########### 5.3.Neural Network 

#### 1st NN model

# Third experiment ( NN )
# To create a sequential model, which is a linear stack of layers.
# Each dense represents a layer in the NN :
# 1.Input layer --> the size of the train images
# 2. 2 Hidden layer --> the first & second one contains 150 , 60 neurons respectively.
# 3.Output layer --> Each node contains a score indicating the current image belongs to one of the 28 classes
# ( no. of Arabic alphabet letters ).

nn_model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train_images.shape[1],)),
    tf.keras.layers.Dense(units=150, activation='relu'),
    tf.keras.layers.Dense(units=60, activation='relu'),
    tf.keras.layers.Dense(28,activation ='softmax')
])


# Compile the model

# optimizer -->  how the model is updated based on the data it sees and its loss function.
# adam --> Combines ideas from RMSprop and momentum. It adapts the learning rates of each parameter based on their historical gradients.
# loss --> Measures how accurate the model is during training.
# metrices --> Used to monitor the training and testing steps.
# Suitable choice when the labels are integers (class indices).
# As it used in scenarios where the classes are mutually exclusive, and each input belongs to exactly one class.
# The following example uses accuracy, the fraction of the images that are correctly classified.
nn_model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training & Testing ( on val dataset ) the model

# To prevent overfitting by stopping training when the best validation performance reached
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model1 = nn_model1.fit(X_train_images, y_train_labels, epochs=50, validation_data=(X_val_images, y_val_labels), callbacks=[early_stopping])

# Evaluating model performance
val_loss, val_acc = nn_model1.evaluate(X_val_images,  y_val_labels)

print('\nTesting loss on validation dataset:', val_loss)
print('\nTesting accuracy on validation dataset:', val_acc)

# Plot the error and accuracy curves for the training data and validation data.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model1.history['accuracy'], label='Training Accuracy')
plt.plot(model1.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Model 1 - Accuracy Curves')

plt.subplot(1, 2, 2)
plt.plot(model1.history['loss'], label='Training Loss')
plt.plot(model1.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Model 1 - Loss Curves')

plt.tight_layout()
plt.show()



#### 2nd NN model

# Third experiment ( NN )
# To create a sequential model, which is a linear stack of layers.
# Each dense represents a layer in the NN:
# 1.Input layer --> the size of the train images
# 2. 4 Hidden layer --> the first , second , third & fourth ones contains 200 , 140 , 100 , 40 neurons respectively.
# 3.Output layer --> Each node contains a score indicating the current image belongs to one of the 28 classes
# ( no. of Arabic alphabet letters ).

nn_model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(X_train_images.shape[1],)),
    tf.keras.layers.Dense(units=200, activation='tanh'),
    tf.keras.layers.Dense(units=140, activation='tanh'),
    tf.keras.layers.Dense(units=100, activation='tanh'),
    tf.keras.layers.Dense(units=40, activation='tanh'),
    tf.keras.layers.Dense(28,activation ='softmax')
])


# Compile the model

# optimizer -->  how the model is updated based on the data it sees and its loss function.
# adam --> Combines ideas from RMSprop and momentum. It adapts the learning rates of each parameter based on their historical gradients.
# loss --> Measures how accurate the model is during training.
# metrices --> Used to monitor the training and testing steps.
# Suitable choice when the labels are integers (class indices).
# As it used in scenarios where the classes are mutually exclusive, and each input belongs to exactly one class.
# The following example uses accuracy, the fraction of the images that are correctly classified.
nn_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training & Testing ( on val dataset ) the model

model2 = nn_model2.fit(X_train_images, y_train_labels, epochs=60 , validation_data=(X_val_images, y_val_labels) , callbacks=[early_stopping])


# Evaluating model performance

val_loss, val_acc = nn_model2.evaluate(X_val_images,  y_val_labels)
print('\nTesting loss on validation dataset:', val_loss)
print('\nTesting accuracy on validation dataset:', val_acc)

# Plot the error and accuracy curves for the training data and validation data.
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model2.history['accuracy'], label='Training Accuracy')
plt.plot(model2.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Model 2 - Accuracy Curves')

plt.subplot(1, 2, 2)
plt.plot(model2.history['loss'], label='Training Loss')
plt.plot(model2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Model 2 - Loss Curves')

plt.tight_layout()
plt.show()

# > __Since the first NN model test accuarcy on validation data = 70.49% , second one = 68.9%, then the best NN model for this data is the first one.__

##################################### 5.4.The best NN model

# Saving the best NN model in a separate file
nn_model1.save('best_model.h5')


# Reload it in a separate file
best_model = tf.keras.models.load_model('best_model.h5')


# To prevent overfitting by stopping training when the best validation performance reached
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
best_model.fit(train_images_normalized, train_labels, epochs=60 , callbacks=[early_stopping])


# Calculating the loss , accuracy of test data
best_test_loss, best_test_accuracy = best_model.evaluate(test_images_normalized, test_labels)

print(f'Test Loss: {best_test_loss}')
print(f'Test Accuracy: {best_test_accuracy}')


# Making predictions on test data

# The model has predicted the label for each image in the testing set
predictions = best_model.predict(test_images_normalized)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

best_nn_f1_scores = f1_score(test_labels, predicted_classes, average='weighted')
print(f"Best NN model : \nAverage F-1 Score on Testing Dataset: {best_nn_f1_scores}")


# Compute the confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_arabic, yticklabels=class_names_arabic)
plt.title('Confusion Matrix')
plt.show()

# Displaying the test predictions
plt.figure(figsize=(10,10))
for i in range(0 , 15 ) :
  plt.subplot(3, 5, i + 1)
  display_image(test_images_normalized.iloc[i, :].values)

  label_of_index  = predicted_classes[i]
  print (f"\nThe predicted label : {class_names_arabic[label_of_index]}\n")

  plt.tight_layout()
  plt.show()
#####################################  6.Comparing all models  #####################################

print(f"\nSVM model : \nTesting Average f1-score : {average_svm_f1}")
print(f"\nKNN model : \nTesting Average f1-score : {knn_f1_scores}")
print(f"\nBest NN model : \nTesting Average f1-score : {best_nn_f1_scores}")

# >__It is obvious thet the NN model is the best model for this data.__