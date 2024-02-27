#     Arabic Characters Recognition project
## problem statment 

- A supervised multi-class classification problem objective is to categorize/classifies images into specific Arabic alphabet letters.
- To identify the optimal model, three different models: SVM, KNN, and 2 distinct NN architectures are built.

## Dataset

- “csvTrainImages 13440x1024.csv” which contains 13440 rows and 1024 columns (i.e. flattened pixels extracted from 13440 images, each image is 32 x 32 pixels)
- “csvTrainLabel 13440x1.csv” which contains 13440 rows. Each row represents the label of the correspinding image in the train file. The label is an index of an Arabic character.  Example: Index is 2 refers to ‘ب‘.
-  “csvTestImages 3360x1024.csv” which contains 3360 rows and 1024 columns (i.e. flattened pixels extracted from 3360 images, each image is 32 x 32 pixels)
-  “csvTestLabel 3360x1.csv” which contains 3360 rows. Each row represents the label of the correspinding image in the test file.
#### Showing a sample of the data as images:

![Capture](https://github.com/shroukhm/Machine-learning-projects/assets/134003439/67f1797c-8a5f-495a-869c-f602f95d3535)

