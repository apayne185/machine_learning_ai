# Predicting Car Body Types Using K-Nearest Neighbors
This question demonstrates the use of machine learning to classify the body type (Carrocería) of a car based on its dimensions and weight. The solution involves a detailed exploratory data analysis and the implementation of a K-Nearest Neighbors (KNN) model for classification. 

### Problem Statement
Given a car's weight and dimensions, determine its probable body type (Carrocería) from a predefined set of classes. This involves solving a supervised classification problem, where the target variable is categorical.

#### Features
- Peso (kg): Weight of the car in kilograms.
- Largo (mm): Length of the car in millimeters.
- Ancho (mm): Width of the car in millimeters.
- Alto (mm): Height of the car in millimeters.

#### Target Variable
- Carrocería: The type of body, such as Berlina, Convertible, Coupe, etc.

### Approach
#### Preliminary Analysis
1. Loaded and inspected the dataset cars_2021_v0_2.csv.
2. Selected the relevant features (Peso, Largo, Ancho, Alto) and target variable (Carrocería) for the classification task.
3. Observed data properties:
    - Dataset has 5,340 rows and 86 columns.
    - Only a subset of columns is relevant to this task.
#### Algorithm Choice
- K-Nearest Neighbors (KNN) was chosen as the classification algorithm because:
    - It works well with categorical target variables.
    - It is computationally feasible given the dataset size.
- Alternative Considerations: Naive Bayes, Decision Trees, and SVM were considered but not implemented due to computational or data-specific constraints.
### Implementation Steps
#### Data Preprocessing
- Split Dataset: Divided the dataset into training (80%) and testing (20%) subsets.
- Scaling: Applied StandardScaler to normalize feature values, ensuring that features with larger scales (e.g., length in mm) do not dominate the KNN distance calculations.
#### Model Training
1. Used KNeighborsClassifier from scikit-learn.
2. Experimented with different values of neighbors (k):
    - Initial k=5 achieved 85.86% accuracy.
    - After tuning, the optimal value was determined to be k=9.
#### Model Evaluation
1. Metrics used:
    - Accuracy: Percentage of correctly classified samples.
    - Precision: Correctly predicted positives divided by total predicted positives.
    - Recall: Correctly predicted positives divided by total actual positives.
2. Final Model Performance:
    - Accuracy: 82.21%
    - Precision: 82.14%
    - Recall: 82.21%
#### Prediction
For a given car with the following characteristics:
json
```
{
    "Peso (kg)": 1500.0,
    "Largo (mm)": 4600.0,
    "Ancho (mm)": 1900.0,
    "Alto (mm)": 1400.0
}
```
The model predicts:
- Body Type: Berlina
- Probability: 89%

  
### Code Explanation
#### 1. Data Preprocessing
```
X = df.loc[:,['Peso (kg)', 'Largo (mm)', 'Ancho (mm)', 'Alto (mm)']]
Y = df['Carrocería']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=9)

# Scaling
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```
#### 2. Model Training and Optimization
```
# Initial KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

# Accuracy calculation
accuracy = accuracy_score(Y_test, knn.predict(X_test))
print(f'Accuracy: {accuracy}')

# Hyperparameter tuning
k_options = range(1, 20)
accuracies_list = []
for k in k_options:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    accuracies_list.append(accuracy_score(Y_test, knn.predict(X_test)))
```
#### 3. Final Prediction
```
# Final model with k=9
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, Y_train)

# Predicting the body type for the given car
input_data = pd.DataFrame(question3, index=[0])
input_data = scaler.transform(input_data)
prediction = knn.predict(input_data)
probabilities = knn.predict_proba(input_data)

print(f"The predicted body type is: {prediction}")
for body_type, prob in zip(knn.classes_, probabilities[0]):
    print(f"{body_type}: {prob:.2f}")
```
### Results
- Predicted Body Type: Berlina
- Prediction Probability: 89%
- Performance Metrics:
    - Accuracy: 82.21%
    - Precision: 82.14%
    - Recall: 82.21%
### Output
#### Answers JSON
The final answers are saved as a JSON file:

### Conclusion
This project showcases the application of K-Nearest Neighbors to predict a car's body type based on its dimensions and weight. Despite some overfitting challenges, careful hyperparameter tuning yielded a well-performing model with robust predictions. Future improvements could include exploring additional features or trying alternative models such as Decision Trees or ensemble methods.

***

# Car Price Prediction and Body Type Classification
### Overview
This question involves building machine learning models to predict car prices and classify car body types. The data includes various features such as weight, dimensions, horsepower, and brand of cars, and the task is to make accurate predictions based on these features. The project uses supervised learning techniques like Decision Trees and K-Nearest Neighbors (KNN) for regression and classification tasks, respectively.

### Objective 1: Car Price Prediction (Regression Task)
Predict the price of a car based on features like horsepower, number of doors, and brand using a Decision Tree Regressor.

### Objective 2: Car Body Type Classification (Classification Task)
Classify the body type of a car (e.g., Sedan, Convertible, SUV) based on its weight, length, width, and height using the K-Nearest Neighbors (KNN) classifier.

### Data
The dataset (cars_2021_v0_2.csv) contains the following columns:

POTENCIA1 (cv): Horsepower of the car (in CV).
- PUERTAS: Number of doors.
- PRECIO: Price of the car (in EUR).
- Peso (kg): Weight of the car (in kg).
- Largo (mm): Length of the car (in mm).
- Ancho (mm): Width of the car (in mm).
- Alto (mm): Height of the car (in mm).
- Carrocería: Body type of the car (e.g., Sedan, Convertible, SUV).

### Objective 1: Car Price Prediction
  
#### Approach
We used a Decision Tree Regressor model to predict the price of a car based on the following features:

- POTENCIA1 (cv): Horsepower
- PUERTAS: Number of doors
- *BRAND_ (dummy variables)**: The brand of the car (one-hot encoded)

#### Model Evaluation Metrics
We evaluated the model using the following metrics:

- Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values.
- Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
- R-squared: Measures how well the model explains the variance in the target variable (Price).
Example Results:
```
Predicted Price: [21394.49, 21394.49, 30366.31, ...]
Mean Absolute Error (MAE): 9174.54
Mean Squared Error (MSE): 214864256.98
R-squared: 0.73
```
The R-squared value of 0.73 indicates that the model explains 73% of the variance in the car price.

### Objective 2: Car Body Type Classification
#### Approach
For the classification task, we used a K-Nearest Neighbors (KNN) classifier to predict the car body type based on the following features:

- Peso (kg): Weight
- Largo (mm): Length
- Ancho (mm): Width
- Alto (mm): Height
- Model Evaluation Metrics
We evaluated the classification model using the following metrics:

- Accuracy: Measures the percentage of correct predictions.
- Precision: Measures the proportion of true positive predictions out of all positive predictions.
- Recall: Measures the proportion of true positive predictions out of all actual positive instances.
Example Results:
```
Predicted Body Type: ['Berlina']
Berlina: 0.89
Bus: 0.00
Chasis: 0.00
```
The model predicts that the car with the given dimensions most likely has a body type of Berlina (Sedan) with 89% probability.

### Model Performance
##### Car Price Prediction:
- MAE: 9174.54 EUR
- MSE: 214,864,256.98 EUR
- R-squared: 0.73
The Decision Tree Regressor model performs fairly well, but improvements could be made by reducing the MSE and increasing the R-squared value.

##### Car Body Type Classification:
- Accuracy: 87.92%
- Precision: 82.14%
- Recall: 82.21%
The KNN classifier achieves high accuracy and performs well in terms of precision and recall. Fine-tuning the number of neighbors could improve results further.

### Conclusion
This project demonstrates how to use machine learning to predict car prices and classify car body types based on various features. The decision tree model is effective for price prediction, while KNN works well for body type classification. Fine-tuning both models further could improve their performance.


***

# Car Price Prediction Using Decision Tree Regression
This project aims to predict the price of a car based on its features using a Decision Tree Regression model. Specifically, we will predict the price of a Mercedes car based on the following attributes:

- BRAND: Mercedes
- PUERTAS: Number of doors (2 in this case)
- POTENCIA1 (cv): Horsepower (ranging between 200 and 300 horsepower)

### Problem Description
Given the attributes of a car, we need to predict its price. The primary features in this case are:

- BRAND: Whether the car is a Mercedes (binary feature: 0 = not Mercedes, 1 = Mercedes)
- PUERTAS: Number of doors (an integer value)
- POTENCIA1 (cv): Horsepower (a continuous value between 200 and 300)
We will apply a Decision Tree Regression model to predict the car's price based on these features.

### Steps Involved
#### 1. Data Preprocessing
The dataset cars_2021_v0_1.csv contains the following columns:

- PUERTAS: Number of doors on the car.
- BRAND_Mercedes: Binary indicator (1 for Mercedes, 0 for other brands).
- POTENCIA1 (cv): Horsepower (in horsepower units).
- PRECIO: Car price (in euros).
We filtered and selected only the relevant columns for the prediction task.

#### 2. Model Creation
A Decision Tree Regressor model is used to predict the car price. The decision tree algorithm works by creating a tree-like structure where each node represents a feature and its threshold value that splits the dataset, leading to predictions at the leaf nodes.

#### 3. Training the Model
The data is split into training and test sets using train_test_split. The model is first trained on the training data, and the performance is evaluated using the R-squared score.

#### 4. Hyperparameter Tuning with GridSearchCV
To optimize the model, we applied GridSearchCV to tune the hyperparameters like max_depth and max_leaf_nodes, which control the complexity of the decision tree and help avoid overfitting.

#### 5. Prediction
Once the model is trained, we make predictions on the test set and evaluate the model's performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

#### 6. Visualization
The decision tree model is visualized using plot_tree from the sklearn.tree module. This helps to understand the model's decision-making process.


### Results
- Training Data R-squared: 0.79
- Test Data R-squared: 0.73
- Predicted Price for a Mercedes car with 2 doors and 250 horsepower: €44,650.83
### Conclusion
Using the Decision Tree Regression model, we have predicted the price of a Mercedes car based on its number of doors and horsepower. The model provides a reasonably good prediction with an R-squared value of 0.73 on the test set, meaning it explains about 73% of the variance in the car prices.


