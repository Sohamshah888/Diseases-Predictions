### README.md

---

# Disease Prediction Using Machine Learning

## Introduction
This project involves the application of various machine learning techniques to predict diseases based on symptoms. The dataset consists of 133 columns, with 132 representing different symptoms and the last column denoting the predicted outcome or prognosis. The symptoms are mapped to 42 diseases, and the goal is to train and test models using this data to achieve accurate predictions.

## Problem Statement
The dataset comprises two CSV files containing a total of 133 columns each. Out of these columns, 132 represent various symptoms that an individual may exhibit. The last column denotes the predicted outcome or prognosis of the disease. These symptoms are mapped to 42 diseases. The objective was to train the model on training data and test it on testing data using different classification models.

## Data Preparation
We began by cleaning the data to remove inconsistencies. Key steps included:
- Calculating the mean of each feature to check for normalization opportunities.
- Removing features such as 'fluid_overload,' which had no significant effect on the output when removed.

## Approach and Algorithms Used
The following machine learning algorithms were implemented and compared:

### 1. Linear Regression
- **Description:** A statistical method used to model the relationship between a dependent variable and one or more independent variables.
- **Implementation:** Applied on the training dataset to predict the accuracy of the prognosis.

### 2. Support Vector Machine (SVM)
- **Description:** A supervised learning algorithm used for classification and regression, focusing on finding a hyperplane that best separates classes in the data.
- **Implementation:** SVM was applied using different kernel functions to predict the prognosis.

### 3. Decision Tree
- **Description:** A machine learning algorithm used for both regression and classification, which builds a tree-like structure of decisions and consequences.
- **Implementation:** Used to predict how the independent variables affect the target variable.

### 4. Random Forest
- **Description:** An ensemble method combining multiple decision trees to improve prediction accuracy.
- **Implementation:** Trained multiple decision trees on random subsets of the training data and features, and aggregated their predictions.

### 5. XGBoost
- **Description:** An ensemble machine learning algorithm using gradient boosting and multiple decision trees to make predictions.
- **Implementation:** Applied for its speed and accuracy, particularly on large datasets with high dimensionality.

## Performance Evaluation
The models were evaluated based on their accuracy scores:

- **Linear Regression:** 0.966
- **Support Vector Machine (SVM):** 1.000
- **Decision Tree:** 0.971
- **Random Forest:** 0.976
- **XGBoost:** 0.920

## Conclusion
After conducting a comparative analysis of the machine learning algorithms on the symptom-based classification problem, it was concluded that the Support Vector Machine (SVM) performed the best with an accuracy score of 1.000. The SVM model's ability to handle high-dimensional data and find the optimal hyperplane made it a suitable choice for this problem. While the Decision Tree was easier to interpret and visualize, it lacked the accuracy and robustness of SVM.

## How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Disease-Prediction-ML.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd Disease-Prediction-ML
   ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook Disease_Prediction_Analysis.ipynb
   ```

## Dependencies
- numpy==1.21.0
- pandas==1.3.0
- scikit-learn==0.24.2
- matplotlib==3.4.2
- xgboost==1.4.2
