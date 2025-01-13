# Salary Prediction Using Simple Linear Regression

## Project Overview

This project aims to predict the salary of an individual based on their years of experience using a simple linear regression model. The dataset used for this project contains information about individuals' years of experience and their corresponding salaries. The goal is to build a predictive model that can estimate salaries based on years of experience.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [License](#license)

## Dataset

The dataset used in this project is named `Salary_dataset.csv`. It contains the following columns:

- **Unnamed: 0**: An index column.
- **YearsExperience**: The number of years of experience an individual has.
- **Salary**: The corresponding salary of the individual.

The dataset consists of 30 entries, with no missing values.

## Technologies Used

- Python 3.x
- Pandas: For data manipulation and analysis.
- NumPy: For numerical operations.
- Matplotlib: For data visualization.
- Seaborn: For enhanced data visualization.
- Scikit-learn: For implementing the linear regression model and evaluation metrics.
- Jupyter Notebook: For interactive coding and visualization.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/salary-prediction.git
   cd salary-prediction
   ```

2. **Install the required packages**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   Using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Download the dataset**:
   Ensure that the `Salary_dataset.csv` file is in the project directory.

## Usage

To run the salary prediction model, you can use the Jupyter Notebook provided in the project. Follow these steps:

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open the `Salary-Prediction-Simple-Linear-Regression.ipynb` notebook.

3. Execute the cells in the notebook sequentially to load the data, train the model, and visualize the results.

### Key Steps in the Notebook

- **Data Loading**: The dataset is loaded into a Pandas DataFrame.
- **Data Exploration**: The first few rows, summary statistics, and information about the dataset are displayed.
- **Data Visualization**: Scatter plots and histograms are created to visualize the relationship between years of experience and salary.
- **Data Preprocessing**: Missing values are handled, and numerical features are normalized.
- **Model Training**: A linear regression model is created and trained on the training dataset.
- **Model Evaluation**: The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
- **Visualizing Predictions**: The model's predictions are compared against actual values through various plots.

## Model Evaluation

The model's performance is evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: The average of the squares of the errors.
- **Root Mean Squared Error (RMSE)**: The square root of the MSE, providing an error metric in the same units as the target variable.
- **R² Score**: Indicates the proportion of variance in the dependent variable that can be explained by the independent variable(s).

## Visualizations

The project includes several visualizations to help understand the data and the model's performance:

- **Scatter Plot**: Displays the relationship between years of experience and salary.
  ![Untitled](https://github.com/user-attachments/assets/98b1f7c3-e78f-4b96-8ae7-f24522053f99)

- **Histogram**: Shows the distribution of salaries.
  ![Untitled-1](https://github.com/user-attachments/assets/049b90a2-0ac8-4017-8a6f-926734ec6403)

- **Regression Line Plot**: Compares actual vs predicted values for the training data.
  ![Untitled](https://github.com/user-attachments/assets/2b498959-5e3a-4c4b-959b-53179d87e6b6)

- **Predicted vs Actual Values Plot**: Compares predicted vs actual values for the test set.
  ![Untitled-1](https://github.com/user-attachments/assets/18726456-a47d-4f6f-92ab-49e483de05b2)

- **Residuals Plot**: Visualizes the residuals to check for patterns that might indicate issues with the model.
  ![Untitled](https://github.com/user-attachments/assets/0f8332f4-4eff-4655-9220-77e20e4a4162)

## Conclusion

This project demonstrates the application of simple linear regression for salary prediction based on years of experience. The model shows a good fit, as indicated by the evaluation metrics. Future work could involve exploring more complex models or additional features to improve prediction accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
