# Credit Score Predictor

This project predicts a customer's credit score using machine learning models. The implementation uses a Random Forest Classifier and a K-Nearest Neighbors (KNN) Classifier. The program follows object-oriented programming (OOP) principles, employs clean code practices, and includes a data analysis visualization.

## Features

- **Data Preprocessing**: Encodes categorical variables using Label Encoding.
- **Machine Learning Models**: Trains and evaluates Random Forest and KNN classifiers.
- **Data Visualization**: Displays the distribution of credit scores.
- **Modular Design**: The code is organized using OOP and separated into files for maintainability.

## Requirements

- Python 3.7+
- Libraries:
  - pandas
  - matplotlib
  - scikit-learn

## File Structure

- `main.py`: Main script to execute the program.
- `credit_score_predictor.py`: Contains the `CreditScorePredictor` class with all preprocessing, training, evaluation, and visualization logic.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vallz0/credit-score-predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-score-predictor
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset in the project directory and name it `clientes.csv`.
2. Run the program:
   ```bash
   python main.py
   ```
3. The program will preprocess the data, analyze it, train the models, and display the accuracy of each.

## Example Output

- **Data Visualization**:
  A bar chart showing the distribution of credit scores.

- **Model Accuracy**:
  ```
  Accuracy for RandomForest: 85.67%
  Accuracy for KNN: 78.45%
  ```

## Customization

- Update `clientes.csv` to use your own dataset.
- Modify the `CreditScorePredictor` class to adjust encoding, model parameters, or other behaviors.
