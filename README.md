# 🏀 NBA Data Preprocessing for Predictive Modeling

This project demonstrates a comprehensive data preprocessing pipeline designed to transform a raw NBA player dataset into a clean, high-quality format ready for machine learning, particularly for predictive modeling of player salaries.

The script automates several crucial steps: data cleaning, feature engineering, handling multicollinearity, and data transformation. The goal is to produce a robust feature matrix (`X`) and target vector (`y`) that can be fed into a linear model.

## ✨ Features

- **Data Cleaning**:
    - Handles missing values in both numerical and categorical columns.
    - Parses and converts complex string columns (e.g., `height`, `weight`, `salary`) into numerical formats.
    - Corrects and standardizes data types, including `datetime` conversions.
- **Feature Engineering**:
    - Creates new, insightful features like `age`, `experience`, and `bmi` from existing data.
- **Feature Selection**:
    - Automatically identifies and removes high-cardinality categorical features that are unsuitable for one-hot encoding.
    - Implements a multicollinearity check to drop redundant features, keeping the one more correlated with the target variable (`salary`).
- **Data Transformation**:
    - Standardizes all numerical features using `StandardScaler` to bring them to a common scale.
    - Applies `OneHotEncoder` to categorical features to convert them into a machine-readable format.
- **Data Splitting**:
    - Includes a function to split the final dataset into training, validation, and test sets (60/20/20 split).

## 🛠️ Technologies Used

*   **Core**: Python
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning Preprocessing**: Scikit-learn
*   **Data Fetching**: Requests

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.8+

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/michalglomsky/NBA-Data-Preprocessing.git
    cd NBA-Data-Preprocessing
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: The code from your project should be saved as a Python file, e.g., `main.py` or `preprocess.py`)*

## 🏃‍♀️ Usage

To run the full preprocessing pipeline:

1.  **Run the script:**
    ```sh
    python main.py 
    ```

2.  **Observe the output:**

    The script will perform the following actions automatically:
    - Check for the dataset and download it if it's missing.
    - Execute all cleaning, feature engineering, and transformation steps.
    - Print a dictionary containing the final shape of the feature matrix `X` and target vector `y`, along with a list of all feature names in the final dataset.

    The output will look like this:
    ```
    {'shape': [(453, 52), (453,)], 'features': ['rating', 'jersey', 'age', 'experience', 'bmi', 'No Team', 'team_Atlanta Hawks', ...]}
    ```

## 📄 License

This project is licensed under the MIT License.

---

Created by Michał Głomski

