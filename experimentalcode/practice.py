import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
num_rows = 2000

# Generate synthetic real-life-like features
age = np.random.randint(18, 65, num_rows)  # Age: 18 to 65
income = np.random.randint(20000, 150000, num_rows)  # Annual Income: $20k to $150k
education_years = np.random.randint(8, 20, num_rows)  # Years of education: 8 to 20
work_experience = np.random.randint(0, 45, num_rows)  # Work experience in years
loan_amount = np.random.randint(1000, 50000, num_rows)  # Loan amount: $1k to $50k
family_members = np.random.randint(1, 7, num_rows)  # Family size: 1 to 6 members
monthly_expenses = income * np.random.uniform(0.3, 0.5, num_rows)  # Expenses: 30%-50% of income
savings = income * np.random.uniform(0.1, 0.3, num_rows)  # Savings: 10%-30% of income
debt = loan_amount * np.random.uniform(0.2, 0.6, num_rows)  # Debt: 20%-60% of loan amount

# Generate a target variable (Affordability Index) as a linear combination of features + noise
affordability_index = (
    income * 0.3
    - monthly_expenses * 0.5
    + savings * 0.7
    - debt * 0.4
    + work_experience * 50
    - family_members * 100
    + education_years * 80
    + np.random.normal(0, 5000, num_rows)  # Adding noise
)

# Create a DataFrame
data = pd.DataFrame({
    "Age": age,
    "Income": income,
    "Education_Years": education_years,
    "Work_Experience": work_experience,
    "Loan_Amount": loan_amount,
    "Family_Members": family_members,
    "Monthly_Expenses": monthly_expenses,
    "Savings": savings,
    "Debt": debt,
    "Affordability_Index": affordability_index
})

# Save to CSV
file_name = "linear_regression_data.csv"
data.to_csv(file_name, index=False)
print(f"Data successfully generated and saved to {file_name}")
