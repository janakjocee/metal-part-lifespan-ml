🚀 Metal Part Lifespan Prediction using Machine Learning
📌 Project Overview

This project implements a complete end-to-end Machine Learning pipeline to predict the lifespan of manufactured metal parts using production parameters.

The business objective is twofold:

Regression Task – Predict the exact lifespan of a metal part.

Classification Task – Determine whether a part is safe for deployment based on a lifespan threshold.

The project includes full experimentation, hyperparameter tuning, performance comparison, and a final deployment recommendation.

🏭 Business Context

In manufacturing environments, destructive lifespan testing is expensive and time-consuming.

By using machine learning models trained on measurable production parameters, we can:

Estimate product longevity without destructive testing

Reduce manufacturing waste

Improve process optimization

Support data-driven production decisions

🧠 Machine Learning Implementation
🔹 Regression Models

Linear Regression

[Your second model – e.g., Random Forest / Neural Network]

Evaluated using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

🔹 Classification Models

Logistic Regression

[Your second model – e.g., ANN / Decision Tree]

Evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

⚙️ Methodology

The project follows a structured ML workflow:

Data Loading

Exploratory Data Analysis

Feature Selection

Data Preprocessing (Scaling, Splitting)

Model Training

Hyperparameter Tuning

Performance Evaluation

Model Comparison

Deployment Recommendation

All experiments use consistent train-test splits to ensure fair model comparison.

📊 Key Insights

Non-linear models demonstrated improved predictive performance.

Feature scaling significantly impacted neural network performance.

Classification provides clearer operational decision support.

Model evaluation was aligned with real-world business priorities.

🗂 Repository Structure
metal-part-lifespan-ml/
│
├── notebook/        → Jupyter Notebook implementation
├── report/          → Full technical report
├── images/          → Visualizations used in analysis
├── requirements.txt → Python dependencies
└── README.md        → Project documentation

🛠 Tech Stack

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

(TensorFlow / Keras if used)

📎 Reproducibility

To run the notebook:

pip install -r requirements.txt


Then open the notebook in Jupyter or Google Colab and execute all cells.

⚠️ Academic Integrity Notice

This repository is shared for educational and reference purposes only.

If you are working on a similar academic assignment, use this project to understand methodology and experimentation strategies — do not copy solutions directly.

👤 Author

Janak
MSc Data Science Candidate | Data Analytics | Machine Learning
