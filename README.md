# AI-Financial-Tracker <a href="https://ai-financial-tracker-1quh.onrender.com"></a> 
Built a responsive web application that analyzes historical expense data to forecast future spending trends. Implemented a machine learning pipeline leveraging Linear Regression and Random Forest Regressor to deliver accurate monthly budget predictions to users.

---

## 🔍 Features

- 📊 **Expense Forecasting** — Predict monthly spending using ML
- 🧠 **ML Models Used** — Linear Regression, Random Forest, Polynomial Regression
- 🔍 **Category-wise Prediction** — Estimate future spending by category
- 📈 **Insights & Tips** — AI-powered insights based on your habits
- 🧮 **Budget Tracking** — Set and manage monthly category budgets
- 📂 **Transaction History** — Add, edit, and view all financial records
- 🔐 **User Auth** — Secure login, registration, and sessions

---

## ⚙️ Tech Stack

| Layer        | Technology                           |
|--------------|--------------------------------------|
| Backend      | Flask (Python)                       |
| Database     | PostgreSQL                           |
| ML Models    | scikit-learn                         |
| Frontend     | HTML, Bootstrap (Jinja2 templates)   |
| Deployment   | Gunicorn                             |     
| Hosting      | Replit / Render / Railway compatible |

---

## 🧠 ML Overview

The app fetches monthly aggregated transaction data and uses:

- `RandomForestRegressor` for robust tree-based modeling
- `LinearRegression` for quick, interpretable predictions
- Optional: Polynomial Regression for non-linear patterns

Model selection is **dynamic**, based on historical data quality and performance (`R² score`).

---

