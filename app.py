import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load data from CSV
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Log transformation
def log_transform_target(y):
    return np.log1p(y)

def main():
    st.title("Program Budget Predictor")

    # Sidebar inputs
    st.sidebar.header("User Input")
    participants = st.sidebar.number_input("Enter the number of participants:", min_value=1, value=30)
    duration = st.sidebar.number_input("Enter the duration of the program (in hours):", min_value=1.0, value=10.0)
    staffs = st.sidebar.number_input("Enter the number of staff members:", min_value=1, value=12)

    program_type = st.sidebar.radio(
        "Select the type of program:",
        ['Competition Program', 'Modeling Program', 'Seminar Program',
         'Sport Program', 'Training Program', 'Workshop Program']
    )

    month = st.sidebar.selectbox(
        "Select the month:",
        ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']
    )

    # Load data
    data_file = "PROGRAM_TAGOLOAN_DATA.csv"
    data = load_data(data_file)

    # Prepare features
    X = data[['Number of Participants', 'Duration of Program/HR', 'Staffs',
              'Program_Competition', 'Program_Modeling', 'Program_Seminar',
              'Program_Sport', 'Program_Training', 'Program_Workshop',
              'Month_January', 'Month_February', 'Month_March',
              'Month_April', 'Month_May', 'Month_June', 'Month_July',
              'Month_August', 'Month_September', 'Month_October',
              'Month_November', 'Month_December']]

    y = data["Budget"]
    y_transformed = log_transform_target(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42
    )

    # Train model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # One-hot program selection
    program_list = ['Competition Program', 'Modeling Program', 'Seminar Program',
                    'Sport Program', 'Training Program', 'Workshop Program']
    program_one_hot = [1 if program_type == p else 0 for p in program_list]

    # One-hot month selection
    month_list = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_one_hot = [1 if month == m else 0 for m in month_list]

    input_data = [[participants, duration, staffs, *program_one_hot, *month_one_hot]]
    input_df = pd.DataFrame(input_data, columns=X.columns)

    # Prediction
    prediction = model.predict(input_df)
    predicted_budget = np.expm1(prediction[0])

    # Model evaluation
    test_pred = model.predict(X_test)
    test_pred_exp = np.expm1(test_pred)
    actual_exp = np.expm1(y_test)

    test_mse = mean_squared_error(actual_exp, test_pred_exp)
    test_r2 = r2_score(actual_exp, test_pred_exp)

    # Display predictions
    st.write(f"### Predicted Budget: **₱{predicted_budget:,.2f}**")
    st.write("### Model Performance on Test Set:")
    st.write(f"- **MSE:** {test_mse:.2f}")
    st.write(f"- **R² Score:** {test_r2:.2f}")

    # Scatter plot: Actual vs Predicted
    fig2, ax2 = plt.subplots()
    ax2.scatter(actual_exp, test_pred_exp, alpha=0.5)
    ax2.plot([actual_exp.min(), actual_exp.max()],
             [actual_exp.min(), actual_exp.max()],
             'r--', label="Perfect Prediction")
    ax2.set_title("Actual vs Predicted Budget")
    ax2.set_xlabel("Actual Budget")
    ax2.set_ylabel("Predicted Budget")
    ax2.legend()
    st.pyplot(fig2)

    # Error distribution
    errors = actual_exp - test_pred_exp
    fig3, ax3 = plt.subplots()
    ax3.hist(errors, bins=20, edgecolor='black')
    ax3.set_title("Distribution of Prediction Errors")
    ax3.set_xlabel("Error (Actual - Predicted)")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    # Monthly prediction chart
    monthly_predictions = []
    for m in month_list:
        month_hot = [1 if m == mo else 0 for mo in month_list]
        input_row = [[participants, duration, staffs, *program_one_hot, *month_hot]]
        pred = model.predict(pd.DataFrame(input_row, columns=X.columns))
        monthly_predictions.append(np.expm1(pred[0]))

    fig4, ax4 = plt.subplots()
    ax4.bar([m[:3] for m in month_list], monthly_predictions)
    ax4.set_title("Predicted Budget by Month")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Predicted Budget")
    st.pyplot(fig4)

if __name__ == "__main__":
    main()

