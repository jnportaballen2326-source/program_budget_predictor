import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpg;base64,{encoded_string});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Add semi-transparent overlay to improve text readability */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.85);
        z-index: -1;
    }}
    
    /* Style the main content area */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Style the sidebar */
    section[data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.95);
    }}
    
    /* Make ALL text black for better readability */
    .stApp, .main, .block-container, .stSidebar, 
    h1, h2, h3, h4, h5, h6, p, span, div, label,
    .stMarkdown, .stText, .stNumberInput, .stSelectbox,
    .stRadio, .stButton, .stMetric, .stTab, .stTabs {{
        color: #000000 !important;
    }}
    
    </style>
    """
    return bg_image

# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Log transformation
def log_transform_target(y):
    return np.log1p(y)

def main():
    # Add background image
    bg_image = add_bg_from_local('background.jpg')
    st.markdown(bg_image, unsafe_allow_html=True)
    
    # Custom CSS for additional styling with black text + REMOVED STEPPERS
    st.markdown("""
    <style>

    /* 🚫 Remove number input +/- stepper buttons */
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield;
    }

    /* Main text styling */
    .stApp, .main, .block-container {
        color: #000000 !important;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #000000 !important;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .prediction-box h3 {
        color: #000000 !important;
        margin-bottom: 20px;
        font-weight: bold;
    }
    
    .prediction-box p {
        color: #000000 !important;
        opacity: 0.9;
    }
    
    .prediction-amount {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
        color: #000000 !important;
    }
    
    .stButton>button {
        background-color: #1f3c5f;
        color: #000000 !important;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #2c5282;
        color: #000000 !important;
    }
    
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("Program Budget Predictor")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Program Parameters")
        
        st.markdown("### Program Details")
        
        participants = st.number_input(
            "**Number of Participants:**",
            min_value=1,
            value=30,
            help="Enter the total number of participants expected"
        )
        
        duration = st.number_input(
            "**Duration of Program (hours):**",
            min_value=1.0,
            value=10.0,
            step=0.5,
            help="Enter the total duration in hours"
        )
        
        staffs = st.number_input(
            "**Number of Staff Members:**",
            min_value=1,
            value=12,
            help="Enter the number of staff required"
        )
        
        st.markdown("---")
        st.markdown("### Program Settings")
        
        program_type = st.radio(
            "**Select Program Type:**",
            ['Competition Program', 'Modeling Program', 'Seminar Program',
             'Sport Program', 'Training Program', 'Workshop Program'],
            help="Choose the type of program"
        )
        
        month = st.selectbox(
            "**Select Month:**",
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November', 'December'],
            help="Select the month when the program will be held"
        )
        
        st.markdown("---")
        predict_button = st.button("Calculate Budget", type="primary", use_container_width=True)

    # Load Data
    try:
        data_file = "PROGRAM_SANTACRUZ_DATA.csv"
        data = load_data(data_file)
    except:
        st.error("Data file missing.")
        return

    try:
        X = data[['Number of Participants', 'Duration of Program/HR', 'Staffs',
                  'Program_Competition', 'Program_Modeling', 'Program_Seminar',
                  'Program_Sport', 'Program_Training', 'Program_Workshop',
                  'Month_January', 'Month_February', 'Month_March',
                  'Month_April', 'Month_May', 'Month_June', 'Month_July',
                  'Month_August', 'Month_September', 'Month_October',
                  'Month_November', 'Month_December']]

        y = data["Budget"]
        y_transformed = log_transform_target(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=0.2, random_state=42
        )

        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)

        program_list = ['Competition Program', 'Modeling Program', 'Seminar Program',
                        'Sport Program', 'Training Program', 'Workshop Program']
        program_one_hot = [1 if program_type == p else 0 for p in program_list]

        month_list = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        month_one_hot = [1 if month == m else 0 for m in month_list]

        input_data = [[participants, duration, staffs, *program_one_hot, *month_one_hot]]
        input_df = pd.DataFrame(input_data, columns=X.columns)

        prediction = model.predict(input_df)
        predicted_budget = np.expm1(prediction[0])

        test_pred = model.predict(X_test)
        test_pred_exp = np.expm1(test_pred)
        actual_exp = np.expm1(y_test)

        test_mse = mean_squared_error(actual_exp, test_pred_exp)
        test_r2 = r2_score(actual_exp, test_pred_exp)

        st.markdown(f"""
        <div class="prediction-box">
            <h3>Budget Prediction</h3>
            <div class="prediction-amount">₱{predicted_budget:,.2f}</div>
            <p>Based on {participants} participants, {duration} hours, and {staffs} staff members</p>
            <p>Program Type: {program_type} | Month: {month}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #000000 !important;'>"
        "Program Budget Predictor | Using Decision Tree Regression"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
