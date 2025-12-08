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
    
    /* Style headers with black color */
    h1, h2, h3 {{
        color: #000000 !important;
    }}
    
    /* Style metric cards with black text */
    .stMetric {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #1f3c5f;
        color: #000000 !important;
    }}
    
    .stMetric label, .stMetric div {{
        color: #000000 !important;
    }}
    
    /* Style sidebar text */
    .stSidebar * {{
        color: #000000 !important;
    }}
    
    /* Style input labels */
    .stNumberInput label, .stSelectbox label, .stRadio label {{
        color: #000000 !important;
    }}
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {{
        color: #000000 !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: #000000 !important;
    }}
    
    /* Style error messages */
    .stAlert {{
        color: #000000 !important;
    }}
    
    .stAlert div {{
        color: #000000 !important;
    }}
    
    /* Style footer */
    footer {{
        color: #000000 !important;
    }}
    </style>
    """
    return bg_image

# Load data from CSV
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
    
    # Custom CSS for additional styling with black text and removing increment buttons
    st.markdown("""
    <style>
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
    
    /* Input field styling */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Radio button styling */
    .stRadio div {
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
        font-weight: bold;
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    /* Divider styling */
    hr {
        border-color: rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Footer styling */
    footer {
        color: #000000 !important;
    }
    
    /* Remove increment/decrement buttons from number input - SIMPLIFIED */
    .stNumberInput button {
        display: none !important;
    }
    
    /* Hide the increment/decrement buttons */
    div[data-baseweb="input"] > div:nth-child(3),
    div[data-baseweb="input"] > div:nth-child(4) {
        display: none !important;
    }
    
    /* Make the input field wider since we removed the buttons */
    .stNumberInput input {
        width: 100% !important;
        padding-right: 10px !important;
    }
    
    /* Remove spinner buttons from number input */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    
    input[type="number"] {
        -moz-appearance: textfield;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Program Budget Predictor")
    st.markdown("---")
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.header("Program Parameters")
        
        # Add some visual separation
        st.markdown("### Program Details")
        
        # Number inputs WITHOUT increment/decrement buttons
        # These will now show as plain text boxes where users can type directly
        participants = st.number_input(
            "**Number of Participants:**",
            min_value=1,
            value=30,
            help="Enter the total number of participants expected",
            step=1,
            key="participants"
        )
        
        duration = st.number_input(
            "**Duration of Program (hours):**",
            min_value=1.0,
            value=10.0,
            step=0.5,
            help="Enter the total duration in hours",
            key="duration"
        )
        
        staffs = st.number_input(
            "**Number of Staff Members:**",
            min_value=1,
            value=12,
            help="Enter the number of staff required",
            step=1,
            key="staffs"
        )
        
        # Add inline help text
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #1f3c5f;">
        <small>💡 <strong>Tip:</strong> Click on any input field above and type your numbers directly.</small>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Add a prediction button
        predict_button = st.button("Calculate Budget", type="primary", use_container_width=True)

    # Load data
    try:
        data_file = "PROGRAM_SANTACRUZ_DATA.csv"
        data = load_data(data_file)
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'PROGRAM_SANTACRUZ_DATA.csv' is in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

    # Prepare features
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

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_transformed, test_size=0.2, random_state=42
        )

        # Train model
        model = DecisionTreeRegressor(random_state=42)
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

        # Display predictions in a styled box - changed text to black
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color: #000000 !important; margin-bottom: 20px; font-weight: bold;">Budget Prediction</h3>
            <div class="prediction-amount" style="color: #000000 !important;">₱{predicted_budget:,.2f}</div>
            <p style="color: #000000 !important; opacity: 0.9;">Based on {participants} participants, {duration} hours, and {staffs} staff members</p>
            <p style="color: #000000 !important; opacity: 0.9;">Program Type: {program_type} | Month: {month}</p>
        </div>
        """, unsafe_allow_html=True)

        # Model Performance Metrics
        st.markdown("---")
        st.header("Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{test_mse:,.2f}")
        with col2:
            st.metric("R² Score", f"{test_r2:.2%}")

        # Visualizations
        st.markdown("---")
        st.header("Visualizations")

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Error Distribution", "Monthly Analysis"])

        with tab1:
            # Scatter plot: Actual vs Predicted
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(actual_exp, test_pred_exp, alpha=0.6, color='steelblue')
            ax1.plot([actual_exp.min(), actual_exp.max()],
                     [actual_exp.min(), actual_exp.max()],
                     'r--', label="Perfect Prediction", linewidth=2)
            ax1.set_title("Actual vs Predicted Budget", fontsize=14, fontweight='bold', color='black')
            ax1.set_xlabel("Actual Budget (₱)", fontsize=12, color='black')
            ax1.set_ylabel("Predicted Budget (₱)", fontsize=12, color='black')
            ax1.tick_params(colors='black')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            # Set plot background to white
            fig1.patch.set_facecolor('white')
            ax1.set_facecolor('white')
            st.pyplot(fig1)

        with tab2:
            # Error distribution
            errors = actual_exp - test_pred_exp
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(errors, bins=20, edgecolor='black', color='lightcoral', alpha=0.7)
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax2.set_title("Distribution of Prediction Errors", fontsize=14, fontweight='bold', color='black')
            ax2.set_xlabel("Error (Actual - Predicted) in ₱", fontsize=12, color='black')
            ax2.set_ylabel("Frequency", fontsize=12, color='black')
            ax2.tick_params(colors='black')
            ax2.grid(True, alpha=0.3)
            # Set plot background to white
            fig2.patch.set_facecolor('white')
            ax2.set_facecolor('white')
            st.pyplot(fig2)

        with tab3:
            # Monthly prediction chart
            monthly_predictions = []
            for m in month_list:
                month_hot = [1 if m == mo else 0 for mo in month_list]
                input_row = [[participants, duration, staffs, *program_one_hot, *month_hot]]
                pred = model.predict(pd.DataFrame(input_row, columns=X.columns))
                monthly_predictions.append(np.expm1(pred[0]))

            fig3, ax3 = plt.subplots(figsize=(12, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(month_list)))
            bars = ax3.bar([m[:3] for m in month_list], monthly_predictions, color=colors)
            
            # Highlight current selected month
            current_month_index = month_list.index(month)
            bars[current_month_index].set_edgecolor('red')
            bars[current_month_index].set_linewidth(3)
            
            ax3.set_title(f"Predicted Budget by Month\n(Current selection: {month})", 
                         fontsize=14, fontweight='bold', color='black')
            ax3.set_xlabel("Month", fontsize=12, color='black')
            ax3.set_ylabel("Predicted Budget (₱)", fontsize=12, color='black')
            ax3.tick_params(colors='black')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'₱{height:,.0f}', ha='center', va='bottom', fontsize=9, color='black')
            
            # Set plot background to white
            fig3.patch.set_facecolor('white')
            ax3.set_facecolor('white')
            st.pyplot(fig3)

        # Display summary statistics
        st.markdown("---")
        st.header("Summary Statistics")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Program Type", program_type)
        with summary_col2:
            st.metric("Selected Month", month)
        with summary_col3:
            st.metric("Participant Count", participants)

    except KeyError as e:
        st.error(f"Missing required column in data: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

    # Footer with black text
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #000000 !important;'>"
        "Program Budget Predictor | Using Decision Tree Regression | Data Source: PROGRAM_SANTACRUZ_DATA.csv"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
