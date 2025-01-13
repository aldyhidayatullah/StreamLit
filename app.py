import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_path = "Regression.csv"
data = None
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("Dataset tidak ditemukan! Pastikan 'Regression.csv' ada di path yang benar.")
    
# Periksa apakah dataset mengandung nilai NaN atau Infinity
if data is not None:
    if data.isnull().sum().sum() > 0:
        st.error("Data mengandung nilai NaN. Silakan bersihkan dataset.")
    if (data == float('inf')).sum().sum() > 0:
        st.error("Data mengandung nilai Infinity. Silakan bersihkan dataset.")
        
# Streamlit App
def main():
    if data is None:
        return

    st.set_page_config(
        page_title="Regression Analysis App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styling Custom CSS for background, header, and sidebar
    st.markdown("""
        <style>
            /* Global styling */
            body {
                background-color: black;
                font-family: 'Arial', sans-serif;
                color: #333;
            }

            /* Header Styling */
            .header {
                text-align: center;
                color: #8E1616;
                font-size: 40px;
                font-weight: 600;
                margin-bottom: 20px;
                animation: fadeIn 2s ease-in-out;
            }

            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            /* Sidebar Styling */
            .sidebar .sidebar-content {
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            }

            .sidebar .sidebar-content .element-container {
                padding: 20px;
            }

            /* Section Title Styling */
            .section-title {
                color: #44c8b1;
                font-size: 30px;
                font-weight: bold;
                margin-bottom: 15px;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            }

            /* Main content styling */
            .main {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 35px;
                border-radius: 20px;
                box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.2);
            }

            .subheader {
                font-size: 24px;
                color: #D84040;
                text-align: center;
                margin-bottom: 25px;
            }

            /* Dataframe Styling */
            .dataframe {
                border-radius: 12px;
                border: 2px solid #44c8b1;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                padding: 15px;
            }

            /* Button Styling */
            .stButton>button {
                background-color: #44c8b1;
                color: black;
                border-radius: 8px;
                padding: 12px 25px;
                font-size: 18px;
                font-weight: 600;
                border: none;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s ease, transform 0.2s;
            }

            .stButton>button:hover {
                background-color: #1e7f5b;
                transform: translateY(-2px);
            }

            /* Input Styling */
            .stNumberInput input {
                background-color: #fff;
                color: #333;
                font-size: 18px;
                padding: 10px;
                border: 2px solid #44c8b1;
                border-radius: 8px;
                transition: border-color 0.3s;
            }
             .custom-text {
            color: #44c8b1;
            }
             .custom-font {
            color: #E50000;
            }
            .stNumberInput input:focus {
                border-color: #1e7f5b;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Regression Analysis App")
    st.sidebar.header("📂 Navigation")

    # Sidebar navigation
    options = ["🏠 Home", "📈 Data Exploration", "📊 Visualization", "🤖 Prediction"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "🏠 Home":
        st.markdown(
            """<div class="header">Selamat Datang di Regression Analysis App!</div>
            <p class="subheader">Analisis Data, Visualisasikan Wawasan, dan Prediksi Hasil Secara Mudah.</p>""",
            unsafe_allow_html=True,
        )
        image_url ="https://i.imgur.com/MdMCXbW.png"
        st.image(image_url, caption="Data Analysis App",)

    elif choice == "📈 Data Exploration":
        st.markdown("<div class='section-title'>🔍 Data Exploration</div>", unsafe_allow_html=True)
        st.markdown('<p class="custom-font">Dataset Overview</p>', unsafe_allow_html=True)
        st.dataframe(data.head(), use_container_width=True)

        st.markdown('<p class="custom-font">Summary Statistics</p>', unsafe_allow_html=True)
        st.dataframe(data.describe(), use_container_width=True)

        st.markdown('<p class="custom-font">Data Types</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data.dtypes, columns=["Type"]).reset_index().rename(columns={"index": "Column"}), use_container_width=True)

    elif choice == "📊 Visualization":
        st.markdown("<div class='section-title'>📊 Data Visualization</div>", unsafe_allow_html=True)

        # Filter hanya kolom numerik
        numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

        st.sidebar.subheader("Scatter Plot Settings")
        col1 = st.sidebar.selectbox("Select X-axis", numeric_columns)
        col2 = st.sidebar.selectbox("Select Y-axis", numeric_columns)
        if col1 and col2:
            st.markdown('<p class="custom-text">Scatter Plot: COL 1 VS COL 2</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.scatterplot(data=data, x=col1, y=col2, ax=ax)
            st.pyplot(fig)

            st.markdown('<p class="custom-text">Correlation Heatmap</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif choice == "🤖 Prediction":
        st.markdown("<div class='section-title'>🤖 Prediction using Support Vector Regression</div>", unsafe_allow_html=True)

        features = ["age", "bmi", "children"]
        if all(feature in data.columns for feature in features) and "charges" in data.columns:
            X = data[features]
            y = data["charges"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model SVR
            model = SVR(kernel="rbf", C=100, epsilon=0.1)
            model.fit(X_train, y_train)

            # Model performance
            y_pred = model.predict(X_test)
            st.markdown("<span style='color:red;'>Model Perfomance</span>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#44c8b1;'>Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:#44c8b1;'>R-squared: {r2_score(y_test, y_pred):.2f}</h3>", unsafe_allow_html=True)

            # Debugging output
            st.markdown(f"<h3 style='color:#44c8b1;'>Predicted Values (some examples): {y_pred[:5]}</h3>", unsafe_allow_html=True)

            st.markdown("<span style='color:red;'>Prediksi Charges</span>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#44c8b1;'>Age</h3>", unsafe_allow_html=True)
            age = st.number_input("", min_value=0, max_value=100, value=30, step=1)

            st.markdown("<h3 style='color:#44c8b1;'>BMI</h3>", unsafe_allow_html=True)
            bmi = st.number_input("", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

            st.markdown("<h3 style='color:#44c8b1;'>Children</h3>", unsafe_allow_html=True)
            children = st.number_input("", min_value=0, max_value=10, value=0, step=1)


            if st.button("Predict"):
                pred = model.predict([[age, bmi, children]])[0]
                st.write(f"Debugging: Predicted Charges for Age={age}, BMI={bmi}, Children={children}: {pred:.2f}")
                st.success(f"Predicted Charges: {pred:.2f}")
        else:
            st.error("Kolom yang dibutuhkan untuk prediksi tidak ada dalam dataset.")

if __name__ == "__main__":
    main()
