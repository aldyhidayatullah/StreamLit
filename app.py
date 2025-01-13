import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_path = "Regression.csv"
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    st.error("Dataset tidak ditemukan! Pastikan 'Regression.csv' ada di path yang benar.")
    data = None

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
            /* Background and layout */
            body {
                background-color: #f0f4f8;
                font-family: 'Arial', sans-serif;
                color: #333;
            }
            
            /* Header Styling */
            .header {
                text-align: center;
                color: #4CAF50;
                font-size: 36px;
                font-weight: bold;
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
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }

            .sidebar .sidebar-content .element-container {
                padding: 15px;
            }

            /* Section Title Styling */
            .section-title {
                color: #3e8e41;
                font-size: 26px;
                font-weight: bold;
                margin-bottom: 10px;
            }

            /* Main content styling */
            .main {
                background-color: rgba(255, 255, 255, 0.8);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            }

            .subheader {
                font-size: 22px;
                color: #333;
                text-align: center;
                margin-bottom: 20px;
            }

            /* Dataframe Styling */
            .dataframe {
                border-radius: 10px;
                border: 2px solid #4CAF50;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }

            /* Button Styling */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s ease;
            }

            .stButton>button:hover {
                background-color: #45a049;
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
        st.image("data.png", caption="Data Analysis App", use_container_width=True, width=300)

    elif choice == "📈 Data Exploration":
        st.markdown("<div class='section-title'>🔍 Data Exploration</div>", unsafe_allow_html=True)
        st.write("### Dataset Overview")
        st.dataframe(data.head(), use_container_width=True)

        st.write("### Summary Statistics")
        st.dataframe(data.describe(), use_container_width=True)

        st.write("### Data Types")
        st.dataframe(pd.DataFrame(data.dtypes, columns=["Type"]).reset_index().rename(columns={"index": "Column"}), use_container_width=True)

    elif choice == "📊 Visualization":
        st.markdown("<div class='section-title'>📊 Data Visualization</div>", unsafe_allow_html=True)

        # Filter hanya kolom numerik
        numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns

        st.sidebar.subheader("Scatter Plot Settings")
        col1 = st.sidebar.selectbox("Select X-axis", numeric_columns)
        col2 = st.sidebar.selectbox("Select Y-axis", numeric_columns)

        # Buat scatter plot jika kedua kolom valid
        if col1 and col2:
            st.write(f"### Scatter Plot: {col1} vs {col2}")
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=data,
                x=col1,
                y=col2,
                hue=data["smoker"] if "smoker" in data else None,
                style=data["sex"] if "sex" in data else None,
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.warning("Silakan pilih kolom numerik untuk sumbu X dan Y.")

        st.write("### Correlation Heatmap")
        if not numeric_columns.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Tidak ada kolom numerik untuk heatmap korelasi.")

    elif choice == "🤖 Prediction":
        st.markdown("<div class='section-title'>🤖 Prediksi</div>", unsafe_allow_html=True)

        # Pemilihan fitur
        features = ["age", "bmi", "children"]
        if all(feature in data.columns for feature in features) and "charges" in data.columns:
            X = data[features]
            y = data["charges"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Menampilkan metrik
            y_pred = model.predict(X_test)
            st.write("### Model Performance")
            st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
            st.metric("R-squared", f"{r2_score(y_test, y_pred):.2f}")

            st.write("### Model Feature Importances")
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.dataframe(importance_df, use_container_width=True)

            # User input untuk prediksi
            st.write("### Prediksi Charges")
            age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            children = st.number_input("Children", min_value=0, max_value=10, value=0, step=1)

            # Tombol prediksi
            if st.button("Predict"):
                pred = model.predict([[age, bmi, children]])[0]
                st.success(f"Predicted Charges: {pred:.2f}")
        else:
            st.error("Kolom yang dibutuhkan untuk prediksi tidak ada dalam dataset.")

if __name__ == "__main__":
    main()
