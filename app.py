import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Heart Disease UCI dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
heart_data = pd.read_csv(url, names=names, na_values="?")

# Drop rows with missing values for simplicity in this example
heart_data = heart_data.dropna()

# Convert categorical variables to numerical using one-hot encoding
heart_data = pd.get_dummies(heart_data, columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

# Separate features and target variable
X = heart_data.drop("target", axis=1)
y = heart_data["target"]

# Create a Random Forest classifier with 100 trees
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X, y)

# Streamlit App
def main():
    st.title("Heart Disease Prediction App")

    # Collect user input
    age = st.slider("Select Age", min_value=1, max_value=100, value=25)
    sex = st.selectbox("Select Gender", ["Male", "Female"])
    cp = st.selectbox("Select Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Select Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.slider("Select Serum Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.slider("Select Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=2.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=1)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [1 if fbs == "Yes" else 0],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [1 if exang == "Yes" else 0],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    # Convert categorical variables to numerical using one-hot encoding
    input_data = pd.get_dummies(input_data, columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"])

    # Align the columns to match the training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Make predictions
    prediction = rf_classifier.predict(input_data)

    # Display the result
    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.write("The model predicts that the patient may have heart disease.")
    else:
        st.write("The model predicts that the patient may not have heart disease.")

if __name__ == "__main__":
    main()
