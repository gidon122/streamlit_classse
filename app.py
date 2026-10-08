import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Flower Prediction")
st.write("Adjust the sliders, then click Predict.")


@st.cache_resource
def train_model():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=42, test_size=0.2
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return iris, model, model.score(x_test, y_test)


iris, model, accuracy = train_model()
st.write(f"Model Accuracy: {accuracy:.2f}")

st.header("Describe your flower")

sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 3.7)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

features = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success(f"Predicted Species: {iris.target_names[prediction]}")