import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Flower Prediction")
st.write("select an option")

iris = load_iris()
x=iris.data
y=iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")
st.header("Confirm your flower")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)

sepal_width = st.slider("Sepal Width", 2.0, 4.4, 3.4)
sepal_length = st.slider("Sepal Length", 4.3, 7.9, 5.8)
petal_length = st.slider("Petal Length", 1.0, 6.9, 3.7)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3)

features = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.button("Predict"):
    predict = model.predict(features)[0]
    st.success(f"Predicted Species: {iris.target_names[predict]}")
predict = model.predict(features)[0]
st.write(predict)
