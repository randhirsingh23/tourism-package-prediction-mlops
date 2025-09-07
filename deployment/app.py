
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# basic page setup for streamlit app
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🧳",
    layout="centered"
)

# load model only once (cached for speed)
@st.cache_resource
def load_model():
    # download the model file from my Hugging Face repo
    model_path = hf_hub_download(
        repo_id="RandhirSingh23/tourism-xgboost-classifier",
        filename="model.joblib",
    )
    # open the saved pipeline (this has preprocessing + xgboost inside)
    model = joblib.load(model_path)
    return model

model = load_model()

# app title and intro text
st.title("Tourism Package Prediction App")
st.write("Fill the form with customer details and check if they are likely to buy the package.")

# form for user inputs
with st.form("prediction_form"):

    # dropdowns for categorical values
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

    # number inputs for numeric values
    Age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
    CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2, step=1)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=200, value=20, step=1)
    NumberOfPersonVisiting = st.number_input("Number Of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
    NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=2, step=1)
    PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3, step=1)
    NumberOfTrips = st.number_input("Number Of Trips (per year)", min_value=0, max_value=50, value=1, step=1)
    Passport = st.selectbox("Has Passport?", [0, 1])
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    OwnCar = st.selectbox("Owns Car?", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (<5 yrs)", min_value=0, max_value=10, value=0, step=1)
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=40000, step=1000)

    # button to trigger prediction
    submitted = st.form_submit_button("Predict")

# if button clicked -> run prediction
if submitted:
    # make a dataframe with user inputs (1 row only)
    input_df = pd.DataFrame([{
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "ProductPitched": ProductPitched,
        "MaritalStatus": MaritalStatus,
        "Designation": Designation,
        "Age": Age,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
    }])

    # model prediction
    proba = model.predict_proba(input_df)[:, 1][0]  # probability of class 1
    pred = int(proba >= 0.5)  # class (1 if proba >= 0.5)

    # show output
    st.subheader("Prediction")
    st.write(f"Probability of Purchase (ProdTaken=1): {proba:.3f}")
    st.write(f"Predicted Class: {pred}  (1 = will purchase, 0 = will not)")

    # simple guidance text for user
    if pred == 1:
        st.success("This customer looks like a likely buyer.")
    else:
        st.info("This customer is unlikely to buy.")
