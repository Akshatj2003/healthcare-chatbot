import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import tempfile

# Streamlit UI to upload CSV files
st.title("Disease Prediction System")

# Upload training dataset and doctor dataset
training_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
doctor_file = st.file_uploader("Upload Doctor Dataset (CSV)", type=["csv"])

if training_file and doctor_file:
    # Load datasets from uploaded files
    training_dataset = pd.read_csv(training_file)
    doc_dataset = pd.read_csv(doctor_file, names=['Name', 'Description'])

    # Preprocess data
    X = training_dataset.iloc[:, :-1].values
    y = training_dataset.iloc[:, -1].values

    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Train Decision Tree Classifier
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(X_train, y_train)

    # Symptom list
    symptom_list = training_dataset.columns[:-1].str.lower().tolist()

    # Prepare doctor dataset
    diseases = pd.DataFrame(training_dataset['prognosis'].unique(), columns=['Disease'])
    doctors = pd.DataFrame({
        'disease': diseases['Disease'],
        'name': doc_dataset['Name'],
        'link': doc_dataset['Description']
    })

    # Function to generate PDF
    def generate_pdf(user_info, predicted_disease, doctor_info, selected_symptoms, fig):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Name: {user_info['Name']}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {user_info['Age']}", ln=True)
        pdf.cell(200, 10, txt=f"City: {user_info['City']}", ln=True)
        pdf.cell(200, 10, txt=f"Email: {user_info['Email']}", ln=True)
        pdf.cell(200, 10, txt=f"Symptoms: {', '.join(selected_symptoms)}", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True)
        pdf.cell(200, 10, txt=f"Doctor's Name: {doctor_info['Name']}", ln=True)
        pdf.cell(200, 10, txt=f"Doctor's Link: {doctor_info['Link']}", ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
            fig.savefig(temp_img.name, format='png')
            pdf.image(temp_img.name, x=10, y=None, w=190)

        pdf_output = io.BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin1'))
        pdf_output.seek(0)
        return pdf_output

    # Step 1: Collect User Data
    st.header("Step 1: Enter Your Information")
    user_info = {
        "Name": st.text_input("Name"),
        "Age": st.number_input("Age", min_value=1, max_value=120, step=1),
        "City": st.text_input("City"),
        "Email": st.text_input("Email")
    }

    if st.button("Next"):
        st.session_state.user_info = user_info
        st.session_state.step = 2

    # Step 2: Symptom Selection and Prediction
    if st.session_state.get("step", 1) == 2:
        st.header("Step 2: Select Your Symptoms")
        selected_symptoms = st.multiselect("Select Symptoms", options=symptom_list)

        if st.button("Predict"):
            input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
            prediction = classifier.predict([input_vector])[0]
            predicted_disease = labelencoder.inverse_transform([prediction])[0]
            confidence_scores = classifier.predict_proba([input_vector])[0]
            confidence = confidence_scores[prediction] * 100

            doctor_info = doctors.loc[doctors['disease'].str.lower() == predicted_disease.lower()].iloc[0]

            st.session_state.prediction = {
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "doctor_info": {
                    "Name": doctor_info["name"],
                    "Link": doctor_info["link"]
                },
                "selected_symptoms": selected_symptoms
            }
            st.session_state.step = 3

    # Step 3: Report Generation
    if st.session_state.get("step", 1) == 3:
        st.header("Step 3: Prediction Report")
        prediction = st.session_state.prediction

        st.write(f"**Predicted Disease**: {prediction['predicted_disease']}")
        st.write(f"**Confidence Level**: {prediction['confidence']:.2f}%")
        st.write(f"**Doctor's Name**: {prediction['doctor_info']['Name']}")
        st.write(f"[Doctor's Link]({prediction['doctor_info']['Link']})")

        # Generate Confidence Graph
        fig, ax = plt.subplots()
        ax.bar(["Confidence"], [prediction["confidence"]], color='blue')
        ax.set_ylim([0, 100])
        ax.set_ylabel("Confidence %")
        st.pyplot(fig)

        pdf_output = generate_pdf(
            user_info=st.session_state.user_info,
            predicted_disease=prediction['predicted_disease'],
            doctor_info=prediction['doctor_info'],
            selected_symptoms=prediction['selected_symptoms'],
            fig=fig
        )

        st.download_button(
            "Download PDF Report",
            data=pdf_output,
            file_name="Disease_Prediction_Report.pdf",
            mime="application/pdf"
        )
