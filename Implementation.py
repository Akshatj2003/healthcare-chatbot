
# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Load datasets
# training_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\Training.csv')
# doc_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\doctors_dataset.csv', names=['Name', 'Description'])
# disease_info_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\DiseaseInfo.csv')  # Local dataset for disease details

# # Preprocessing: Extract features and labels
# X = training_dataset.iloc[:, :-1].values
# y = training_dataset.iloc[:, -1].values

# # Label Encoding for the target variable (disease names)
# labelencoder = LabelEncoder()
# y = labelencoder.fit_transform(y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# # Train Decision Tree Classifier
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(X_train, y_train)

# # List of symptoms from the dataset
# symptom_list = training_dataset.columns[:-1].str.lower().tolist()

# # Prepare doctor dataset
# diseases = pd.DataFrame(training_dataset['prognosis'].unique(), columns=['Disease'])
# doctors = pd.DataFrame({
#     'disease': diseases['Disease'], 
#     'name': doc_dataset['Name'], 
#     'link': doc_dataset['Description']
# })

# # Function to fetch disease details from the local dataset
# def get_disease_details_local(disease_name):
#     disease_data = disease_info_dataset[disease_info_dataset['Disease'].str.lower() == disease_name.lower()]
#     if not disease_data.empty:
#         description = disease_data['Description'].values[0]
#         causes = disease_data['Causes'].values[0]
#         treatment = disease_data['Treatment'].values[0]
#         return description, causes, treatment
#     else:
#         return "Description not available.", "Causes not available.", "Treatment options not available."

# # Function to generate chatbot response
# def chatbot_response(predicted_disease):
#     description, causes, treatment = get_disease_details_local(predicted_disease)
#     response = f"**Disease: {predicted_disease.capitalize()}**\n\n"
#     response += f"**Description:** {description}\n\n"
#     response += f"**Causes:** {causes}\n\n"
#     response += f"**Treatment:** {treatment}\n\n"
    
#     doctor_info = doctors[doctors['disease'].str.lower() == predicted_disease.lower()]
#     if not doctor_info.empty:
#         response += f"\n**Suggested Doctor:** {doctor_info['name'].values[0]}\n"
#         response += f"Visit: [Link]({doctor_info['link'].values[0]})"
#     else:
#         response += "\nNo doctor information available for this disease."
#     return response

# # Streamlit UI
# st.title("Disease Prediction Chatbot")
# st.header("Talk to the Chatbot")

# # Display initial message
# if "conversation" not in st.session_state:
#     st.session_state.conversation = ["Hello! I am your Disease Prediction chatbot. Please select your symptoms from the dropdown."]

# # Display chat history
# for message in st.session_state.conversation:
#     st.write(message)

# # Dropdown menu for symptom selection
# selected_symptoms = st.multiselect("Select symptoms from the list:", symptom_list)

# if st.button("Predict Disease"):
#     if selected_symptoms:
#         # Create binary input vector
#         input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
#         prediction = classifier.predict([input_vector])[0]
#         predicted_disease = labelencoder.inverse_transform([prediction])[0]
#         response = chatbot_response(predicted_disease)
#         st.session_state.conversation.append(f"Bot: {response}")
#     else:
#         st.session_state.conversation.append("Bot: Please select symptoms to proceed.")

# # Display updated conversation
# for message in st.session_state.conversation:
#     st.write(message)

# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from fpdf import FPDF

# # Load datasets
# training_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\Training.csv')
# doc_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\doctors_dataset.csv', names=['Name', 'Description'])
# disease_info_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\DiseaseInfo.csv')  # Local dataset for disease details

# # Preprocessing: Extract features and labels
# X = training_dataset.iloc[:, :-1].values
# y = training_dataset.iloc[:, -1].values

# # Label Encoding for the target variable (disease names)
# labelencoder = LabelEncoder()
# y = labelencoder.fit_transform(y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# # Train Decision Tree Classifier
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(X_train, y_train)

# # List of symptoms from the dataset
# symptom_list = training_dataset.columns[:-1].str.lower().tolist()

# # Prepare doctor dataset
# diseases = pd.DataFrame(training_dataset['prognosis'].unique(), columns=['Disease'])
# doctors = pd.DataFrame({
#     'disease': diseases['Disease'], 
#     'name': doc_dataset['Name'], 
#     'link': doc_dataset['Description']
# })

# # Function to fetch disease details locally
# def get_disease_details_local(disease_name):
#     disease_data = disease_info_dataset[disease_info_dataset['Disease'].str.lower() == disease_name.lower()]
#     if not disease_data.empty:
#         description = disease_data['Description'].values[0]
#         causes = disease_data['Causes'].values[0]
#         treatment = disease_data['Treatment'].values[0]
#         return description, causes, treatment
#     else:
#         return "Description not available.", "Causes not available.", "Treatment options not available."

# # PDF report generator function
# def generate_pdf(user_data, predicted_disease, disease_details):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font('Arial', 'B', 16)
#     pdf.cell(200, 10, 'Disease Prediction Report', ln=True, align='C')
#     pdf.ln(10)
    
#     pdf.set_font('Arial', '', 12)
#     for key, value in user_data.items():
#         pdf.cell(0, 10, f"{key}: {value}", ln=True)
    
#     pdf.ln(10)
#     pdf.set_font('Arial', 'B', 14)
#     pdf.cell(0, 10, 'Predicted Disease Details:', ln=True)
#     pdf.ln(5)
    
#     pdf.set_font('Arial', '', 12)
#     pdf.cell(0, 10, f"Disease: {predicted_disease}", ln=True)
#     pdf.ln(5)
#     pdf.multi_cell(0, 10, f"Description: {disease_details[0]}")
#     pdf.ln(5)
#     pdf.multi_cell(0, 10, f"Causes: {disease_details[1]}")
#     pdf.ln(5)
#     pdf.multi_cell(0, 10, f"Treatment: {disease_details[2]}")
    
#     pdf_output_path = 'disease_report.pdf'
#     pdf.output(pdf_output_path)
#     return pdf_output_path

# # Streamlit UI
# st.title("Disease Prediction System")

# # Page 1: Collect User Information
# if "page" not in st.session_state:
#     st.session_state.page = 1
# if st.session_state.page == 1:
#     st.header("Patient Information")
#     name = st.text_input("Name:")
#     age = st.number_input("Age:", min_value=1, max_value=120, step=1)
#     city = st.text_input("City:")
#     email = st.text_input("Email Address:")
    
#     if st.button("Next"):
#         if name and city and email:
#             st.session_state.user_data = {
#                 "Name": name,
#                 "Age": age,
#                 "City": city,
#                 "Email": email
#             }
#             st.session_state.page = 2
#         else:
#             st.warning("Please fill out all fields.")

# # Page 2: Symptom Selection and Disease Prediction
# elif st.session_state.page == 2:
#     st.header("Select Symptoms")
#     selected_symptoms = st.multiselect("Select symptoms from the list:", symptom_list)
    
#     if st.button("Predict Disease"):
#         if selected_symptoms:
#             input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
#             prediction = classifier.predict([input_vector])[0]
#             predicted_disease = labelencoder.inverse_transform([prediction])[0]
#             disease_details = get_disease_details_local(predicted_disease)
            
#             # Generate PDF report
#             pdf_path = generate_pdf(st.session_state.user_data, predicted_disease, disease_details)
#             st.success("Prediction complete! Download your detailed report below.")
#             st.download_button("Download Report", data=open(pdf_path, "rb").read(), file_name="Disease_Report.pdf")
#         else:
#             st.warning("Please select symptoms to proceed.")
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# import io

# # Load datasets
# training_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\Training.csv')
# doc_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\doctors_dataset.csv', names=['Name', 'Link'])

# # Preprocessing
# X = training_dataset.iloc[:, :-1].values
# y = training_dataset.iloc[:, -1].values

# labelencoder = LabelEncoder()
# y = labelencoder.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# # Train Decision Tree Classifier
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(X_train, y_train)

# # Symptom List
# symptom_list = training_dataset.columns[:-1].str.lower().tolist()

# # Prepare Doctor dataset
# diseases = pd.DataFrame(training_dataset['prognosis'].unique(), columns=['Disease'])
# doctors = pd.DataFrame({
#     'Disease': diseases['Disease'], 
#     'Name': doc_dataset['Name'], 
#     'Link': doc_dataset['Link']
# })

# # Collect User Information
# st.title("Disease Prediction System")

# # Page 1: User Information
# if "user_info_submitted" not in st.session_state:
#     st.session_state.user_info_submitted = False

# if not st.session_state.user_info_submitted:
#     with st.form("User Info"):
#         name = st.text_input("Name")
#         age = st.number_input("Age", min_value=1, max_value=120, step=1)
#         city = st.text_input("City")
#         email = st.text_input("Email")
#         submitted = st.form_submit_button("Submit")
        
#         if submitted:
#             st.session_state.user_info = {
#                 "Name": name,
#                 "Age": age,
#                 "City": city,
#                 "Email": email
#             }
#             st.session_state.user_info_submitted = True
#             st.experimental_rerun()

# # Page 2: Symptoms and Prediction
# else:
#     st.header("Symptom Selection")
#     selected_symptoms = st.multiselect("Select your symptoms:", symptom_list)

#     if st.button("Predict Disease"):
#         if selected_symptoms:
#             # Generate input vector for prediction
#             input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
#             prediction_probs = classifier.predict_proba([input_vector])[0]
#             prediction = np.argmax(prediction_probs)
#             predicted_disease = labelencoder.inverse_transform([prediction])[0]

#             # Fetch doctor details
#             doctor_info = doctors[doctors['Disease'] == predicted_disease].iloc[0]

#             st.subheader("Prediction Results")
#             st.write(f"**Predicted Disease:** {predicted_disease}")
#             st.write(f"**Suggested Doctor:** {doctor_info['Name']}")
#             st.write(f"[Doctor's Link]({doctor_info['Link']})")

#             # Display Confidence Level Graph
#             fig, ax = plt.subplots()
#             ax.bar(labelencoder.inverse_transform(np.argsort(prediction_probs)[::-1]), 
#                    np.sort(prediction_probs)[::-1])
#             ax.set_title("Confidence Levels for Predicted Diseases")
#             ax.set_ylabel("Confidence Score")
#             ax.set_xlabel("Disease")
#             st.pyplot(fig)

#             # Generate PDF Report
#             if st.button("Download PDF Report"):
#                 pdf = FPDF()
#                 pdf.add_page()
#                 pdf.set_font("Arial", size=12)

#                 # Add User Info
#                 pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
#                 pdf.cell(200, 10, txt=f"Name: {st.session_state.user_info['Name']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Age: {st.session_state.user_info['Age']}", ln=True)
#                 pdf.cell(200, 10, txt=f"City: {st.session_state.user_info['City']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Email: {st.session_state.user_info['Email']}", ln=True)

#                 # Add Prediction Info
#                 pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True)
#                 pdf.cell(200, 10, txt=f"Doctor's Name: {doctor_info['Name']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Doctor's Link: {doctor_info['Link']}", ln=True)

#                 # Add Confidence Graph
#                 pdf.cell(200, 10, txt="See Confidence Graph Below", ln=True)
#                 pdf.image(io.BytesIO(fig.canvas.tostring_rgb()), x=10, y=None, w=190)

#                 # Save and Serve PDF
#                 pdf_output = io.BytesIO()
#                 pdf.output(pdf_output)
#                 pdf_output.seek(0)

#                 st.download_button("Download Report", pdf_output, file_name="Disease_Report.pdf")

#     else:
#         st.info("Please select symptoms and click Predict Disease.")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# import io

# # Load datasets
# training_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\Training.csv')
# doc_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\doctors_dataset.csv', names=['Name', 'Link'])

# # Preprocessing
# X = training_dataset.iloc[:, :-1].values
# y = training_dataset.iloc[:, -1].values

# labelencoder = LabelEncoder()
# y = labelencoder.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# # Train Decision Tree Classifier
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(X_train, y_train)

# # Symptom List
# symptom_list = training_dataset.columns[:-1].str.lower().tolist()

# # Prepare Doctor dataset
# diseases = pd.DataFrame(training_dataset['prognosis'].unique(), columns=['Disease'])
# doctors = pd.DataFrame({
#     'Disease': diseases['Disease'], 
#     'Name': doc_dataset['Name'], 
#     'Link': doc_dataset['Link']
# })

# # Streamlit App Title
# st.title("Disease Prediction System")

# # Page 1: User Information
# if "user_info_submitted" not in st.session_state:
#     st.session_state.user_info_submitted = False

# if not st.session_state.user_info_submitted:
#     with st.form("User Info"):
#         name = st.text_input("Name")
#         age = st.number_input("Age", min_value=1, max_value=120, step=1)
#         city = st.text_input("City")
#         email = st.text_input("Email")
#         submitted = st.form_submit_button("Submit")
        
#         if submitted:
#             st.session_state.user_info = {
#                 "Name": name,
#                 "Age": age,
#                 "City": city,
#                 "Email": email
#             }
#             st.session_state.user_info_submitted = True
#             st.experimental_rerun()

# # Page 2: Symptoms and Prediction
# else:
#     st.header("Symptom Selection")
#     selected_symptoms = st.multiselect("Select your symptoms:", symptom_list)

#     if st.button("Predict Disease"):
#         if selected_symptoms:
#             # Generate input vector for prediction
#             input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
#             prediction_probs = classifier.predict_proba([input_vector])[0]
#             prediction = np.argmax(prediction_probs)
#             predicted_disease = labelencoder.inverse_transform([prediction])[0]

#             # Fetch doctor details
#             doctor_info = doctors[doctors['Disease'] == predicted_disease].iloc[0]

#             st.subheader("Prediction Results")
#             st.write(f"**Predicted Disease:** {predicted_disease}")
#             st.write(f"**Suggested Doctor:** {doctor_info['Name']}")
#             st.write(f"[Doctor's Link]({doctor_info['Link']})")

#             # Display Confidence Level Graph
#             fig, ax = plt.subplots()
#             ax.bar(labelencoder.inverse_transform(np.argsort(prediction_probs)[::-1]), 
#                    np.sort(prediction_probs)[::-1])
#             ax.set_title("Confidence Levels for Predicted Diseases")
#             ax.set_ylabel("Confidence Score")
#             ax.set_xlabel("Disease")
#             st.pyplot(fig)

#             # Generate PDF Report
#             def generate_pdf(user_info, predicted_disease, doctor_info, selected_symptoms, fig):
#                 pdf = FPDF()
#                 pdf.add_page()
#                 pdf.set_font("Arial", size=12)

#                 # Add Report Title
#                 pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')

#                 # Add User Info
#                 pdf.cell(200, 10, txt=f"Name: {user_info['Name']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Age: {user_info['Age']}", ln=True)
#                 pdf.cell(200, 10, txt=f"City: {user_info['City']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Email: {user_info['Email']}", ln=True)

#                 # Add Symptoms
#                 pdf.cell(200, 10, txt=f"Symptoms: {', '.join(selected_symptoms)}", ln=True)

#                 # Add Prediction Info
#                 pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True)
#                 pdf.cell(200, 10, txt=f"Doctor's Name: {doctor_info['Name']}", ln=True)
#                 pdf.cell(200, 10, txt=f"Doctor's Link: {doctor_info['Link']}", ln=True)

#                 # Add Confidence Graph
#                 pdf.cell(200, 10, txt="Confidence Graph:", ln=True)
#                 img_buffer = io.BytesIO()
#                 fig.savefig(img_buffer, format='PNG')
#                 img_buffer.seek(0)
#                 pdf.image(img_buffer, x=10, y=None, w=190)
#                 img_buffer.close()

#                 pdf_output = io.BytesIO()
#                 pdf.output(pdf_output)
#                 pdf_output.seek(0)
#                 return pdf_output

#             pdf_output = generate_pdf(
#                 st.session_state.user_info, 
#                 predicted_disease, 
#                 doctor_info, 
#                 selected_symptoms, 
#                 fig
#             )
#             st.download_button(
#                 label="Download Report",
#                 data=pdf_output,
#                 file_name="Disease_Report.pdf",
#                 mime="application/pdf"
#             )

#     else:
#         st.info("Please select symptoms and click Predict Disease.")
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from fpdf import FPDF
# import io
# import tempfile

# # Load datasets
# training_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\Training.csv')
# doc_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\doctors_dataset.csv', names=['Name', 'Description'])

# # Preprocess data
# X = training_dataset.iloc[:, :-1].values
# y = training_dataset.iloc[:, -1].values

# labelencoder = LabelEncoder()
# y = labelencoder.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# # Train Decision Tree Classifier
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(X_train, y_train)

# # Symptom list
# symptom_list = training_dataset.columns[:-1].str.lower().tolist()

# # Prepare doctor dataset
# diseases = pd.DataFrame(training_dataset['prognosis'].unique(), columns=['Disease'])
# doctors = pd.DataFrame({
#     'disease': diseases['Disease'],
#     'name': doc_dataset['Name'],
#     'link': doc_dataset['Description']
# })

# # Function to generate PDF
# def generate_pdf(user_info, predicted_disease, doctor_info, selected_symptoms, fig):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     pdf.cell(200, 10, txt="Disease Prediction Report", ln=True, align='C')
#     pdf.cell(200, 10, txt=f"Name: {user_info['Name']}", ln=True)
#     pdf.cell(200, 10, txt=f"Age: {user_info['Age']}", ln=True)
#     pdf.cell(200, 10, txt=f"City: {user_info['City']}", ln=True)
#     pdf.cell(200, 10, txt=f"Email: {user_info['Email']}", ln=True)
#     pdf.cell(200, 10, txt=f"Symptoms: {', '.join(selected_symptoms)}", ln=True)
#     pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True)
#     pdf.cell(200, 10, txt=f"Doctor's Name: {doctor_info['Name']}", ln=True)
#     pdf.cell(200, 10, txt=f"Doctor's Link: {doctor_info['Link']}", ln=True)

#     with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
#         fig.savefig(temp_img.name, format='png')
#         pdf.cell(200, 10, txt="Confidence Graph:", ln=True)
#         pdf.image(temp_img.name, x=10, y=None, w=190)

#     pdf_output = io.BytesIO()
#     pdf.output(pdf_output)
#     pdf_output.seek(0)
#     return pdf_output

# # Streamlit UI
# st.title("Disease Prediction System")

# # Step 1: Collect User Data
# st.header("Step 1: Enter Your Information")
# user_info = {
#     "Name": st.text_input("Name"),
#     "Age": st.number_input("Age", min_value=1, max_value=120, step=1),
#     "City": st.text_input("City"),
#     "Email": st.text_input("Email")
# }

# if st.button("Next"):
#     st.session_state.user_info = user_info
#     st.session_state.step = 2

# # Step 2: Symptom Selection and Prediction
# if st.session_state.get("step", 1) == 2:
#     st.header("Step 2: Select Your Symptoms")
#     selected_symptoms = st.multiselect("Select Symptoms", options=symptom_list)

#     if st.button("Predict"):
#         input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
#         prediction = classifier.predict([input_vector])[0]
#         predicted_disease = labelencoder.inverse_transform([prediction])[0]
#         confidence_scores = classifier.predict_proba([input_vector])[0]
#         confidence = confidence_scores[prediction] * 100

#         doctor_info = doctors.loc[doctors['disease'].str.lower() == predicted_disease.lower()].iloc[0]

#         st.session_state.prediction = {
#             "predicted_disease": predicted_disease,
#             "confidence": confidence,
#             "doctor_info": {
#                 "Name": doctor_info["name"],
#                 "Link": doctor_info["link"]
#             },
#             "selected_symptoms": selected_symptoms
#         }
#         st.session_state.step = 3

# # Step 3: Report Generation
# if st.session_state.get("step", 1) == 3:
#     st.header("Step 3: Prediction Report")
#     prediction = st.session_state.prediction

#     st.write(f"**Predicted Disease**: {prediction['predicted_disease']}")
#     st.write(f"**Confidence Level**: {prediction['confidence']:.2f}%")
#     st.write(f"**Doctor's Name**: {prediction['doctor_info']['Name']}")
#     st.write(f"[Doctor's Link]({prediction['doctor_info']['Link']})")

#     # Generate Confidence Graph
#     fig, ax = plt.subplots()
#     ax.bar(["Confidence"], [prediction["confidence"]], color='blue')
#     ax.set_ylim([0, 100])
#     ax.set_ylabel("Confidence %")
#     st.pyplot(fig)

#     pdf_output = generate_pdf(
#         user_info=st.session_state.user_info,
#         predicted_disease=prediction['predicted_disease'],
#         doctor_info=prediction['doctor_info'],
#         selected_symptoms=prediction['selected_symptoms'],
#         fig=fig
#     )

#     st.download_button(
#         "Download PDF Report",
#         data=pdf_output,
#         file_name="Disease_Prediction_Report.pdf",
#         mime="application/pdf"
#     )
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

# Load datasets
training_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\Training.csv')
doc_dataset = pd.read_csv(r'C:\Users\text0\OneDrive\Documents\Downloads\doctors_dataset.csv', names=['Name', 'Description'])

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

# Streamlit UI
st.title("Disease Prediction System")

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
