import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Model and prediction imports
from model_handler import (
    VirusPredictor, get_virus_predictor, refresh_virus_mappings,
    VIRUS_MAPPING, OTHER_VIRUS_MAPPING, COMBINED_VIRUS_MAPPING, ALL_SYMPTOMS
)

refresh_virus_mappings()

# Symptom display mapping (no-space keys -> user-friendly display names)
SYMPTOM_DISPLAY_NAMES = {
    'HEADACHE': 'Headache',
    'IRRITABILITY': 'Irritability', 
    'ALTEREDSENSORIUM': 'Altered Sensorium',
    'SOMNOLENCE': 'Somnolence',
    'NECKRIGIDITY': 'Neck Rigidity',
    'SEIZURES': 'Seizures',
    'DIARRHEA': 'Diarrhea',
    'DYSENTERY': 'Dysentery',
    'NAUSEA': 'Nausea',
    'VOMITING': 'Vomiting',
    'ABDOMINALPAIN': 'Abdominal Pain',
    'MALAISE': 'Malaise',
    'MYALGIA': 'Myalgia',
    'ARTHRALGIA': 'Arthralgia',
    'CHILLS': 'Chills',
    'RIGORS': 'Rigors',
    'FEVER': 'Fever',
    'BREATHLESSNESS': 'Breathlessness',
    'COUGH': 'Cough',
    'RHINORRHEA': 'Rhinorrhea',
    'SORETHROAT': 'Sore Throat',
    'BULLAE': 'Bullae',
    'PAPULARRASH': 'Papular Rash',
    'PUSTULARRASH': 'Pustular Rash',
    'MUSCULARRASH': 'Muscular Rash',
    'MACULOPAPULARRASH': 'Maculopapular Rash',
    'ESCHAR': 'Eschar',
    'DARKURINE': 'Dark Urine',
    'HEPATOMEGALY': 'Hepatomegaly',
    'JAUNDICE': 'Jaundice',
    'REDEYE': 'Red Eye',
    'DISCHARGEEYES': 'Discharge Eyes',
    'CRUSHINGEYES': 'Crushing Eyes',
    'SWELLINGEYES': 'Swelling Eyes',
    'RETROORBITALPAIN': 'Retro-orbital Pain',
}

# Database imports (minimal addition)
from data_handler import save_prediction_to_db, get_db_health, get_prediction_stats, save_validation_to_db


# Page configuration - with error handling for deployment consistency
try:
    st.set_page_config(
        page_title="Virus Detection System",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as config_error:
    # Fallback for deployment issues
    st.set_page_config(
        page_title="Virus Detection System",
        layout="wide"
    )

@st.cache_data
def load_mappings():
    """Load state, district, and district-state mapping CSV files"""
    try:
        state_map = pd.read_csv('state_encoding_map.csv')
        district_map = pd.read_csv('district_encoding_map.csv')
        district_state_map = pd.read_csv('district_state_mapping.csv')
        return state_map, district_map, district_state_map
    except Exception as e:
        st.error(f"Error loading mapping files: {e}")
        return None, None, None

def main():
    # Top logos
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        try:
            st.image("logo_1.jpeg", width=300)
        except:
            st.write("")  # Skip if image not found
    with col3:
        try:
            st.image("Amity_logo2.png", width=250)
        except:
            st.write("")  # Skip if image not found
    with col2:
        try:
            st.image("logo_2.jpeg", width=250)
        except:
            st.write("")  # Skip if image not found
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Prediction", "About"])

    if page == "Home":
        st.markdown(
            "<h1 style='text-align: center;'>🦠 Virus Detection and Classification System</h1>",
            unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align: center;'>Advanced AI-Powered Diagnostic Tool for Viral Infections</h2>",
            unsafe_allow_html=True)
        st.write("""
        Welcome to the Virus Detection and Classification System!
        
        This advanced AI-driven system assists healthcare professionals by analyzing patient symptoms 
        and demographic information to predict the most probable viral infection from a comprehensive 
        database of 26+ virus categories.
        
        **Key Features:**
        - **Dual-Model Architecture**: Primary classification for major virus categories and secondary classification for "Other Viruses"
        - **Comprehensive Symptom Analysis**: Covers neurological, gastrointestinal, respiratory, dermatological, and systemic symptoms
        - **Geo-temporal Intelligence**: Incorporates seasonal patterns and geographical factors
        - **Real-time Predictions**: Instant probability scores and confidence metrics
        
        Navigate to the **Prediction** page using the sidebar to input patient details 
        and get comprehensive virus classification results.
        """)
        st.warning("**Medical Disclaimer**: This system is designed to assist healthcare professionals and should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult qualified medical personnel for patient care decisions.")

    elif page == "About":
        st.title("About Virus Detection System")
        st.write("""
        ### System Overview
        This application utilizes advanced machine learning techniques to analyze patient symptoms 
        and predict viral infections with high accuracy.
        
        ### Technical Specifications
        - **Primary Model**: Custom Gated Residual Tabular Transformer
        - **Secondary Model**: Custom Gated Residual Tabular Transformer classifier for "Other Viruses" subcategorization  
        - **Feature Engineering**: 80+ engineered features including temporal, geographical, and symptom interaction variables
        - **Optimization**: Cached models and pre-computed lookup tables for real-time performance
        
        ### Supported Virus Categories
        The system can identify and classify the following major virus categories:
        - Dengue Virus, Chikungunya Virus, Japanese Encephalitis
        - Hepatitis A/B/C/E Viruses
        - Influenza variants (H1N1, H3N2, Victoria)
        - Respiratory viruses (RSV, Adenovirus, SARS-CoV-2)
        - And many more...
        
        ### Data Sources
        The models are trained on comprehensive clinical datasets with proper encoding 
        for states, districts, and symptom combinations to ensure accurate predictions 
        across different geographical regions.
        """)
        st.warning("**Medical Disclaimer**: This system provides diagnostic assistance and should not replace professional medical evaluation and treatment decisions.")

    elif page == "Prediction":
        st.title("🦠 Virus Detection and Classification System")
        st.markdown("---")
        st.write("Enter patient information and clinical symptoms to predict the most likely virus.")

        # Initialize virus predictor (cached for performance)
        try:
            predictor = get_virus_predictor()
            if predictor.model1 is None or predictor.model2 is None:
                st.error("Failed to load models. Please ensure the .pth model files are in the 'models/' directory.")
                st.info("Expected files: `models/streamlit_virus_model_Major.pth` and `models/streamlit_virus_model_other.pth`")
                return
        except Exception as e:
            st.error(f"Error initializing predictor: {e}")
            return

        state_map, district_map, district_state_map = load_mappings()
        if state_map is None or district_map is None or district_state_map is None:
            st.error("Failed to load mapping files. Please check the CSV files.")
            return

        # Sidebar for patient demographics
        st.sidebar.header("Patient Information")

        patient_data = {}

        # Demographics (MATCH EXACT TRAINING COLUMN NAMES)
        patient_data['age'] = st.sidebar.number_input("Age(If age is less than 1, enter 0)", min_value=0.0, max_value=120.0, value=30.0, step=0.1)
        patient_data['SEX'] = st.sidebar.selectbox("Sex", options=[0, 1], 
                                                    format_func=lambda x: "Female" if x == 0 else "Male", index=1)
        patient_data['PATIENTTYPE'] = st.sidebar.selectbox("Patient Type", options=[0, 1], 
                                                            format_func=lambda x: "Outpatient" if x == 0 else "Inpatient", index=1)
        patient_data['durationofillness'] = st.sidebar.number_input("Duration of Illness (days)", 
                                                                     min_value=0, max_value=365, value=3)

        # State selection with names
        state_names = state_map['state_name'].tolist()
        # Set Tamil Nadu as default if available, otherwise use first state
        default_state_index = 0
        if 'Tamil Nadu' in state_names:
            default_state_index = state_names.index('Tamil Nadu')
        selected_state_name = st.sidebar.selectbox("State", options=state_names, index=default_state_index)
        patient_data['labstate'] = int(state_map[state_map['state_name'] == selected_state_name]['encoded_value'].values[0])

        # District selection filtered by state
        filtered_districts = district_state_map[district_state_map['state'] == selected_state_name]
        district_names = filtered_districts['district_name'].tolist()

        if len(district_names) > 0:
            selected_district_name = st.sidebar.selectbox("District", options=district_names, index=0)
            patient_data['districtencoded'] = int(filtered_districts[filtered_districts['district_name'] == selected_district_name]['district_encoded'].values[0])
        else:
            st.sidebar.warning("No districts available for selected state")
            patient_data['districtencoded'] = 0

        # Temporal features
        current_month = datetime.now().month
        patient_data['month'] = st.sidebar.selectbox("Month of Illness", options=list(range(1, 13)), 
                                                      index=current_month - 1,  # index is 0-based
                                                      format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
        patient_data['year'] = st.sidebar.number_input("Year", min_value=2012, max_value=2026, value=datetime.now().year)

        # Syndrome Selection
        st.header("Syndrome Classification")
        st.write("Select the primary syndrome that best describes the patient's condition:")
        
        # Syndrome options mapping
        SYNDROME_OPTIONS = {
            1: "Acute Diarrheal Disease",
            2: "Dysentery", 
            3: "Acute Flaccid Paralysis",
            4: "Acute Hepatitis",
            5: "ARI/Influenza Like Illness (ILI)",
            6: "Fever with Altered sensorium",
            7: "Severe Acute Respiratory Infection (SARI)",
            8: "Cough <= 2 weeks without fever",
            9: "Cough <=2 weeks with fever",
            10: "Cough > 2 weeks with fever",
            11: "Acute Encephalitis Syndrome (AES)",
            12: "Conjunctivitis",
            13: "Fever with Bleeding",
            14: "Fever with Rash",
            15: "Hemorrhagic fever",
            16: "Jaundice of < 4 weeks",
            17: "Only Fever < 7 days",
            18: "Fever >= 7 days",
            19: "Other",
        }
        
        # Syndrome selection dropdown
        syndrome_options = list(SYNDROME_OPTIONS.keys())
        selected_syndrome = st.selectbox(
            "Primary Syndrome",
            options=syndrome_options,
            format_func=lambda x: SYNDROME_OPTIONS[x],
            help="Select the syndrome that best matches the clinical presentation"
        )
        
        patient_data['syndrome'] = selected_syndrome
        patient_data['syndrome_name'] = SYNDROME_OPTIONS[selected_syndrome]
        
        # If "Other" is selected, provide text input for specification
        if selected_syndrome == 19:
            other_syndrome = st.text_input(
                "Please specify the syndrome:",
                placeholder="Enter the specific syndrome...",
                help="Describe the specific syndrome not listed above"
            )
            patient_data['other_syndrome_specification'] = other_syndrome
        else:
            patient_data['other_syndrome_specification'] = ""
        
        st.markdown("---")

        # Main area for symptoms
        st.header("Clinical Symptoms")
        st.write("Select all symptoms present in the patient:")

        # Display all symptoms in a simple grid layout
        cols = st.columns(4)  # 4 columns for better space utilization
        for idx, symptom in enumerate(ALL_SYMPTOMS):
            with cols[idx % 4]:
                display_name = SYMPTOM_DISPLAY_NAMES.get(symptom, symptom.replace('_', ' ').title())
                patient_data[symptom] = 1 if st.checkbox(display_name, key=symptom) else 0

        st.markdown("---")

        # Prediction button
        if st.button("Predict Virus", type="primary", use_container_width=True):
            # Check if at least one symptom is selected
            symptoms_selected = any(patient_data.get(symptom, 0) == 1 for symptom in ALL_SYMPTOMS)
            
            if not symptoms_selected:
                st.warning("Please select at least one symptom before making a prediction.")
                st.info("Expand the symptom groups above and check the boxes for symptoms present in the patient.")
            else:
                with st.spinner("Analyzing patient data..."):
                    try:
                        # Make prediction using the predictor
                        prediction_results = predictor.predict(patient_data)
                        
                        y_pred = prediction_results['y_pred']
                        y_pred_proba = prediction_results['y_pred_proba']
                        top_5_indices = prediction_results['top_5_indices']
                        second_model_results = prediction_results['second_model_results']

                        # Save prediction results to session state for validation form
                        st.session_state['prediction_results'] = {
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'top_5_indices': top_5_indices,
                            'second_model_results': second_model_results,
                            'patient_data': patient_data.copy(),
                            'selected_state_name': selected_state_name,
                            'selected_district_name': selected_district_name
                        }

                        # Display results
                        st.success("Prediction Complete!")
                        
                        # Save prediction to database (minimal intrusion)
                        try:
                            # Prepare prediction result for database
                            prediction_result = {
                                'predicted_virus': VIRUS_MAPPING[y_pred],
                                'predicted_virus_id': int(y_pred),
                                'confidence': float(y_pred_proba[y_pred] * 100),
                                'top_5_predictions': [
                                    {
                                        'virus': VIRUS_MAPPING[idx],
                                        'virus_id': int(idx),
                                        'confidence': float(y_pred_proba[idx] * 100)
                                    } for idx in top_5_indices
                                ]
                            }
                            
                            # Add second model results if available
                            if second_model_results:
                                prediction_result['sub_classification'] = {
                                    'predicted_sub_virus': OTHER_VIRUS_MAPPING[second_model_results['prediction']],
                                    'predicted_sub_virus_id': int(second_model_results['prediction']),
                                    'sub_confidence': float(second_model_results['probabilities'][second_model_results['prediction']] * 100),
                                    'top_5_sub_predictions': [
                                        {
                                            'virus': OTHER_VIRUS_MAPPING[idx],
                                            'virus_id': int(idx),
                                            'confidence': float(second_model_results['probabilities'][idx] * 100)
                                        } for idx in second_model_results['top_5']
                                    ]
                                }
                            
                            # Save to database (non-blocking)
                            saved_id = save_prediction_to_db(
                                patient_data=patient_data,
                                prediction_result=prediction_result,
                                model_info={'model1': 'XGB_M1_16JAN', 'model2': 'xgb_filtered_model2'},
                                state_name=selected_state_name,
                                district_name=selected_district_name
                            )
                            
                            # Store saved_id in session state for validation
                            st.session_state['saved_id'] = saved_id
                                
                        except Exception as db_error:
                            # Don't let database errors break the prediction display
                            st.sidebar.warning("⚠️ Database save failed")
                            st.sidebar.caption(f"Error: {str(db_error)[:50]}...")

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.subheader("Most Likely Virus")

                            # Check if primary prediction is Other_Viruses
                            if y_pred == 15 and second_model_results:
                                sub_virus = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                                sub_confidence = second_model_results['probabilities'][second_model_results['prediction']] * 100
                                st.metric(
                                    label="Predicted Virus",
                                    value=f"Other_Viruses → {sub_virus}",
                                    delta=f"{y_pred_proba[y_pred]*100:.2f}% (M1) | {sub_confidence:.2f}% (M2)"
                                )
                            else:
                                st.metric(
                                    label="Predicted Virus",
                                    value=VIRUS_MAPPING[y_pred],
                                    delta=f"{y_pred_proba[y_pred]*100:.2f}% confidence"
                                )

                        with col2:
                            st.subheader("Top 5 Predictions")
                            for rank, idx in enumerate(top_5_indices, 1):
                                virus_name = VIRUS_MAPPING[idx]
                                confidence = y_pred_proba[idx] * 100

                                # Add indicator if this is Other_Viruses
                                if idx == 15 and second_model_results:
                                    sub_virus = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                                    st.write(f"{rank}. **{virus_name}** → *{sub_virus}*: {confidence:.2f}%")
                                else:
                                    st.write(f"{rank}. **{virus_name}**: {confidence:.2f}%")

                        # Display second model results if available
                        if second_model_results:
                            st.markdown("---")
                            st.subheader("Other Viruses Sub-Classification")
                            # st.info("Since 'Other_Viruses' appeared in top 5, secondary classification was performed.")

                            col3, col4 = st.columns([1, 1])

                            with col3:
                                st.write("**Top Prediction:**")
                                top_sub = OTHER_VIRUS_MAPPING[second_model_results['prediction']]
                                top_conf = second_model_results['probabilities'][second_model_results['prediction']] * 100
                                st.metric(label="Sub-Category", value=top_sub, delta=f"{top_conf:.2f}% confidence")

                            with col4:
                                st.write("**Top 5 Sub-Categories:**")
                                for rank, idx in enumerate(second_model_results['top_5'], 1):
                                    sub_virus = OTHER_VIRUS_MAPPING[idx]
                                    sub_confidence = second_model_results['probabilities'][idx] * 100
                                    st.write(f"{rank}. **{sub_virus}**: {sub_confidence:.2f}%")

                        # Display probability distribution
                        st.markdown("---")
                        st.subheader("Probability Distribution")

                        if second_model_results:
                            tab1, tab2 = st.tabs(["Model 1 (Major Classes)", "Model 2 (Other Viruses)"])
                        else:
                            tabs = st.tabs(["Model 1 (Major Classes)"])
                            tab1 = tabs[0]

                        with tab1:
                            st.write("**Top 10 Major Virus Categories**")
                            top_10_indices = np.argsort(y_pred_proba)[-10:][::-1]
                            prob_df = pd.DataFrame({
                                'Virus': [VIRUS_MAPPING[i] for i in top_10_indices],
                                'Probability (%)': [y_pred_proba[i]*100 for i in top_10_indices]
                            })
                            st.bar_chart(prob_df.set_index('Virus'))

                        if second_model_results:
                            with tab2:
                                st.write("**Top 10 Other Virus Sub-Categories**")
                                top_10_indices_m2 = np.argsort(second_model_results['probabilities'])[-10:][::-1]
                                prob_df_m2 = pd.DataFrame({
                                    'Virus': [OTHER_VIRUS_MAPPING[i] for i in top_10_indices_m2],
                                    'Probability (%)': [second_model_results['probabilities'][i]*100 for i in top_10_indices_m2]
                                })
                                st.bar_chart(prob_df_m2.set_index('Virus'))

                        # Feature summary
                        with st.expander("Input Summary"):
                            st.write("**Patient Demographics:**")
                            st.write(f"- Age: {patient_data['age']} years")
                            st.write(f"- Sex: {'Male' if patient_data['SEX'] == 1 else 'Female'}")
                            st.write(f"- Patient Type: {'Inpatient' if patient_data['PATIENTTYPE'] == 1 else 'Outpatient'}")
                            st.write(f"- Duration: {patient_data['durationofillness']} days")

                            active_symptoms = [k.replace('_', ' ').title() for k, v in patient_data.items() 
                                             if k in ALL_SYMPTOMS and v == 1]
                            st.write(f"\n**Active Symptoms ({len(active_symptoms)}):**")
                            if active_symptoms:
                                st.write(", ".join(active_symptoms))
                            else:
                                st.write("None reported")

                        st.warning("**Medical Disclaimer**: This prediction is generated by AI and should be used only as a diagnostic aid. Always consult with qualified healthcare professionals for proper medical diagnosis and treatment decisions.")

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        import traceback
                        st.error(traceback.format_exc())

        # Validation Section - Show if prediction results exist in session state
        if 'prediction_results' in st.session_state and 'saved_id' in st.session_state and st.session_state['saved_id']:
            st.markdown("---")
            st.subheader("🩺 Medical Validation (Optional)")
            st.info("Help us improve the AI model by providing the actual diagnosis. This data will be used for model validation and improvement.")
            
            # Get prediction results from session state
            pred_results = st.session_state['prediction_results']
            saved_id = st.session_state['saved_id']
            
            # Create validation form
            with st.form(key=f"validation_form_{saved_id}"):
                col_val1, col_val2 = st.columns([2, 1])
                
                with col_val1:
                    # Combined dropdown with all virus options
                    virus_options = list(COMBINED_VIRUS_MAPPING.keys())
                    
                    selected_virus_key = st.selectbox(
                        "Actual Virus Diagnosis",
                        options=[None] + virus_options,
                        format_func=lambda x: "Please select the actual diagnosis..." if x is None else COMBINED_VIRUS_MAPPING[x],
                        help="Select the confirmed virus diagnosis from laboratory results or clinical assessment"
                    )
                
                with col_val2:
                    validation_confidence = st.selectbox(
                        "Confidence Level",
                        options=["High", "Medium", "Low"],
                        help="How confident are you in this diagnosis?"
                    )
                
                # Notes field
                validation_notes = st.text_area(
                    "Additional Notes (Optional)",
                    help="Any additional clinical observations or context",
                    placeholder="e.g., confirmed by RT-PCR, clinical presentation consistent with..."
                )
                
                # Submit validation button
                validation_submitted = st.form_submit_button(
                    "Submit Validation",
                    type="secondary",
                    use_container_width=True
                )
                
                if validation_submitted and selected_virus_key is not None:
                    # Save validation to database using session state data
                    validation_data = {
                        'prediction_id': saved_id,
                        'actual_virus_key': selected_virus_key,
                        'actual_virus_name': COMBINED_VIRUS_MAPPING[selected_virus_key],
                        'confidence_level': validation_confidence,
                        'notes': validation_notes,
                        'predicted_virus': VIRUS_MAPPING[pred_results['y_pred']],
                        'prediction_confidence': float(pred_results['y_pred_proba'][pred_results['y_pred']] * 100),
                        'patient_summary': {
                            'age': pred_results['patient_data']['age'],
                            'sex': 'Male' if pred_results['patient_data']['SEX'] == 1 else 'Female',
                            'state': pred_results['selected_state_name'],
                            'district': pred_results['selected_district_name']
                        }
                    }
                    
                    try:
                        validation_id = save_validation_to_db(validation_data)
                        if validation_id:
                            st.success("✅ Thank you! Validation submitted successfully.")
                            st.balloons()
                            st.caption(f"Validation ID: {validation_id[:8]}...")
                            # Clear the session state after successful validation
                            if 'prediction_results' in st.session_state:
                                del st.session_state['prediction_results']
                            if 'saved_id' in st.session_state:
                                del st.session_state['saved_id']
                        else:
                            st.error("❌ Failed to save validation. Please try again.")
                    except Exception as val_error:
                        st.error(f"❌ Validation error: {str(val_error)}")
                        # Add more detailed error info
                        st.error(f"Full error: {val_error}")
                
                elif validation_submitted and selected_virus_key is None:
                    st.warning("⚠️ Please select an actual diagnosis before submitting.")


if __name__ == "__main__":
    main()
