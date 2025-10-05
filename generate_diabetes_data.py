from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd
import numpy as np

# Create your sample data as before
real_data = pd.DataFrame({
    'patient_id': range(1, 21),
    'age': np.random.normal(55, 10, 20).round(0),
    'gender': np.random.choice(['Male', 'Female'], 20),
    'bmi': np.random.normal(28, 4, 20).round(1),
    'blood_glucose': np.random.normal(140, 30, 20).round(1),
    'hba1c': np.random.normal(7.0, 1.2, 20).round(1),
    'insulin_dosage': np.random.normal(30, 10, 20).round(1),
    'duration_diabetes_years': np.random.normal(8, 5, 20).round(0),
    'physical_activity': np.random.choice(['Low', 'Moderate', 'High'], 20),
    'medication_adherence': np.random.choice(['Poor', 'Fair', 'Good'], 20),
    'cholesterol': np.random.normal(200, 35, 20).round(1),
    'blood_pressure_systolic': np.random.normal(130, 15, 20).round(0),
    'blood_pressure_diastolic': np.random.normal(80, 10, 20).round(0),
    'smoking_status': np.random.choice(['Never', 'Former', 'Current'], 20),
    'alcohol_consumption': np.random.choice(['None', 'Occasional', 'Regular'], 20),
    'comorbidities': np.random.choice(['None', 'Hypertension', 'Cardiovascular', 'Kidney Disease'], 20)
})

def assign_control(row):
    control = False
    if (row['hba1c'] < 7.0 and row['blood_glucose'] < 140 and row['medication_adherence'] == 'Good'):
        control = True
    elif (row['hba1c'] < 7.5 and row['blood_glucose'] < 160 and row['medication_adherence'] in ['Good', 'Fair']):
        control = True
    if row['smoking_status'] == 'Current' or row['alcohol_consumption'] == 'Regular':
        control = False
    if row['comorbidities'] in ['Cardiovascular', 'Kidney Disease']:
        control = False
    return 'Controlled' if control else 'Uncontrolled'

real_data['diabetic_control'] = real_data.apply(assign_control, axis=1)

# Generate metadata from data automatically
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_data.drop(columns=['diabetic_control', 'patient_id']))

# Check and edit metadata JSON if needed (optional)
metadata_dict = metadata.to_dict()

# Uncomment next lines to print out metadata JSON and verify/edit categories
# import json
# print(json.dumps(metadata_dict, indent=4))

# Fit SDV model
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data.drop(columns=['diabetic_control', 'patient_id']))

# Generate synthetic data
synthetic_data = synthesizer.sample(100)
synthetic_data['diabetic_control'] = synthetic_data.apply(assign_control, axis=1)
synthetic_data['patient_id'] = range(1, len(synthetic_data) + 1)

synthetic_data.to_csv('synthetic_diabetes_data.csv', index=False)

print("Sample synthetic diabetic data:")
print(synthetic_data.head())
