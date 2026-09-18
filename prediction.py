"""Inference with the user's unchanged model and scaler. Never trains estimators."""
from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = [
    'Age at enrollment', 'Admission grade', 'Previous qualification (grade)',
    'Tuition fees up to date', 'Debtor', 'Scholarship holder',
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
]
EXAMPLES = [
    [20, 150, 145, 'Yes', 'No', 'Yes', 6, 6, 15, 6, 6, 15],
    [23, 125, 125, 'Yes', 'No', 'No', 6, 4, 11, 6, 3, 10],
    [25, 110, 110, 'No', 'Yes', 'No', 6, 1, 8, 6, 0, 0],
]


def risk_category(probability):
    """Application-defined interpretation bands, not learned class boundaries."""
    if probability < 0.30:
        return 'LOW RISK'
    if probability < 0.60:
        return 'MEDIUM RISK'
    return 'HIGH RISK'


class Predictor:
    def __init__(self, directory=ROOT):
        directory = Path(directory)
        checksums = json.loads((directory / 'artifact_checksums.json').read_text())
        for name, expected in checksums.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Artifact integrity check failed: {name}')
        # Only load the trusted, supplied project artifacts; never accept uploaded pickles.
        self.model = joblib.load(directory / 'logistic_regression_model.pkl')
        self.scaler = joblib.load(directory / 'scaler.pkl')
        self.columns = list(joblib.load(directory / 'feature_columns.pkl'))
        self.defaults = joblib.load(directory / 'default_student.pkl').copy()
        self.preprocessing = json.loads((directory / 'input_preprocessing.json').read_text())
        if len(self.columns) != 36 or len(set(self.columns)) != 36:
            raise ValueError('Expected 36 unique training features.')
        if list(self.scaler.feature_names_in_) != self.columns:
            raise ValueError('Scaler feature order does not match feature_columns.pkl.')
        if self.model.n_features_in_ != 36 or self.scaler.n_features_in_ != 36:
            raise ValueError('Model/scaler feature count mismatch.')
        if list(self.model.classes_) != [0, 1]:
            raise ValueError('Expected classes [0, 1], with Dropout in probability column 1.')
        if self.preprocessing['feature_columns'] != self.columns:
            raise ValueError('Input conversion feature order mismatch.')
        if not isinstance(self.defaults, pd.Series) or not self.defaults.index.is_unique:
            raise ValueError('Expected a feature-indexed default Series.')
        if set(self.defaults.index) != set(self.columns) or not set(FIELDS) <= set(self.columns):
            raise ValueError('Missing or unexpected default features.')
        self.defaults = self.defaults.reindex(self.columns).astype(float)
        self.mean = pd.Series(self.preprocessing['mean']).reindex(self.columns)
        self.scale = pd.Series(self.preprocessing['scale']).reindex(self.columns)
        if not all(np.isfinite(s).all() for s in [self.defaults, self.mean, self.scale]) or (self.scale <= 0).any():
            raise ValueError('Non-finite defaults or invalid input conversion constants.')

    def prepare(self, values):
        if len(values) != len(FIELDS):
            raise ValueError('Provide all 12 student inputs.')
        cleaned = []
        for i, (field, value) in enumerate(zip(FIELDS, values)):
            if i in (3, 4, 5):
                if value not in ('Yes', 'No'):
                    raise ValueError(f'{field}: select Yes or No.')
                cleaned.append(1.0 if value == 'Yes' else 0.0)
                continue
            try:
                number = float(value)
            except (ValueError, TypeError):
                raise ValueError(f'{field}: enter a number.') from None
            if not math.isfinite(number):
                raise ValueError(f'{field}: enter a finite number.')
            low, high = (1, 120) if i == 0 else (0, 200) if i in (1, 2) else (0, 20) if i in (8, 11) else (0, 100)
            if not low <= number <= high:
                raise ValueError(f'{field}: enter a value from {low} to {high}.')
            if i in (0, 6, 7, 9, 10) and not number.is_integer():
                raise ValueError(f'{field}: enter a whole number.')
            cleaned.append(number)
        for enrolled, approved in [(6, 7), (9, 10)]:
            if cleaned[approved] > cleaned[enrolled]:
                raise ValueError(f'{FIELDS[approved]} cannot exceed enrolled units.')
        # Defaults already use the notebook CSV's standardized coordinate system.
        # Convert only the 12 raw UI values before replacing their default fields.
        student = self.defaults.copy(deep=True)
        for field, raw_value in zip(FIELDS, cleaned):
            student[field] = (raw_value - self.mean[field]) / self.scale[field]
        frame = pd.DataFrame([student], columns=self.columns)
        if frame.shape != (1, 36) or not np.isfinite(frame.to_numpy()).all():
            raise ValueError('Invalid complete student feature representation.')
        return frame

    def predict(self, values):
        frame = self.prepare(values)
        scaled = self.scaler.transform(frame)
        predicted = int(self.model.predict(scaled)[0])
        probability = float(self.model.predict_proba(scaled)[:, 1][0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Model returned an invalid probability.')
        return {
            'dropout_probability': probability,
            'predicted_class': 'Dropout' if predicted == 1 else 'Not Dropout',
            'risk_category': risk_category(probability),
        }
