import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from prediction import EXAMPLES, FIELDS, ROOT, Predictor, risk_category


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()

    def test_order_and_hidden_defaults(self):
        p = self.predictor
        frame = p.prepare(EXAMPLES[0])
        self.assertEqual(frame.shape, (1, 36))
        self.assertEqual(list(frame.columns), list(p.scaler.feature_names_in_))
        hidden = [name for name in p.columns if name not in FIELDS]
        np.testing.assert_array_equal(frame[hidden].iloc[0], p.defaults[hidden])
        self.assertTrue(np.isfinite(p.scaler.transform(frame)).all())

    def test_known_raw_age_conversion(self):
        # Independently recorded from the original raw/standardized dataset pair.
        # Age 20 has training-CSV coordinate -0.4422121140689147.
        frame = self.predictor.prepare(EXAMPLES[0])
        self.assertAlmostEqual(frame['Age at enrollment'].iloc[0], -0.4422121140689147, places=10)

    def test_actual_example_predictions(self):
        expected = [0.026284305429048405, 0.8071377860104834, 0.9999764399708152]
        for values, probability in zip(EXAMPLES, expected):
            with self.subTest(values=values):
                result = self.predictor.predict(values)
                self.assertAlmostEqual(result['dropout_probability'], probability, places=10)

    def test_class_comes_from_predict_independently_of_band(self):
        with patch.object(self.predictor.model, 'predict', return_value=np.array([1])), \
             patch.object(self.predictor.model, 'predict_proba', return_value=np.array([[0.45, 0.55]])):
            result = self.predictor.predict(EXAMPLES[0])
            self.assertEqual(result['predicted_class'], 'Dropout')
            self.assertEqual(result['risk_category'], 'MEDIUM RISK')

    def test_band_boundaries(self):
        for probability, band in [(0, 'LOW RISK'), (0.299999, 'LOW RISK'), (0.30, 'MEDIUM RISK'),
                                  (0.599999, 'MEDIUM RISK'), (0.60, 'HIGH RISK'), (1, 'HIGH RISK')]:
            self.assertEqual(risk_category(probability), band)

    def test_reject_invalid_inputs(self):
        for index, value in [(0, -1), (0, 20.5), (1, 201), (2, float('nan')), (3, 'Maybe'),
                             (6, -1), (7, 7), (8, 21), (9, float('inf')), (10, 7), (11, -1), (1, None)]:
            values = EXAMPLES[0].copy()
            values[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.predictor.predict(values)

    def test_predictions_do_not_mutate_artifacts_or_defaults(self):
        baseline = self.predictor.defaults.copy()
        with patch.object(self.predictor.model, 'fit', side_effect=AssertionError('No retraining')), \
             patch.object(self.predictor.scaler, 'fit', side_effect=AssertionError('No refitting')):
            for example in EXAMPLES:
                self.predictor.predict(example)
        np.testing.assert_array_equal(baseline, self.predictor.defaults)
        checksums = json.loads((ROOT / 'artifact_checksums.json').read_text())
        for name, digest in checksums.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
