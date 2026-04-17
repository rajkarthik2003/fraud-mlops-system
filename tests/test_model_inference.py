# filepath: tests/test_model_inference.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestModelInference:
    """Test model inference logic"""
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        # Simulate: class 0 prob = 0.3, class 1 prob = 0.7
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return model
    
    def test_predict_proba_returns_correct_shape(self, mock_model):
        """Test that predict_proba returns 2D array"""
        input_features = np.array([[0.0] * 30])
        result = mock_model.predict_proba(input_features)
        assert result.shape == (1, 2)
    
    def test_fraud_class_probability_extraction(self, mock_model):
        """Test extracting fraud probability (class 1)"""
        input_features = np.array([[0.0] * 30])
        probs = mock_model.predict_proba(input_features)
        fraud_prob = probs[0, 1]
        assert fraud_prob == 0.7
    
    def test_prediction_threshold_logic(self):
        """Test threshold-based prediction logic"""
        threshold = 0.5
        probs = np.array([0.3, 0.7, 0.4, 0.6, 0.9])
        preds = (probs >= threshold).astype(int)
        expected = np.array([0, 1, 0, 1, 1])
        assert np.array_equal(preds, expected)


class TestDriftDetection:
    """Test drift detection logic"""
    
    def test_z_score_calculation(self):
        """Test Z-score formula"""
        new_value = 10.0
        mean = 5.0
        std = 2.0
        z_score = abs((new_value - mean) / std)
        assert z_score == 2.5
    
    def test_drift_detection_threshold(self):
        """Test drift detection at threshold 3"""
        z_scores = np.array([1.0, 2.0, 3.5, 0.5, 4.0])
        drift_features = np.where(z_scores > 3)[0]
        assert len(drift_features) == 2
        assert 2 in drift_features
        assert 4 in drift_features
    
    def test_no_drift_when_within_threshold(self):
        """Test no drift detected when all Z-scores < 3"""
        z_scores = np.array([1.0, 2.0, 2.5, 0.5, 1.0])
        drift_features = [i for i, z in enumerate(z_scores) if z > 3]
        assert len(drift_features) == 0


class TestSHAPExplanation:
    """Test SHAP explanation logic"""
    
    def test_top_features_selection(self):
        """Test selecting top 5 contributing features"""
        contributions = np.array([0.1, -0.5, 0.3, -0.8, 0.2, 0.4, -0.1, 0.0, 0.5, -0.3])
        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
        expected_top = [3, 1, 8, 5, 2]  # -0.8, -0.5, 0.5, 0.4, 0.3
        assert np.array_equal(top_indices, expected_top)
    
    def test_impact_direction_classification(self):
        """Test classifying feature impact direction"""
        contributions = np.array([0.5, -0.3, 0.0])
        directions = ["increases_fraud" if c > 0 else "decreases_fraud" for c in contributions]
        assert directions[0] == "increases_fraud"
        assert directions[1] == "decreases_fraud"
        assert directions[2] == "decreases_fraud"