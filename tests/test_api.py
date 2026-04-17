# filepath: tests/test_api.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock model loading before importing app
with patch('joblib.load') as mock_load, \
     patch('shap.TreeExplainer') as mock_explainer, \
     patch('builtins.open', create=True) as mock_open:
    
    # Setup mocks
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_load.return_value = mock_model
    mock_explainer.return_value = MagicMock()
    mock_open.return_value.__enter__.return_value.read.return_value = "0.5"
    
    # Mock numpy loads
    with patch('numpy.load') as mock_np:
        mock_np.return_value = np.zeros(30)
        
        from app.api.main import (
            Transaction, 
            BatchTransaction,
            FEATURE_NAMES,
            BEST_THRESHOLD
        )


class TestTransactionValidation:
    """Test Pydantic input validation"""
    
    def test_valid_transaction(self):
        tx = Transaction(features=[0.0] * 30)
        assert len(tx.features) == 30
    
    def test_transaction_wrong_length(self):
        with pytest.raises(ValueError):
            Transaction(features=[0.0] * 25)
    
    def test_transaction_with_nan(self):
        with pytest.raises(ValueError):
            Transaction(features=[0.0] * 29 + [float('nan')])
    
    def test_transaction_with_inf(self):
        with pytest.raises(ValueError):
            Transaction(features=[0.0] * 29 + [float('inf')])


class TestBatchTransactionValidation:
    """Test batch transaction validation"""
    
    def test_valid_batch(self):
        bt = BatchTransaction(transactions=[[0.0] * 30])
        assert len(bt.transactions) == 1
    
    def test_empty_batch(self):
        with pytest.raises(ValueError):
            BatchTransaction(transactions=[])
    
    def test_batch_too_large(self):
        with pytest.raises(ValueError):
            BatchTransaction(transactions=[[0.0] * 30] * 1001)


class TestFeatureNames:
    """Test feature name configuration"""
    
    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 30
    
    def test_feature_names_include_v1_to_v28(self):
        v_features = [f for f in FEATURE_NAMES if f.startswith('V')]
        assert len(v_features) == 28
    
    def test_feature_names_include_time_and_amount(self):
        assert "Time" in FEATURE_NAMES
        assert "Amount" in FEATURE_NAMES


class TestThreshold:
    """Test threshold configuration"""
    
    def test_threshold_is_float(self):
        assert isinstance(BEST_THRESHOLD, float)
    
    def test_threshold_in_valid_range(self):
        assert 0 < BEST_THRESHOLD <= 1