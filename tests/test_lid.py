import pytest
import numpy as np
from models.mms_lid import MMSLidModel

# Note: We won't download the real model in a fast sanity test
# We just verify the class instantiates correctly if mocked or verify its method signatures.
def test_mms_lid_mocked(mocker):
    mocker.patch('models.mms_lid.Wav2Vec2ForSequenceClassification.from_pretrained')
    mocker.patch('models.mms_lid.Wav2Vec2FeatureExtractor.from_pretrained')
    
    # Just checking instantiation doesn't crash given mocks
    try:
        model = MMSLidModel(model_id='mock_model', device='cpu')
        assert model.device == 'cpu'
    except Exception as e:
        pytest.fail(f"MMSLidModel instantiation failed: {e}")
