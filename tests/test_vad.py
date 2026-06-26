import pytest
from models.vad import SileroVADModel

def test_vad_instantiation(mocker):
    # Mock torch.hub.load to prevent actual download during tests
    mocker.patch('torch.hub.load', return_value=(None, (None, None, None, None, None)))
    
    try:
        model = SileroVADModel(repo_or_dir='snakers4/silero-vad', model_name='silero_vad')
        assert model.utils is not None
    except Exception as e:
        pytest.fail(f"SileroVADModel instantiation failed: {e}")
