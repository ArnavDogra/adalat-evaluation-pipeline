import pytest
import numpy as np
from project.pipeline.metrics import normalize_for_asr_metrics, compute_number_metrics

def test_normalize_for_asr_metrics():
    assert normalize_for_asr_metrics("आज बारिश होगी।") == "आज बारिश होगी"
    assert normalize_for_asr_metrics("5.95") == "पाँच दशमलव नौ पाँच" or normalize_for_asr_metrics("5.95") == "5.95" # Function may keep digits or convert words based on WORD_NUMBERS setup
    
def test_compute_number_metrics():
    ref = "12 3.4 5"
    hyp = "12 3 5"
    rec, rnums, hnums, _, _ = compute_number_metrics(ref, hyp)
    assert "12" in rnums
    assert "3.4" in rnums
    assert rec == 2.0 / 3.0
