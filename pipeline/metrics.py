import re
import unicodedata
import jiwer
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import pandas as pd
import numpy as np
import time
from config import WORD_NUMBERS

wer_transform = jiwer.Compose([
    jiwer.ToLowerCase(), 
    jiwer.RemoveMultipleSpaces(), 
    jiwer.RemovePunctuation(), 
    jiwer.Strip()
])

def normalize_for_asr_metrics(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    trans = str.maketrans('०१२३४५६७८९', '0123456789')
    text = text.translate(trans)
    
    for num_str in ['जीरो', 'ज़ीरो', '0']: text = text.replace('डबल ' + num_str, '00').replace('डबल' + num_str, '00')
    for num_str in ['वन', '1']: text = text.replace('डबल ' + num_str, '11').replace('डबल' + num_str, '11')
    for num_str in ['टू', '2']: text = text.replace('डबल ' + num_str, '22').replace('डबल' + num_str, '22')
    for num_str in ['थ्री', '3']: text = text.replace('डबल ' + num_str, '33').replace('डबल' + num_str, '33')
    for num_str in ['फोर', '4']: text = text.replace('डबल ' + num_str, '44').replace('डबल' + num_str, '44')
    for num_str in ['फाइव', 'फाईव', '5']: text = text.replace('डबल ' + num_str, '55').replace('डबल' + num_str, '55')
    for num_str in ['सिक्स', '6']: text = text.replace('डबल ' + num_str, '66').replace('डबल' + num_str, '66')
    for num_str in ['सेवन', 'सबन', '7']: text = text.replace('डबल ' + num_str, '77').replace('डबल' + num_str, '77')
    for num_str in ['एट', '8']: text = text.replace('डबल ' + num_str, '88').replace('डबल' + num_str, '88')
    for num_str in ['नाइन', '9']: text = text.replace('डबल ' + num_str, '99').replace('डबल' + num_str, '99')
    
    text = text.replace('जीरोफोन', '0 4')
    text = text.replace('*', ' star ').replace('#', ' hash ').replace('%', ' percent ').replace('@', ' at ')
    
    words = text.split()
    for i, w in enumerate(words):
        clean_w = re.sub(r'^[\W_]+|[\W_]+$', '', w)
        if clean_w in WORD_NUMBERS:
            words[i] = w.replace(clean_w, WORD_NUMBERS[clean_w])
    text = " ".join(words)
    
    text = re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text)
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = re.sub(r'(?<=\d)\.(?=\d)', '<DECIMAL>', text)
    
    punct_pattern = r"[,।\.\?!:;\"'\(\)\[\]\{\}\-—_/\\]"
    text = re.sub(punct_pattern, ' ', text)
    text = text.replace('<DECIMAL>', '.')
    
    text = text.replace('हैँ', 'हैं').replace('मेँ', 'में').replace('नहीँ', 'नहीं')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text):
    return re.findall(r'\b\d+(?:\.\d+)?\b', text)

def compute_number_metrics(ref_text, hyp_text):
    ref_nums = extract_numbers(ref_text)
    hyp_nums = extract_numbers(hyp_text)
    if not ref_nums:
        return 1.0, "", ", ".join(hyp_nums), "", ", ".join(hyp_nums)
    ref_nums_copy = list(ref_nums)
    hyp_nums_copy = list(hyp_nums)
    matched = []
    for rn in ref_nums_copy[:]:
        if rn in hyp_nums_copy:
            matched.append(rn)
            hyp_nums_copy.remove(rn)
            ref_nums_copy.remove(rn)
    recall = len(matched) / len(ref_nums)
    return recall, ", ".join(ref_nums), ", ".join(hyp_nums), ", ".join(ref_nums_copy), ", ".join(hyp_nums_copy)

def extract_critical_terms(text):
    return set([w for w in str(text).split() if len(w) > 3])
    
def compute_critical_term_metrics(ref, hyp):
    ref_kw = extract_critical_terms(ref)
    hyp_kw = extract_critical_terms(hyp)
    if len(ref_kw) == 0: return 1.0, "", ""
    matched_terms = []
    missed_terms = []
    for rw in ref_kw:
        is_detected = False
        for hw in hyp_kw:
            if Levenshtein.distance(rw, hw) / len(rw) <= 0.30:
                is_detected = True
                break
        if is_detected:
            matched_terms.append(rw)
        else:
            missed_terms.append(rw)
    recall = len(matched_terms) / len(ref_kw) if len(ref_kw) > 0 else 1.0
    return recall, ", ".join(matched_terms), ", ".join(missed_terms)

def compute_keyword_metrics(ref, hyp):
    ref_words = str(ref).split()
    hyp_words = str(hyp).split()
    if not ref_words: return 1.0, "", ""
    matched_keywords = []
    missed_keywords = []
    for rw in ref_words:
        if len(rw) == 0: continue
        is_detected = False
        for hw in hyp_words:
            if len(hw) == 0: continue
            if Levenshtein.distance(rw, hw) / len(rw) <= 0.30:
                is_detected = True; break
        if is_detected: matched_keywords.append(rw)
        else: missed_keywords.append(rw)
    recall = len(matched_keywords) / len(ref_words)
    return recall, ", ".join(matched_keywords), ", ".join(missed_keywords)

def compute_all_metrics(ref, hyp):
    start_t = time.time()
    
    if not ref.strip():
        return np.nan, np.nan, 0.0, 1.0, "", "", 1.0, "", "", 1.0, "", "", "", "", time.time() - start_t
        
    try:
        ref_clean = wer_transform(ref)
        hyp_clean = wer_transform(hyp)
        wer = jiwer.wer(ref_clean, hyp_clean)
        cer = jiwer.cer(ref_clean, hyp_clean)
    except Exception:
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)
        
    sim = fuzz.ratio(ref, hyp)
    
    ct_rec, ct_match, ct_miss = compute_critical_term_metrics(ref, hyp)
    kw_rec, kw_match, kw_miss = compute_keyword_metrics(ref, hyp)
    num_recall, ref_nums, hyp_nums, num_missed, num_incorrect = compute_number_metrics(ref, hyp)
    
    metrics_time = time.time() - start_t
    
    return wer, cer, sim, kw_rec, kw_match, kw_miss, ct_rec, ct_match, ct_miss, num_recall, ref_nums, hyp_nums, num_missed, num_incorrect, metrics_time
