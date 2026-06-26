import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MinMaxScaler
import time

def profile_audio(path):
    try:
        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.sqrt(np.mean(y**2))
        intervals = librosa.effects.split(y, top_db=30)
        non_silent = sum([(e - s) for s, e in intervals]) / sr
        silence_ratio = 1.0 - (non_silent / duration) if duration > 0 else 0
        dynamic_range = 20 * np.log10(np.max(np.abs(y)) / (np.min(np.abs(y[y!=0])) + 1e-9) + 1e-9)
        signal_variance = np.var(y)
        clipping_ratio = np.sum(np.abs(y) >= 0.99) / len(y)
        
        return pd.Series([duration, sr, rms, silence_ratio, dynamic_range, signal_variance, clipping_ratio, y])
    except Exception as e: 
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None])

def compute_buckets(master_df):
    start_t = time.time()
    
    master_df['word_count'] = master_df['reference_transcript'].apply(lambda x: len(str(x).split()))
    master_df['speaking_rate'] = master_df['word_count'] / (master_df['duration'] + 1e-9)
    
    scaler = MinMaxScaler()
    norm_rms = scaler.fit_transform(master_df[['rms_energy']])
    norm_var = scaler.fit_transform(master_df[['signal_variance']])
    norm_dr = scaler.fit_transform(master_df[['dynamic_range']])
    norm_silence = 1 - scaler.fit_transform(master_df[['silence_ratio']])
    norm_clipping = 1 - scaler.fit_transform(master_df[['clipping_ratio']])
    norm_clarity = scaler.fit_transform(master_df[['speaking_rate']])
    
    score = (
        0.15 * norm_rms + 
        0.10 * norm_var + 
        0.10 * norm_dr + 
        0.15 * norm_silence + 
        0.20 * norm_clipping + 
        0.30 * norm_clarity  
    )
    master_df['quality_score'] = score
    
    q33, q66 = master_df['quality_score'].quantile([0.33, 0.66])
    def assign_bucket(s): return 'GOOD' if s >= q66 else ('MODERATE' if s >= q33 else 'BAD')
    master_df['audio_bucket'] = master_df['quality_score'].apply(assign_bucket)
    
    bucketing_time = time.time() - start_t
    return master_df, bucketing_time
