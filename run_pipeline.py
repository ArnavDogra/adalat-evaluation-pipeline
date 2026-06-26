import argparse
import time
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import load_config
from utils import setup_logging, validate_environment
from models.mms_lid import MMSLidModel
from models.adalat_asr import AdalatASRModel
from models.vad import SileroVADModel
from pipeline.profiling import profile_audio, compute_buckets
from pipeline.metrics import compute_all_metrics
from pipeline.reporting import generate_reports

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Audio Evaluation Pipeline")
    parser.add_argument("--audio_dir", type=str, help="Path to input audio directory")
    parser.add_argument("--transcript_csv", type=str, help="Path to reference transcripts CSV")
    parser.add_argument("--output_dir", type=str, help="Path to save outputs")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    return parser.parse_args()

def main():
    pipeline_start_t = time.time()
    args = parse_args()
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.audio_dir: config['directories']['audio_dir'] = args.audio_dir
    if args.transcript_csv: config['directories']['transcript_csv'] = args.transcript_csv
    if args.output_dir: config['directories']['output_dir'] = args.output_dir
        
    out_dir = Path(config['directories']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(out_dir)
    logger.info("Starting Pipeline Execution...")
    
    device = validate_environment(logger)
    
    # 1. Discover Audio and Match Transcripts
    logger.info("Scanning for audio files...")
    audio_dir = Path(config['directories']['audio_dir'])
    audio_paths = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.mp3"))
    if not audio_paths:
        logger.error(f"No audio files found in {audio_dir}")
        return
        
    audio_df = pd.DataFrame({'full_path': [str(p) for p in audio_paths]})
    audio_df['filename'] = audio_df['full_path'].apply(lambda x: Path(x).name)
    audio_df['clip_id'] = audio_df['filename'].apply(lambda x: Path(x).stem)
    
    csv_path = Path(config['directories']['transcript_csv'])
    if csv_path.exists():
        sarvam_df = pd.read_csv(csv_path)
        if 'clip_id' not in sarvam_df.columns and 'audio' in sarvam_df.columns:
            sarvam_df['clip_id'] = sarvam_df['audio'].astype(str).apply(lambda x: Path(x).stem)
            
        master_df = pd.merge(audio_df, sarvam_df[['clip_id', 'reference_transcript']], on='clip_id', how='inner')
        logger.info(f"Successfully matched {len(master_df)} files with transcripts.")
    else:
        logger.warning(f"Transcript CSV not found at {csv_path}. Running in timing-only mode (metrics will be skipped).")
        master_df = audio_df.copy()
        
    if master_df.empty:
        logger.error("No valid audio files to process.")
        return
        
    # 2. Audio Profiling
    logger.info("Profiling Audio...")
    tqdm.pandas(desc="Profiling")
    cols = ['duration', 'sample_rate', 'rms_energy', 'silence_ratio', 'dynamic_range', 'signal_variance', 'clipping_ratio', 'audio_array']
    profiling_start = time.time()
    master_df[cols] = master_df['full_path'].progress_apply(profile_audio)
    profiling_time = time.time() - profiling_start
    master_df['profiling_time'] = profiling_time / len(master_df)
    
    min_dur = config['thresholds']['min_duration']
    max_dur = config['thresholds']['max_duration']
    master_df = master_df[(master_df['duration'] > min_dur) & (master_df['duration'] <= max_dur)].copy()
    logger.info(f"Eligible files after duration filtering: {len(master_df)}")
    if master_df.empty: return
    
    # 3. Bucketing
    logger.info("Computing Buckets...")
    master_df, bucketing_time = compute_buckets(master_df)
    master_df['bucketing_time'] = bucketing_time / len(master_df)
    
    # 4. Initialize Models
    logger.info("Initializing Models (this may trigger downloads)...")
    lid_model = MMSLidModel(model_id=config['models']['lid_model'], device=device)
    asr_model = AdalatASRModel(model_id=config['models']['asr_model'], device=device, chunk_length_s=config['thresholds']['asr_chunk_length_s'])
    vad_model = SileroVADModel(repo_or_dir=config['models']['vad_repo_name'], model_name='silero_vad')
    
    # 5. Execution Loop
    results = []
    logger.info("Starting Pipeline Inference...")
    for idx, row in tqdm(master_df.iterrows(), total=len(master_df)):
        file_start_t = time.time()
        
        # LID
        lid_start = time.time()
        lid_res = lid_model.detect_language(row['audio_array'], row['sample_rate'])
        lid_time = time.time() - lid_start
        
        res = {
            'clip_id': row['clip_id'],
            'detected_language': lid_res['detected_language'],
            'language_confidence': lid_res['language_confidence'],
            'top5_languages': str(lid_res['top5_languages']),
            'top5_scores': str(lid_res['top5_scores']),
            'lid_time': lid_time
        }
        
        # Routing
        if res['detected_language'] in ['hin', 'hi_in']:
            # VAD
            vad_res = vad_model.process_audio(row['full_path'], sampling_rate=config['thresholds']['vad_sampling_rate'])
            res.update({
                'vad_time': vad_res['vad_time'],
                'high_lr_segment_count': vad_res['segment_count'],
                'high_lr_avg_segment_duration': vad_res['avg_segment_duration'],
                'high_lr_total_speech_duration': vad_res['total_speech_duration']
            })
            
            # ASR
            transcript, asr_time = asr_model.transcribe_chunks(vad_res['chunks'])
            res['high_lr_transcript'] = transcript
            res['asr_time'] = asr_time
            
            # Metrics
            if 'reference_transcript' in row and pd.notna(row['reference_transcript']):
                met = compute_all_metrics(str(row['reference_transcript']), transcript)
                res.update({
                    'high_lr_wer': met[0],
                    'high_lr_cer': met[1],
                    'high_lr_similarity': met[2],
                    'high_lr_keyword_recall': met[3],
                    'high_lr_matched_keywords': met[4],
                    'high_lr_missed_keywords': met[5],
                    'high_lr_critical_term_recall': met[6],
                    'high_lr_matched_critical_terms': met[7],
                    'high_lr_missed_critical_terms': met[8],
                    'high_lr_number_recall': met[9],
                    'high_lr_reference_numbers': met[10],
                    'high_lr_adalat_numbers': met[11],
                    'high_lr_missed_numbers': met[12],
                    'high_lr_incorrect_numbers': met[13],
                    'metrics_time': met[14],
                    'reason': 'Success'
                })
            else:
                res.update({
                    'high_lr_wer': np.nan,
                    'high_lr_cer': np.nan,
                    'high_lr_similarity': np.nan,
                    'high_lr_keyword_recall': np.nan,
                    'high_lr_matched_keywords': '',
                    'high_lr_missed_keywords': '',
                    'high_lr_critical_term_recall': np.nan,
                    'high_lr_matched_critical_terms': '',
                    'high_lr_missed_critical_terms': '',
                    'high_lr_number_recall': np.nan,
                    'high_lr_reference_numbers': '',
                    'high_lr_adalat_numbers': '',
                    'high_lr_missed_numbers': '',
                    'high_lr_incorrect_numbers': '',
                    'metrics_time': 0.0,
                    'reason': 'Timing only (No Transcript)'
                })
        else:
            # Skip ASR for unsupported languages
            res.update({
                'vad_time': 0.0,
                'high_lr_segment_count': np.nan,
                'high_lr_avg_segment_duration': np.nan,
                'high_lr_total_speech_duration': np.nan,
                'high_lr_transcript': 'unsupported_language',
                'asr_time': 0.0,
                'high_lr_wer': np.nan,
                'high_lr_cer': np.nan,
                'high_lr_similarity': np.nan,
                'high_lr_keyword_recall': np.nan,
                'high_lr_matched_keywords': '',
                'high_lr_missed_keywords': '',
                'high_lr_critical_term_recall': np.nan,
                'high_lr_matched_critical_terms': '',
                'high_lr_missed_critical_terms': '',
                'high_lr_number_recall': np.nan,
                'high_lr_reference_numbers': '',
                'high_lr_adalat_numbers': '',
                'high_lr_missed_numbers': '',
                'high_lr_incorrect_numbers': '',
                'metrics_time': 0.0,
                'reason': 'Unsupported language'
            })
        
        file_time = time.time() - file_start_t
        res['pipeline_time_per_file'] = file_time
        res['pipeline_rtf'] = file_time / row['duration']
        res['lid_rtf'] = lid_time / row['duration']
        res['asr_rtf'] = res['asr_time'] / row['duration'] if res['asr_time'] > 0 else np.nan
        
        results.append(res)
        
    res_df = pd.DataFrame(results)
    final_df = pd.merge(master_df.drop(columns=['audio_array']), res_df, on='clip_id', how='left')
    
    # 6. Reporting
    logger.info("Generating Reports...")
    report_time = generate_reports(final_df, out_dir)
    
    # Backfill report time for completeness in final summary (apportioned per file)
    final_df['report_time'] = report_time / len(final_df)
    
    # Re-save with report time
    final_df.to_csv(out_dir / 'pipeline_results.csv', index=False)
    
    total_t = time.time() - pipeline_start_t
    logger.info(f"Pipeline Completed Successfully in {total_t:.2f} seconds.")

if __name__ == "__main__":
    main()
