import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

def generate_reports(df, output_dir: Path):
    start_t = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pipeline Results CSV
    df.to_csv(output_dir / 'pipeline_results.csv', index=False)
    
    # 2. Timing Breakdowns
    time_cols = ['profiling_time', 'bucketing_time', 'lid_time', 'vad_time', 'asr_time', 'metrics_time']
    time_sums = df[time_cols].sum()
    total_sum = time_sums.sum()
    
    timing_breakdown = pd.DataFrame({
        'Stage': [c.replace('_time', '').upper() for c in time_cols],
        'Total_Seconds': time_sums.values,
        'Percentage': (time_sums.values / total_sum) * 100
    })
    timing_breakdown.to_csv(output_dir / 'timing_breakdown.csv', index=False)
    
    # Pie Chart
    plt.figure(figsize=(10, 8))
    plt.pie(timing_breakdown['Percentage'], labels=timing_breakdown['Stage'], autopct='%1.1f%%', startangle=140)
    plt.title('Pipeline Timing Breakdown (Percentage)')
    plt.savefig(output_dir / 'timing_pie_chart.png')
    plt.close()
    
    # Horizontal Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=timing_breakdown, x='Total_Seconds', y='Stage', palette='viridis')
    plt.title('Pipeline Timing Breakdown (Seconds)')
    plt.savefig(output_dir / 'timing_bar_chart.png')
    plt.close()
    
    # 3. LID Summary
    lid_summary = df.groupby(['audio_bucket', 'detected_language']).agg(
        Sample_Count=('clip_id', 'count'),
        Average_Confidence=('language_confidence', 'mean'),
        Average_LID_Time=('lid_time', 'mean')
    ).reset_index()
    lid_summary.to_csv(output_dir / 'lid_summary.csv', index=False)
    
    # 4. One Minute Benchmark
    avg_pipeline_rtf = df['pipeline_rtf'].mean()
    benchmark_df = pd.DataFrame({
        'Audio_Duration': ['1 minute', '5 minutes', '10 minutes', '30 minutes', '60 minutes'],
        'Seconds': [60, 300, 600, 1800, 3600]
    })
    benchmark_df['Estimated_Processing_Time_Seconds'] = benchmark_df['Seconds'] * avg_pipeline_rtf
    benchmark_df['Estimated_Processing_Time_Minutes'] = benchmark_df['Estimated_Processing_Time_Seconds'] / 60
    benchmark_df.to_csv(output_dir / 'one_minute_benchmark.csv', index=False)
    
    # 5. HTML Report
    html_content = f"""
    <html><head><title>Pipeline Execution Report</title></head>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
    <h1>Pipeline Execution Report</h1>
    
    <h2>1. Timing Breakdown</h2>
    {timing_breakdown.to_html(index=False)}
    <br>
    <img src="timing_pie_chart.png" width="500">
    <img src="timing_bar_chart.png" width="500">
    
    <h2>2. One Minute Benchmark (Average RTF: {avg_pipeline_rtf:.3f})</h2>
    {benchmark_df.drop(columns=['Seconds']).to_html(index=False)}
    
    <h2>3. Language Identification Summary</h2>
    {lid_summary.to_html(index=False)}
    
    <h2>4. Audio Bucket Summary (Hindi Only for ASR)</h2>
    """
    
    hindi_df = df[df['detected_language'] == 'hin']
    if not hindi_df.empty:
        overall = {
            'Sample Count': len(hindi_df),
            'Mean WER': hindi_df['high_lr_wer'].mean(),
            'Mean CER': hindi_df['high_lr_cer'].mean(),
            'Mean Keyword Recall': hindi_df['high_lr_keyword_recall'].mean(),
            'Mean Critical Term Recall': hindi_df['high_lr_critical_term_recall'].mean(),
            'Mean Number Recall': hindi_df['high_lr_number_recall'].mean(),
        }
        overall_df = pd.DataFrame([overall])
        html_content += overall_df.to_html(index=False)
        html_content += "<h3>Bucket-wise Metrics</h3>"
        
        bucket_summary = []
        for bucket in ['GOOD', 'MODERATE', 'BAD']:
            b_df = hindi_df[hindi_df['audio_bucket'] == bucket]
            if not b_df.empty:
                bucket_summary.append({
                    'Bucket': bucket,
                    'Sample Count': len(b_df),
                    'Mean WER': b_df['high_lr_wer'].mean(),
                    'Mean CER': b_df['high_lr_cer'].mean(),
                })
        bucket_df = pd.DataFrame(bucket_summary)
        html_content += bucket_df.to_html(index=False)
    else:
        html_content += "<p>No Hindi audio found for ASR metrics.</p>"
        
    html_content += "</body></html>"
    with open(output_dir / 'final_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    report_time = time.time() - start_t
    return report_time
