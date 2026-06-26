import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset

def main():
    os.makedirs('audio', exist_ok=True)
    
    print('Downloading FLEURS Hindi dataset...')
    # Load the test split of the Hindi subset of FLEURS
    ds = load_dataset('google/fleurs', 'hi_in', split='test', streaming=False)
    
    # Select the first 25 files to match your previous evaluation size
    ds_subset = ds.select(range(25))
    
    records = []
    print('Saving audio files to ./audio and generating transcripts.csv...')
    for i, item in enumerate(ds_subset):
        clip_id = f'fleurs_hi_{i:03d}'
        audio_path = f'audio/{clip_id}.wav'
        
        # Save audio arrays to WAV format
        audio_data = item['audio']['array']
        sr = item['audio']['sampling_rate']
        sf.write(audio_path, audio_data, sr)
        
        records.append({
            'clip_id': clip_id,
            'reference_transcript': item['raw_transcription']
        })
        
    df = pd.DataFrame(records)
    df.to_csv('transcripts.csv', index=False)
    print(f'Successfully downloaded 25 Hindi files to audio/ and generated transcripts.csv!')

if __name__ == '__main__':
    main()
