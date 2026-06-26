import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

def main():
    os.makedirs('audio', exist_ok=True)
    
    indian_languages = ['hi_in', 'bn_in', 'ta_in', 'te_in', 'mr_in']
    
    records = []
    
    for lang in indian_languages:
        print(f'Downloading FLEURS {lang} dataset (streaming mode)...')
        ds = load_dataset('google/fleurs', lang, split='test', streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        ds_subset = ds.take(25)
        
        print(f'Saving {lang} audio files to ./audio ...')
        for i, item in enumerate(ds_subset):
            clip_id = f'fleurs_{lang}_{i:03d}'
            audio_path = f'audio/{clip_id}.wav'
            
            # Save raw audio bytes directly to bypass Windows torchcodec DLL errors
            with open(audio_path, 'wb') as f:
                f.write(item['audio']['bytes'])
            
            records.append({
                'clip_id': clip_id,
                'reference_transcript': item['raw_transcription']
            })
        
    print('Generating transcripts.csv...')
    df = pd.DataFrame(records)
    df.to_csv('transcripts.csv', index=False)
    print(f'Successfully downloaded 125 files (25 per language) to audio/ and generated transcripts.csv!')

if __name__ == '__main__':
    main()
