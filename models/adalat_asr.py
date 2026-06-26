from transformers import pipeline
import time

class AdalatASRModel:
    def __init__(self, model_id='adalat-ai/whisper-medium-hi-high-lr', device='cpu', chunk_length_s=30):
        self.device_id = 0 if device == 'cuda' else -1
        self.pipe = pipeline(
            'automatic-speech-recognition', 
            model=model_id, 
            device=self.device_id, 
            chunk_length_s=chunk_length_s
        )

    def transcribe_chunks(self, chunks_list):
        start_t = time.time()
        # Ensure chunks is a list of numpy arrays
        text_pieces = [self.pipe(c).get('text', '') for c in chunks_list]
        transcript = ' '.join(text_pieces).strip()
        inf_time = time.time() - start_t
        return transcript, inf_time
