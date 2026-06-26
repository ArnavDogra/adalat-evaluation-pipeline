import torch
import time

class SileroVADModel:
    def __init__(self, repo_or_dir='snakers4/silero-vad', model_name='silero_vad'):
        self.vad_model, self.utils = torch.hub.load(
            repo_or_dir=repo_or_dir, 
            model=model_name, 
            force_reload=False, 
            onnx=False
        )
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils

    def process_audio(self, path, sampling_rate=16000):
        start_t = time.time()
        
        wav = self.read_audio(path, sampling_rate=sampling_rate)
        speech_timestamps = self.get_speech_timestamps(wav, self.vad_model, sampling_rate=sampling_rate)
        
        seg_count = len(speech_timestamps) if speech_timestamps else 1
        chunks = [wav[s['start']:s['end']].numpy() for s in speech_timestamps] if speech_timestamps else [wav.numpy()]
        
        total_speech_dur = sum([len(c)/sampling_rate for c in chunks])
        avg_seg_dur = total_speech_dur / seg_count if seg_count > 0 else 0
        
        vad_time = time.time() - start_t
        
        return {
            'chunks': chunks,
            'segment_count': seg_count,
            'avg_segment_duration': avg_seg_dur,
            'total_speech_duration': total_speech_dur,
            'vad_time': vad_time
        }
