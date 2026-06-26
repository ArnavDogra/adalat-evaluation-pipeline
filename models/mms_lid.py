import torch
import torchaudio
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class MMSLidModel:
    def __init__(self, model_id='facebook/mms-lid-256', device='cpu'):
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def detect_language(self, audio_array, orig_sr):
        # MMS requires 16000 Hz input
        if orig_sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=orig_sr, target_sr=16000)
            
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device))
            logits = outputs.logits
            
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        top5_indices = torch.topk(probs, 5).indices.tolist()
        top5_probs = torch.topk(probs, 5).values.tolist()
        
        top5_langs = [self.id2label[i] for i in top5_indices]
        
        detected_lang = top5_langs[0]
        confidence = top5_probs[0]
        
        return {
            'detected_language': detected_lang,
            'language_confidence': confidence,
            'top5_languages': top5_langs,
            'top5_scores': top5_probs
        }
