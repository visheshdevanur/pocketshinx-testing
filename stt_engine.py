import io
import time
import soundfile as sf
import librosa
from pocketsphinx import Decoder

class STTEngine:
    def __init__(self):
        # Initialize PocketSphinx decoder with default acoustic and language models
        self.decoder = Decoder()
        
    def _convert_to_16k_mono(self, audio_bytes):
        """
        Converts any audio byte stream to 16kHz mono PCM raw bytes.
        """
        # Load audio from bytes
        y, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y.T)
            
        # Resample to 16kHz if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            
        # Export to a temporary 16kHz mono wav in memory
        out_io = io.BytesIO()
        sf.write(out_io, y, 16000, format='WAV', subtype='PCM_16')
        out_io.seek(0)
        return out_io.read()
        
    def transcribe(self, audio_bytes):
        """
        Transcribes the audio bytes and returns text and latency.
        """
        start_time = time.time()
        
        # Ensure 16k mono WAV format
        wav_bytes = self._convert_to_16k_mono(audio_bytes)
        
        # Get raw PCM data directly (skip WAV header)
        y, _ = sf.read(io.BytesIO(wav_bytes), dtype='int16')
        raw_pcm = y.tobytes()
        
        # Transcribe using PocketSphinx
        self.decoder.start_utt()
        self.decoder.process_raw(raw_pcm, False, True)
        self.decoder.end_utt()
        
        latency = time.time() - start_time
        hypothesis = self.decoder.hyp()
        text = hypothesis.hypstr if hypothesis else ""
        
        return text.strip(), latency
