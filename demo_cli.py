from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from functools import lru_cache, cached_property
import numpy as np
import torch
import librosa
import soundfile as sf
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

@dataclass(frozen=True)
class VoiceClonerConfig:
 enc_model_fpath: Path
 syn_model_fpath: Path
 voc_model_fpath: Path
 cpu: bool
 no_sound: bool
 seed: Optional[int]

class DeviceManager:
 def __init__(self) -> None:
  self._devices: List[str] = ["CPU", "CUDA", "MPS"]

 @lru_cache(maxsize=None)
 def _canonicalize(self, device: str) -> str:
  return device.upper()

 def canonicalize(self, device: Optional[str]) -> str:
  return self._canonicalize(device) if device is not None else self.DEFAULT

 @cached_property
 def DEFAULT(self) -> str:
  if torch.cuda.is_available(): return "CUDA"
  elif torch.backends.mps.is_available(): return "MPS"
  return "CPU"

 def get_device(self, device: str) -> torch.device:
  device = self.canonicalize(device)
  if device == "CPU": return torch.device("cpu")
  elif device == "CUDA": return torch.device("cuda")
  elif device == "MPS": return torch.device("mps")
  raise ValueError(f"Unsupported device: {device}")

class AudioProcessor:
 @staticmethod
 def load_wav(path: Path) -> np.ndarray:
  return encoder.preprocess_wav(path)

 @staticmethod
 def embed_utterance(wav: np.ndarray) -> np.ndarray:
  return encoder.embed_utterance(wav)

class SpeechSynthesizer:
 def __init__(self, config: VoiceClonerConfig):
  self.config = config
  self.synthesizer: Optional[Synthesizer] = None
  self.device_manager = DeviceManager()

 def setup(self) -> None:
  if self.config.cpu: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  device = self.device_manager.get_device(None)
  print(f"Using device: {device}")
  if self.config.seed is not None: torch.manual_seed(self.config.seed)
  self._load_models()

 def _load_models(self) -> None:
  print("Preparing the encoder, synthesizer, and vocoder...")
  ensure_default_models(Path("saved_models"))
  encoder.load_model(self.config.enc_model_fpath)
  self.synthesizer = Synthesizer(self.config.syn_model_fpath)
  vocoder.load_model(self.config.voc_model_fpath)

 def test_configuration(self) -> None:
  print("Testing your configuration with small inputs.")
  print("\tTesting the encoder...")
  encoder.embed_utterance(np.zeros(encoder.sampling_rate))
  embed = np.random.rand(speaker_embedding_size)
  embed /= np.linalg.norm(embed)
  embeds = [embed, np.zeros(speaker_embedding_size)]
  texts = ["test 1", "test 2"]
  print("\tTesting the synthesizer...")
  mels = self.synthesizer.synthesize_spectrograms(texts, embeds)
  mel = np.concatenate(mels, axis=1)
  print("\tTesting the vocoder...")
  vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=lambda *args: None)
  print("All tests passed! You can now synthesize speech.\n\n")

 def generate_speech(self, embed: np.ndarray, text: str) -> Tuple[np.ndarray, int]:
  if self.config.seed is not None:
   torch.manual_seed(self.config.seed)
   self.synthesizer = Synthesizer(self.config.syn_model_fpath)
  specs = self.synthesizer.synthesize_spectrograms([text], [embed])
  spec = specs[0]
  if self.config.seed is not None:
   torch.manual_seed(self.config.seed)
   vocoder.load_model(self.config.voc_model_fpath)
  generated_wav = vocoder.infer_waveform(spec)
  generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
  generated_wav = encoder.preprocess_wav(generated_wav)
  return generated_wav, self.synthesizer.sample_rate

class AudioPlayer:
 @staticmethod
 def play(wav: np.ndarray, sample_rate: int, no_sound: bool) -> None:
  if not no_sound:
   import sounddevice as sd
   try:
    sd.stop()
    sd.play(wav, sample_rate)
   except sd.PortAudioError as e:
    print(f"\nCaught exception: {repr(e)}")
    print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
   except Exception:
    raise

class VoiceCloner:
 def __init__(self, config: VoiceClonerConfig):
  self.config = config
  self.synthesizer = SpeechSynthesizer(config)
  self.audio_processor = AudioProcessor()
  self.audio_player = AudioPlayer()

 def run(self) -> None:
  self.synthesizer.setup()
  self.synthesizer.test_configuration()
  print("Interactive generation loop")
  num_generated = 0
  while True:
   try:
    in_fpath = Path(input("Reference voice: enter an audio filepath of a voice to be cloned (mp3, wav, m4a, flac, ...):\n").replace("\"", "").replace("\'", ""))
    wav = self.audio_processor.load_wav(in_fpath)
    embed = self.audio_processor.embed_utterance(wav)
    print("Created the embedding")
    text = input("Write a sentence (+-20 words) to be synthesized:\n")
    generated_wav, sample_rate = self.synthesizer.generate_speech(embed, text)
    self.audio_player.play(generated_wav, sample_rate, self.config.no_sound)
    filename = f"demo_output_{num_generated:02d}.wav"
    sf.write(filename, generated_wav.astype(np.float32), sample_rate)
    num_generated += 1
    print(f"\nSaved output as {filename}\n\n")
   except Exception as e:
    print(f"Caught exception: {repr(e)}")
    print("Restarting\n")

def parse_arguments() -> VoiceClonerConfig:
 parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 parser.add_argument("-e", "--enc_model_fpath", type=Path, default="saved_models/default/encoder.pt")
 parser.add_argument("-s", "--syn_model_fpath", type=Path, default="saved_models/default/synthesizer.pt")
 parser.add_argument("-v", "--voc_model_fpath", type=Path, default="saved_models/default/vocoder.pt")
 parser.add_argument("--cpu", action="store_true", help="If True, processing is done on CPU, even when a GPU is available.")
 parser.add_argument("--no_sound", action="store_true", help="If True, audio won't be played.")
 parser.add_argument("--seed", type=int, default=None, help="Optional random number seed value to make toolbox deterministic.")
 args = parser.parse_args()
 print_args(args, parser)
 return VoiceClonerConfig(enc_model_fpath=args.enc_model_fpath, syn_model_fpath=args.syn_model_fpath, voc_model_fpath=args.voc_model_fpath, cpu=args.cpu, no_sound=args.no_sound, seed=args.seed)

if __name__ == '__main__':
 config = parse_arguments()
 cloner = VoiceCloner(config)
 cloner.run()