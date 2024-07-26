from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from functools import partial

import librosa
import numpy as np
import soundfile as sf
import torch

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

class VoiceCloner:
    def __init__(self, config: VoiceClonerConfig):
        self.config: VoiceClonerConfig = config
        self.synthesizer: Optional[Synthesizer] = None
        self.num_generated: int = 0

    def setup_environment(self) -> None:
        """Set up the environment based on configuration."""
        if self.config.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print("Running a test of your configuration...\n")

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
            print(f"Found {torch.cuda.device_count()} GPUs available. Using GPU {device_id} "
                  f"({gpu_properties.name}) of compute capability {gpu_properties.major}."
                  f"{gpu_properties.minor} with {gpu_properties.total_memory / 1e9:.1f}Gb total memory.\n")
        else:
            print("Using CPU for inference.\n")

    def load_models(self) -> None:
        """Load the encoder, synthesizer, and vocoder models."""
        print("Preparing the encoder, the synthesizer and the vocoder...")
        ensure_default_models(Path("saved_models"))
        encoder.load_model(self.config.enc_model_fpath)
        self.synthesizer = Synthesizer(self.config.syn_model_fpath)
        vocoder.load_model(self.config.voc_model_fpath)

    def test_configuration(self) -> None:
        """Run a test of the configuration with small inputs."""
        print("Testing your configuration with small inputs.")
        
        print("\tTesting the encoder...")
        encoder.embed_utterance(np.zeros(encoder.sampling_rate))

        embed = np.random.rand(speaker_embedding_size)
        embed /= np.linalg.norm(embed)
        embeds = [embed, np.zeros(speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        
        print("\tTesting the synthesizer... (loading the model will output a lot of text)")
        mels = self.synthesizer.synthesize_spectrograms(texts, embeds)

        mel = np.concatenate(mels, axis=1)
        
        print("\tTesting the vocoder...")
        vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=lambda *args: None)

        print("All tests passed! You can now synthesize speech.\n\n")

    @staticmethod
    def preprocess_audio(in_fpath: Path) -> np.ndarray:
        """Preprocess the input audio file."""
        wav = encoder.preprocess_wav(in_fpath)
        return encoder.embed_utterance(wav)

    def generate_speech(self, embed: np.ndarray, text: str) -> Tuple[np.ndarray, int]:
        """Generate speech from embedding and text."""
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

    @staticmethod
    def save_audio(wav: np.ndarray, sample_rate: int, filename: str) -> None:
        """Save the generated audio to a file."""
        sf.write(filename, wav.astype(np.float32), sample_rate)

    def play_audio(self, wav: np.ndarray, sample_rate: int) -> None:
        """Play the generated audio if not disabled."""
        if not self.config.no_sound:
            import sounddevice as sd
            try:
                sd.stop()
                sd.play(wav, sample_rate)
            except sd.PortAudioError as e:
                print(f"\nCaught exception: {repr(e)}")
                print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
            except Exception:
                raise

    def run_interactive_loop(self) -> None:
        """Run the interactive speech generation loop."""
        print("Interactive generation loop")
        while True:
            try:
                in_fpath = self.get_input_filepath()
                embed = self.preprocess_audio(in_fpath)
                print("Created the embedding")

                text = input("Write a sentence (+-20 words) to be synthesized:\n")

                generated_wav, sample_rate = self.generate_speech(embed, text)

                self.play_audio(generated_wav, sample_rate)

                filename = f"demo_output_{self.num_generated:02d}.wav"
                self.save_audio(generated_wav, sample_rate, filename)
                self.num_generated += 1
                print(f"\nSaved output as {filename}\n\n")

            except Exception as e:
                print(f"Caught exception: {repr(e)}")
                print("Restarting\n")

    @staticmethod
    def get_input_filepath() -> Path:
        """Get the input filepath from user."""
        message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, wav, m4a, flac, ...):\n"
        return Path(input(message).replace("\"", "").replace("\'", ""))

def parse_arguments() -> VoiceClonerConfig:
    """Parse command-line arguments and return a VoiceClonerConfig object."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--enc_model_fpath", type=Path, default="saved_models/default/encoder.pt")
    parser.add_argument("-s", "--syn_model_fpath", type=Path, default="saved_models/default/synthesizer.pt")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, default="saved_models/default/vocoder.pt")
    parser.add_argument("--cpu", action="store_true", help="If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help="If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random number seed value to make toolbox deterministic.")
    
    args = parser.parse_args()
    print_args(args, parser)
    
    return VoiceClonerConfig(
        enc_model_fpath=args.enc_model_fpath,
        syn_model_fpath=args.syn_model_fpath,
        voc_model_fpath=args.voc_model_fpath,
        cpu=args.cpu,
        no_sound=args.no_sound,
        seed=args.seed
    )

def main() -> None:
    """Main function to set up and run the Voice Cloner."""
    config = parse_arguments()
    cloner = VoiceCloner(config)
    cloner.setup_environment()
    cloner.load_models()
    cloner.test_configuration()
    cloner.run_interactive_loop()

if __name__ == '__main__':
    main()