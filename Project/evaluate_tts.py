# evaluate_tts.py
# Usage:
#   python evaluate_tts.py --sentences test_sentences.txt --outdir outputs --device cuda
#
import os
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import torch

# ----------------------
# Utils: device movers / safe vocoder call
# ----------------------
def move_model_to_device(model, device):
    """Recursively try to move nested submodules/params to device."""
    try:
        model.to(device)
    except Exception:
        pass
    # try common nested names
    for attr in ('generator', 'model', 'netG', 'wavegen'):
        if hasattr(model, attr):
            sub = getattr(model, attr)
            try:
                sub.to(device)
            except Exception:
                pass

def safe_vocoder_call(vocoder, mel_tensor):
    """Try common invocation patterns and return numpy waveform (float32, -1..1)."""
    # prefer .inference()
    if hasattr(vocoder, 'inference'):
        out = vocoder.inference(mel_tensor)
    else:
        # try callable
        try:
            out = vocoder(mel_tensor)
        except Exception:
            out = vocoder.forward(mel_tensor)
    # out might be tensor or np array
    if isinstance(out, torch.Tensor):
        out_np = out.detach().cpu().numpy()
    else:
        out_np = np.asarray(out)
    # handle shape (1, T) or (T,) or (1,1,T)
    out_np = out_np.squeeze()
    return out_np

# ----------------------
# MCD computation (MFCC + DTW)
# ----------------------
def compute_mcd(ref_wav, syn_wav, sr, n_mfcc=13, hop_length=256):
    """
    Compute an approximate MCD using MFCCs + DTW alignment.
    Returns average MCD (dB) across aligned frames.
    """
    # load/resample outside if given arrays; here ref_wav, syn_wav are arrays
    # compute mfcc (librosa returns shape (n_mfcc, frames))
    S_ref = librosa.feature.mfcc(y=ref_wav, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    S_syn = librosa.feature.mfcc(y=syn_wav, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # distance matrix (euclidean)
    D = np.linalg.norm(S_ref[:, :, None] - S_syn[:, None, :], axis=0)
    # DTW (librosa.sequence.dtw)
    from librosa.sequence import dtw
    path, wp = dtw(C=D)  # path is (m+1, n+1) cumulative cost? librosa returns (Dacc, wp)
    # librosa >= 0.8 returns D, wp; wp is mapping; take wp
    # wp is list of (i,j) from end to start; reverse for order
    wp = wp[::-1]
    # compute per-pair euclidean on MFCC and then MCD formula
    diffs = []
    for (i, j) in wp:
        # ensure indices in bounds for S_ref / S_syn
        if i < S_ref.shape[1] and j < S_syn.shape[1]:
            vec_r = S_ref[:, i]
            vec_s = S_syn[:, j]
            diffs.append(np.linalg.norm(vec_r - vec_s))
    diffs = np.array(diffs)
    if diffs.size == 0:
        return np.nan
    # Convert MFCC Euclidean to MCD approximate:
    # standard conversion factor: MCD = (10 / ln10) * sqrt(2) * sqrt(sum diff^2)
    # but diffs already sqrt(sum diff^2) per frame, so:
    k = (10.0 / np.log(10.0)) * np.sqrt(2.0)
    mcd_frame = k * diffs  # per frame
    return float(np.mean(mcd_frame))

# ----------------------
# Main evaluation flow
# ----------------------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wavdir = outdir / 'wavs'
    wavdir.mkdir(exist_ok=True)

    # load sentences
    with open(args.sentences, 'r', encoding='utf-8') as f:
        sents = [l.strip() for l in f if l.strip()]
    # optional ground truth directory: files named 000.wav, 001.wav ... or matching pattern
    has_ref = args.ground_truth is not None
    if has_ref:
        ref_dir = Path(args.ground_truth)
        if not ref_dir.exists():
            print("Ground truth dir not found:", ref_dir)
            has_ref = False

    # Load models via torch.hub (NVIDIA tacotron2 and seungwonpark/melgan)
    print("Loading Tacotron2 from torch.hub ...")
    tacotron = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
    tacotron.eval()
    print("Loading tacotron utils ...")
    tacotron_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32', verbose=False)
    # (above may return same or utils, fallback: try to import local text processing)
    # For MelGAN:
    print("Loading MelGAN vocoder via torch.hub ...")
    try:
        vocoder = torch.hub.load('seungwonpark/melgan', 'melgan', pretrained=True)
    except Exception:
        vocoder = torch.hub.load('seungwonpark/melgan', 'melgan', force_reload=True)
    vocoder.eval()

    # Ensure moved to device
    move_model_to_device(tacotron, device)
    move_model_to_device(vocoder, device)

    results = []
    sr = args.sr

    # helper: text -> sequence using tacotron utils if available
    try:
        from tacotron2.text import text_to_sequence  # sometimes available in repo
        def prepare_sequence(txt):
            seq = text_to_sequence(txt, ['english_cleaners'])
            return torch.LongTensor(seq).unsqueeze(0)
    except Exception:
        # fallback using NVIDIA hub utils if present (nvidia_tacotron2 provides `text_to_sequence` in some versions)
        try:
            from torch.hub import load_state_dict_from_url
            # fallback minimal prepare: simple char->ord mapping (NOT ideal for production)
            def prepare_sequence(txt):
                seq = [ord(c) for c in txt]
                return torch.LongTensor(seq).unsqueeze(0)
            print("Warning: using simple char->ord fallback for tokenization (best to use proper tts utils).")
        except Exception:
            raise RuntimeError("No text processing utility available. Please ensure tacotron2 utils are present.")

    # iterate sentences
    for i, text in enumerate(sents):
        print(f"[{i}] Synthesizing: {text[:80]}...")
        # 1) prepare sequence
        seq = prepare_sequence(text).to(device)

        # Tacotron2 inference (compatible with NVIDIA torch.hub)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()

        # 1. Tacotron2 hub version requires both seq and input_lengths
        input_lengths = torch.LongTensor([seq.shape[1]]).to(device)

        try:
            mel_outputs, mel_lengths, alignments = tacotron.infer(seq, input_lengths)
        except TypeError:
            # fallback: some versions return different tuple or accept different signatures
            try:
                mel_outputs = tacotron.infer(seq, input_lengths)
            except Exception:
                mel_outputs = tacotron(seq)

        # extract mel
        if isinstance(mel_outputs, (tuple, list)):
            mel = mel_outputs[0]
        else:
            mel = mel_outputs

        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.time()
        tacotron_time = t1 - t0


        # ensure mel is a torch.Tensor on device and shape [1, n_mels, T]
        if isinstance(mel, torch.Tensor):
            mel_tensor = mel.to(device)
        else:
            mel_tensor = torch.tensor(mel).to(device)

        # 3) Vocoder timing
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        audio = safe_vocoder_call(vocoder, mel_tensor)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.time()
        vocoder_time = t1 - t0

        # normalize audio to float32 -1..1
        if np.max(np.abs(audio)) > 0:
            audio = audio / max(0.9999, np.max(np.abs(audio)))

        out_wav = wavdir / f"{i:04d}.wav"
        sf.write(str(out_wav), audio.astype(np.float32), sr)

        duration = librosa.get_duration(y=audio, sr=sr)
        total_time = tacotron_time + vocoder_time
        rtf = total_time / duration if duration > 0 else np.nan

        # compute optional metrics if reference exists
        mcd_val = None
        mel_l2 = None
        if has_ref:
            ref_path = ref_dir / f"{i:04d}.wav"
            if ref_path.exists():
                ref_audio, _ = librosa.load(str(ref_path), sr=sr)
                # compute MCD
                try:
                    mcd_val = compute_mcd(ref_audio, audio, sr, n_mfcc=13, hop_length=256)
                except Exception as e:
                    print("MCD compute error:", e)
                    mcd_val = np.nan
                # mel-L2: compute mel spectrograms and L2
                ref_mel = librosa.feature.melspectrogram(y=ref_audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
                syn_mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
                # align frames by truncation to min frames
                minf = min(ref_mel.shape[1], syn_mel.shape[1])
                mel_l2 = float(np.linalg.norm(ref_mel[:, :minf] - syn_mel[:, :minf]))

        results.append({
            'idx': i,
            'text': text,
            'out_wav': str(out_wav),
            'duration_s': float(duration),
            'tacotron_s': float(tacotron_time),
            'vocoder_s': float(vocoder_time),
            'total_time_s': float(total_time),
            'RTF': float(rtf),
            'MCD_dB': mcd_val,
            'mel_L2': mel_l2
        })
        # flush
        pd.DataFrame(results).to_csv(outdir / 'results_partial.csv', index=False)

    # save final CSV
    df = pd.DataFrame(results)
    df.to_csv(outdir / 'results.csv', index=False)
    print("Saved results to", outdir / 'results.csv')

    # quick plots (RTF histogram and MCD scatter if present)
    plt.figure(figsize=(6,4))
    plt.bar(df['idx'].astype(str), df['RTF'])
    plt.ylabel('RTF')
    plt.xlabel('sample idx')
    plt.title('RTF per sample')
    plt.tight_layout()
    plt.savefig(outdir / 'rtf_per_sample.png', dpi=200)
    plt.close()

    if 'MCD_dB' in df.columns and df['MCD_dB'].notna().any():
        plt.figure(figsize=(6,4))
        plt.plot(df['idx'], df['MCD_dB'], marker='o')
        plt.ylabel('MCD (dB)')
        plt.xlabel('sample idx')
        plt.title('MCD per sample')
        plt.tight_layout()
        plt.savefig(outdir / 'mcd_per_sample.png', dpi=200)
        plt.close()

    print("Plots saved to", outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', type=str, default='test_sentences.txt', help='one sentence per line')
    parser.add_argument('--outdir', type=str, default='outputs', help='output dir for wavs and csv')
    parser.add_argument('--ground_truth', type=str, default=None, help='dir with reference wavs named 0000.wav etc (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--sr', type=int, default=22050, help='sampling rate for synthesis')
    args = parser.parse_args()
    main(args)
