#!/usr/bin/env python3
"""
synthesize_hub_only.py

A robust inference-only Tacotron2 -> MelGAN pipeline with optional mel/waveform plotting.
Usage examples:
  # single text on GPU
  python synthesize_hub_only.py --text "Hello world" --outdir outputs --device cuda

  # sentences file (one sentence per line), with local checkpoints
  python synthesize_hub_only.py --sentences test_sentences.txt --taco_ckpt ./checkpoints/tacotron2_statedict.pt \
       --vocoder_ckpt ./checkpoints/melgan.pt --outdir outputs --device cuda

Notes:
  - If you supply local checkpoint paths (--taco_ckpt, --vocoder_ckpt) the script will prefer them.
  - If not, the script will try to load common implementations from torch.hub (best-effort).
  - The script saves mel numpy, mel spectrogram PNG and waveform PNG and writes wav output.
"""

import argparse
import sys
import os
from pathlib import Path
import time
import warnings
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

# try to import torch
try:
    import torch
except Exception as e:
    print("ERROR: torch not found. Please install PyTorch. (pip install torch)")
    raise e

# ----------------------
# Utility: mel & plotting (no librosa dependency)
# ----------------------
def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10**(mel / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft=1024, n_mels=80, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2 + 1), dtype=np.float32)
    for i in range(1, n_mels+1):
        left, center, right = bins[i-1], bins[i], bins[i+1]
        if center > left:
            fb[i-1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i-1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb

def mel_from_wave(wave, sr, n_fft=1024, hop_length=256, n_mels=80):
    win = get_window("hann", n_fft, fftbins=True)
    f, t, Zxx = stft(wave, fs=sr, window=win, nperseg=n_fft, noverlap=n_fft-hop_length, nfft=n_fft, boundary=None, padded=False)
    S = np.abs(Zxx)**2
    fb = mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels)
    mel = np.dot(fb, S)
    return mel, t

def save_mel_and_plots(out_dir, basename, mel_tensor=None, audio=None, sr=22050,
                       n_fft=1024, hop_length=256, n_mels=80):
    """
    Save mel numpy array and produce two PNGs:
      - {basename}_melspec.png  (mel in dB)
      - {basename}_waveform.png (if audio provided)
    Accepts mel_tensor as torch.Tensor or numpy array. Returns dict of paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert mel_tensor -> numpy (shape n_mels x T)
    mel_np = None
    if mel_tensor is not None:
        try:
            if isinstance(mel_tensor, torch.Tensor):
                mel_np = mel_tensor.detach().cpu().numpy()
            else:
                mel_np = np.array(mel_tensor)
        except Exception:
            mel_np = np.array(mel_tensor)
        if mel_np.ndim == 3 and mel_np.shape[0] == 1:
            mel_np = mel_np[0]
        elif mel_np.ndim == 3 and mel_np.shape[0] != 1:
            mel_np = mel_np[0]

    if mel_np is None:
        if audio is None:
            raise ValueError("Either mel_tensor or audio must be provided.")
        mel_np, _ = mel_from_wave(audio, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Save raw mel
    mel_path = out_dir / f"{basename}_mel.npy"
    np.save(str(mel_path), mel_np.astype(np.float32))

    # Convert to dB for plotting
    amin = 1e-10
    mel_db = 10.0 * np.log10(np.maximum(mel_np, amin))
    mel_db = mel_db - mel_db.max()

    # Plot mel spectrogram
    plt.figure(figsize=(8,4))
    plt.imshow(mel_db, origin='lower', aspect='auto', interpolation='nearest')
    plt.xlabel("Frame")
    plt.ylabel("Mel bin")
    plt.title(f"Mel-spectrogram (dB): {basename}")
    plt.colorbar(format="%+2.0f dB")
    mel_png = out_dir / f"{basename}_melspec.png"
    plt.tight_layout()
    plt.savefig(str(mel_png), dpi=200)
    plt.close()

    wave_png = None
    if audio is not None:
        # normalize audio for plotting
        if np.max(np.abs(audio)) > 0:
            audio_plot = audio / max(0.9999, np.max(np.abs(audio)))
        else:
            audio_plot = audio
        t = np.arange(len(audio_plot)) / float(sr)
        plt.figure(figsize=(8,3))
        plt.plot(t, audio_plot, linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform: {basename}")
        wave_png = out_dir / f"{basename}_waveform.png"
        plt.tight_layout()
        plt.savefig(str(wave_png), dpi=200)
        plt.close()

        # save wav as 16-bit PCM
        out_wav = out_dir / f"{basename}_synth.wav"
        try:
            sf.write(str(out_wav), audio.astype(np.float32), sr, subtype='PCM_16')
        except Exception:
            sf.write(str(out_wav), audio.astype(np.float32), sr)

    return {"mel_npy": str(mel_path), "mel_png": str(mel_png), "wave_png": str(wave_png) if wave_png else None, "wav": str(out_wav) if audio is not None else None}

# ----------------------
# Robust vocoder call helper
# ----------------------
def call_vocoder(vocoder, mel_tensor, device):
    """
    Robustly call a vocoder-like object loaded from hub or local checkpoint.
    Try .inference(), then call(), then forward().
    mel_tensor expected shape [1, n_mels, T] or torch tensor on device.
    Returns numpy waveform (float32, -1..1).
    """
    # ensure mel on CPU or device matching vocoder params
    try:
        # some vocoders expect CPU numpy; others expect cuda tensors
        # we'll first ensure mel is a torch Tensor
        if not isinstance(mel_tensor, torch.Tensor):
            mel = torch.tensor(mel_tensor, device=device)
        else:
            mel = mel_tensor.to(device)
    except Exception:
        mel = mel_tensor

    # Try common invocation patterns
    try:
        if hasattr(vocoder, 'inference') and callable(vocoder.inference):
            out = vocoder.inference(mel)
        elif callable(vocoder):
            out = vocoder(mel)
        elif hasattr(vocoder, 'forward') and callable(vocoder.forward):
            out = vocoder.forward(mel)
        else:
            raise RuntimeError("Vocoder object is not callable and has no inference/forward.")
    except Exception as e:
        # try to move vocoder to device then call again
        try:
            vocoder.to(device)
            if hasattr(vocoder, 'inference') and callable(vocoder.inference):
                out = vocoder.inference(mel)
            elif callable(vocoder):
                out = vocoder(mel)
            elif hasattr(vocoder, 'forward') and callable(vocoder.forward):
                out = vocoder.forward(mel)
            else:
                raise
        except Exception as ex:
            raise RuntimeError(f"Vocoder call failed: {ex}") from e

    # normalize output to numpy 1D float32
    if isinstance(out, torch.Tensor):
        audio = out.detach().cpu().numpy()
    else:
        audio = np.array(out)

    # if shape is (1, N) or (N,1) try to flatten
    if audio.ndim > 1 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim > 1 and audio.shape[-1] == 1:
        audio = audio.reshape(-1)

    # scale if necessary (if vocoder outputs in [-1,1] already, ok)
    # ensure float32
    audio = audio.astype(np.float32)

    maxv = np.max(np.abs(audio)) if audio.size>0 else 1.0
    if maxv > 1.0:
        audio = audio / maxv

    return audio

# ----------------------
# Try to load Tacotron2 from hub (best-effort)
# ----------------------
def load_tacotron2_from_hub(device):
    """
    Best-effort hub loader for Tacotron2. Returns model object or raises.
    """
    print("[INFO] Attempting to load Tacotron2 from torch.hub ...")
    try:
        # Common NVIDIA hub: 'NVIDIA/DeepLearningExamples:torchhub'
        # The exact API name may vary between torch versions. Try common names.
        tacotron = None
        try:
            tacotron = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
        except Exception:
            try:
                tacotron = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'tacotron2')
            except Exception:
                pass

        if tacotron is None:
            # Try other repositories commonly used
            tacotron = torch.hub.load('bkachlik/tacotron2', 'tacotron2', verbose=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load Tacotron2 from torch.hub: {e}")

    tacotron = tacotron.to(device)
    tacotron.eval()
    return tacotron

# ----------------------
# Try to load MelGAN from hub (best-effort)
# ----------------------
def load_melgan_from_hub(device):
    """
    Best-effort hub loader for MelGAN (seungwonpark or other).
    """
    print("[INFO] Attempting to load MelGAN from torch.hub ...")
    try:
        # seungwonpark melgan repo often provides inference wrapper
        voc = torch.hub.load('seungwonpark/melgan', 'melgan')
    except Exception:
        try:
            # Alternative: 'descriptinc/melgan-neurips' or other
            voc = torch.hub.load('seungwonpark/melgan', 'melgan', verbose=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load MelGAN from torch.hub: {e}")

    voc = voc.to(device)
    voc.eval()
    return voc

# ----------------------
# Main synthesis loop
# ----------------------
def synthesize_list(sentences, outdir, device, taco_ckpt=None, voc_ckpt=None, sr=22050):
    outdir = Path(outdir)
    figs_dir = outdir / "figs"
    aud_dir = outdir / "wavs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    aud_dir.mkdir(parents=True, exist_ok=True)

    # Load models (prefer local checkpoints)
    tacotron = None
    vocoder = None

    # 1) Try local checkpoints if provided
    if taco_ckpt:
        if not Path(taco_ckpt).exists():
            raise FileNotFoundError(f"Tacotron2 checkpoint not found: {taco_ckpt}")
        print(f"[INFO] Loading Tacotron2 checkpoint from {taco_ckpt}")
        # Many tacotron checkpoints are state_dicts; implement safe loader attempts
        try:
            # try typical class from nvidia tacotron2 repo if available
            tac = torch.load(taco_ckpt, map_location=device)
            # If it's a full model, return it; else user should adapt
            if isinstance(tac, dict) and ('state_dict' in tac or 'checkpoint' in tac):
                print("[WARN] Loaded dict checkpoint; you may need to wrap into a Tacotron2 model class.")
                # fall back to hub loader for model class and then load state dict
                try:
                    tac_model = load_tacotron2_from_hub(device)
                    state_dict = tac.get('state_dict', tac)
                    tac_model.load_state_dict(state_dict)
                    tacotron = tac_model
                except Exception as e:
                    print("[ERROR] Could not instantiate Tacotron2 model class to load state dict:", e)
            else:
                tacotron = tac
        except Exception as e:
            print("[WARN] direct torch.load failed for tacotron ckpt:", e)

    if not tacotron:
        try:
            tacotron = load_tacotron2_from_hub(device)
            print("[INFO] Tacotron2 loaded from hub.")
        except Exception as e:
            print("[ERROR] Tacotron2 load failed:", e)
            print("Please provide --taco_ckpt path or ensure torch.hub can access a Tacotron2 impl.")
            raise

    # vocoder
    if voc_ckpt:
        if not Path(voc_ckpt).exists():
            raise FileNotFoundError(f"Vocoder checkpoint not found: {voc_ckpt}")
        print(f"[INFO] Loading vocoder from {voc_ckpt}")
        try:
            voc = torch.load(voc_ckpt, map_location=device)
            vocoder = voc
        except Exception as e:
            print("[WARN] direct torch.load of vocoder ckpt failed:", e)

    if not vocoder:
        try:
            vocoder = load_melgan_from_hub(device)
            print("[INFO] MelGAN loaded from hub.")
        except Exception as e:
            print("[ERROR] MelGAN load failed:", e)
            print("Please provide --vocoder_ckpt path or ensure torch.hub can access a MelGAN impl.")
            raise

    # Ensure eval mode and device
    try:
        tacotron.to(device).eval()
    except Exception:
        tacotron.eval()
    try:
        vocoder.to(device).eval()
    except Exception:
        vocoder.eval()

    results = []
    counter = 0
    for i, txt in enumerate(sentences):
        txt = txt.strip()
        if len(txt) == 0:
            continue
        baseid = f"{i:04d}"
        print(f"\n[INFO] Synthesizing sample {baseid}: \"{txt[:80]}...\"")

        # ---------- Text -> mel (many tacotron hub APIs differ)
        mel = None
        start_t = time.time()
        try:
            # try common tacotron invocation patterns
            if hasattr(tacotron, 'infer') and callable(tacotron.infer):
                tac_out = tacotron.infer(txt)
                # tac_out may be (mel, mel_postnet, align) or a Tensor
                if isinstance(tac_out, (list, tuple)) and len(tac_out) > 0:
                    mel = tac_out[0]
                else:
                    mel = tac_out
            elif hasattr(tacotron, 'forward') and callable(tacotron.forward):
                # some torch.hub tacotron functions expect integer-coded text; try a simple wrapper if available
                try:
                    tac_out = tacotron.forward(txt)
                    mel = tac_out
                except Exception:
                    # some tacotron hub implementations provide a helper 'sequence_to_text' or 'text_to_sequence'
                    if hasattr(tacotron, 'sequence_to_text'):
                        seq = tacotron.sequence_to_text(txt)
                        tac_out = tacotron.forward(seq)
                        mel = tac_out
                    else:
                        # fallback: try calling tacotron as a callable
                        tac_out = tacotron(txt)
                        mel = tac_out
            elif callable(tacotron):
                tac_out = tacotron(txt)
                mel = tac_out
            else:
                raise RuntimeError("Unknown Tacotron2 invocation pattern")
        except Exception as e:
            print("[ERROR] Tacotron invocation failed:", e)
            raise

        # Normalize mel to numpy tensor shape (1, n_mels, T) or (n_mels, T)
        mel_np = None
        try:
            if isinstance(mel, torch.Tensor):
                mel_np = mel.detach().cpu().numpy()
            else:
                mel_np = np.array(mel)
        except Exception:
            mel_np = np.array(mel)

        # Ensure shape is [1, n_mels, T] for vocoder if needed
        if mel_np.ndim == 2:
            mel_np = mel_np[np.newaxis, :, :]
        if mel_np.ndim == 3 and mel_np.shape[0] != 1:
            # keep first batch
            mel_np = mel_np[0:1, :, :]

        tac_time = time.time() - start_t

        # save mel plot (without waveform yet)
        try:
            save_mel_and_plots(figs_dir, baseid, mel_tensor=mel_np, audio=None, sr=sr)
            print(f"[INFO] Saved mel plot for {baseid}")
        except Exception as e:
            print("[WARN] could not save mel plot:", e)

        # ---------- Vocoder: mel -> waveform
        start_v = time.time()
        try:
            audio = call_vocoder(vocoder, torch.tensor(mel_np, device=device), device)
        except Exception as e:
            print("[ERROR] Vocoder failed:", e)
            raise
        v_time = time.time() - start_v

        # Save waveform + recomputed mel/wave plot + wav file
        try:
            res = save_mel_and_plots(figs_dir, baseid, mel_tensor=mel_np, audio=audio, sr=sr)
            # copy wav to aud_dir
            if res.get("wav"):
                target_wav = aud_dir / f"{baseid}.wav"
                os.replace(res["wav"], str(target_wav))
                res["wav"] = str(target_wav)
            print(f"[INFO] Saved waveform and wav for {baseid}: {res.get('wav')}")
        except Exception as e:
            print("[WARN] could not save waveform/mel plots:", e)

        total_time = tac_time + v_time
        rtf = total_time / (len(audio) / float(sr)) if len(audio) > 0 else float('inf')

        results.append({
            "idx": i,
            "text": txt,
            "duration_s": round(len(audio)/float(sr), 3),
            "tacotron_s": round(tac_time, 3),
            "melgan_s": round(v_time, 3),
            "rtf": round(rtf, 3),
            "wav": res.get("wav")
        })

        print(f"[INFO] Sample {baseid} times: tacotron={tac_time:.3f}s, vocoder={v_time:.3f}s, total={total_time:.3f}s, RTF={rtf:.3f}")
        counter += 1

    # save CSV summary
    import csv
    csv_path = outdir / "synthesis_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["idx","text","duration_s","tacotron_s","melgan_s","rtf","wav"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[INFO] Wrote summary CSV: {csv_path}")

    return results

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Tacotron2 + MelGAN inference + mel/wave plotting utility")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text to synthesize (enclose in quotes)")
    group.add_argument("--sentences", type=str, help="Path to file with sentences, one per line")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save WAV/plots")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    p.add_argument("--taco_ckpt", type=str, default=None, help="Optional local Tacotron2 checkpoint path")
    p.add_argument("--vocoder_ckpt", type=str, default=None, help="Optional local vocoder checkpoint path")
    p.add_argument("--sr", type=int, default=22050, help="Sampling rate for waveform output")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    texts = []
    if args.text:
        texts = [args.text]
    else:
        # read sentences file
        if not Path(args.sentences).exists():
            print(f"[ERROR] sentences file not found: {args.sentences}")
            sys.exit(1)
        with open(args.sentences, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]

    print(f"[INFO] Device: {device}, samples: {len(texts)}, outdir: {args.outdir}")
    synthesize_list(texts, args.outdir, device, taco_ckpt=args.taco_ckpt, voc_ckpt=args.vocoder_ckpt, sr=args.sr)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
