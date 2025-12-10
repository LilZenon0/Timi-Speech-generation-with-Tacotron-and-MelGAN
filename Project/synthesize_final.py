import argparse, sys, os
import numpy as np
import soundfile as sf

try:
    import torch
except Exception as e:
    print("ERROR: torch not installed in this environment. Install PyTorch and re-run."); raise

# ---------------- helpers ----------------
def hub_load(repo, name):
    return torch.hub.load(repo, name)

def move_model_to_device(model, device):
    """
    Ensure the whole model and common nested submodules are moved to device.
    Returns (ok: bool, info: str).
    """
    try:
        model.to(device)
    except Exception as e:
        print("Warning: model.to(device) raised:", e)

    for attr in ("generator", "model", "netG", "g"):
        if hasattr(model, attr):
            try:
                getattr(model, attr).to(device)
            except Exception as e:
                print(f"Warning: moving submodule {attr} to device raised: {e}")

    try:
        for p in model.parameters():
            if p.device != device:
                p.data = p.data.to(device)
        for name, buf in model.named_buffers():
            if buf.device != device:
                model.register_buffer(name, buf.to(device))
    except Exception as e:
        print("Warning: manual params/buffers move raised:", e)

    try:
        any_param = next(model.parameters())
        param_dev = any_param.device
        return True, f"model param device: {param_dev}"
    except StopIteration:
        return True, "model has no parameters"
    except Exception as e:
        return False, f"error checking params: {e}"

def call_vocoder_safely_and_move(vocoder_obj, mel_tensor, device):
    """
    Move vocoder (and nested modules) to device, then call it safely.
    Priority: inference(mel) -> callable(voc)(mel) -> forward(mel)
    Returns waveform numpy array.
    """
    ok, info = move_model_to_device(vocoder_obj, device)
    print("move_model_to_device:", ok, info)

    try:
        vocoder_obj.eval()
    except Exception:
        pass
    try:
        if hasattr(vocoder_obj, "remove_weight_norm"):
            try:
                vocoder_obj.remove_weight_norm()
            except Exception:
                pass
        if hasattr(vocoder_obj, "generator") and hasattr(vocoder_obj.generator, "remove_weight_norm"):
            try:
                vocoder_obj.generator.remove_weight_norm()
            except Exception:
                pass
    except Exception:
        pass

    try:
        param_dev = next(vocoder_obj.parameters()).device
        if mel_tensor.device != param_dev:
            print("Moving mel tensor to device:", param_dev)
            mel_tensor = mel_tensor.to(param_dev)
    except Exception as e:
        print("Could not inspect model params; moving mel to target device:", device, "err:", e)
        mel_tensor = mel_tensor.to(device)

    with torch.no_grad():
        if hasattr(vocoder_obj, "inference") and callable(getattr(vocoder_obj, "inference")):
            out = vocoder_obj.inference(mel_tensor)
        elif callable(vocoder_obj):
            out = vocoder_obj(mel_tensor)
        elif hasattr(vocoder_obj, "forward") and callable(getattr(vocoder_obj, "forward")):
            out = vocoder_obj.forward(mel_tensor)
        else:
            raise RuntimeError("Vocoder object has no callable inference/forward/__call__ methods. Type: {}".format(type(vocoder_obj)))

    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    out = np.asarray(out).squeeze()
    if out.ndim > 1:
        out = out.reshape(-1)
    return out

def prepare_text_with_utils(utils, text, device):
    seqs, lengths = utils.prepare_input_sequence([text])
    return seqs.to(device), lengths.to(device)

def tacotron2_to_mel(tacotron2, utils, text, device):
    seq, lengths = prepare_text_with_utils(utils, text, device)
    with torch.no_grad():
        try:
            mel, mel_lengths, align = tacotron2.infer(seq, lengths)
        except Exception:
            mel = tacotron2.infer(seq)
    if isinstance(mel, (list, tuple)):
        mel = mel[0]
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    return mel

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Hello world", help="Text to synthesize")
    parser.add_argument("--out", type=str, default="sample_final.wav", help="Output WAV path")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "torch:", torch.__version__)

    # Load Tacotron2 + utils from NVIDIA hub
    try:
        print("Loading Tacotron2 and utils from NVIDIA/DeepLearningExamples (torch.hub)...")
        tacotron2 = hub_load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
        utils = hub_load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    except Exception as e:
        print("ERROR: failed to load NVIDIA tacotron2 or utils via torch.hub:", e)
        sys.exit(1)
    tacotron2 = tacotron2.to(device).eval()
    print("Tacotron2 loaded.")

    # Load vocoder: try melgan then ParallelWaveGAN
    vocoder = None
    try:
        print("Trying seungwonpark/melgan -> 'melgan' ...")
        vocoder = hub_load('seungwonpark/melgan', 'melgan')
        print("Loaded vocoder type:", type(vocoder))
    except Exception as e:
        print("seungwonpark/melgan failed:", e)
        try:
            print("Trying kan-bayashi/ParallelWaveGAN -> 'parallel_wavegan_ljspeech' ...")
            vocoder = hub_load('kan-bayashi/ParallelWaveGAN', 'parallel_wavegan_ljspeech')
            print("Loaded PWG type:", type(vocoder))
        except Exception as e2:
            print("ParallelWaveGAN failed:", e2)
            print("No vocoder available. Exiting.")
            sys.exit(1)

    # debug print about vocoder
    try:
        print("Vocoder repr:", repr(vocoder)[:1000])
        print("Vocoder type:", type(vocoder))
    except Exception:
        pass

    # Generate mel
    mel = tacotron2_to_mel(tacotron2, utils, args.text, device)
    print("Mel shape:", mel.shape)

    # Vocoder -> waveform (ensuring device match)
    wav = call_vocoder_safely_and_move(vocoder, mel, device)

    # normalize and save
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav)) * 0.99
    sf.write(args.out, wav.astype(np.float32), args.sr)
    print("Saved synthesized audio to:", args.out)

if __name__ == "__main__":
    main()