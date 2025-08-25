#!/usr/bin/env python3
"""
keystrokes_pipeline_windows.py

End-to-end pipeline for Windows:
  A) Record from the current microphone while logging keypresses (0-9, a-z)
  B) Isolate keystrokes (timestamp-based when available; else FFT-energy)
  C) Generate 64x64 mel-spectrograms with optional SpecAugment
  D) Modes: 'windows' (alias of 'macbook'), 'macbook', 'phone', 'zoom'
  E) Extras for Windows:
     - 'devices' subcommand to list microphones
     - --device to choose input device index or name
     - --listener {pynput, keyboard} to choose key hook backend

PowerShell examples
-------------------
# 1) LIST MICROPHONES
python keystrokes_pipeline_windows.py devices

# 2) RECORD (until ESC)
python keystrokes_pipeline_windows.py record `
  --out_dir sessions\\win_run1 `
  --sr 44100 `
  --channels 1 `
  --device 1 `
  --listener pynput

# 3) PROCESS a recorded session (uses timestamps if present)
python keystrokes_pipeline_windows.py process `
  --input_dir sessions\\win_run1 `
  --output_dir dataset_win `
  --mode windows `
  --target 25 `
  --segment_len 14400 `
  --sr 44100 `
  --save_labels_csv

# 4) PROCESS a folder of WAVs you already have (one file per key/class)
python keystrokes_pipeline_windows.py process `
  --input_dir raw_wavs `
  --output_dir dataset_phone `
  --mode phone `
  --fft_threshold 0.25 `
  --target 25 `
  --segment_len 14400 `
  --sr 44100 `
  --save_labels_csv

Notes
-----
- On Windows, check Settings > Privacy & security > Microphone to allow access.
- 'keyboard' backend may require running PowerShell "As Administrator".
- If keystrokes.csv exists next to session.wav, timestamp extraction is used.
- Otherwise FFT-energy isolation:
    * phone/windows/macbook: fixed threshold (set --fft_threshold)
    * zoom: adaptive threshold loop to hit exactly --target segments
"""

import argparse
import os
import csv
import re
import json
import time
import queue
import threading
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import find_peaks
from PIL import Image

from collections import Counter

ALLOWED_KEYS = "0123456789abcdefghijklmnopqrstuvwxyz"


# For recording & key logging
try:
    import sounddevice as sd
except Exception:
    sd = None

# Two keyboard listener options: pynput (cross-platform), keyboard (Windows-native)
try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

try:
    import keyboard as kbd  # pip install keyboard
except Exception:
    kbd = None

VALID_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


# -----------------------
# Utility / Mode mapping
# -----------------------
def normalize_mode(mode: str) -> str:
    """
    Map synonyms so 'windows', 'pc', 'laptop' behave like 'macbook' mode.
    """
    if mode.lower() in ("windows", "pc", "laptop"):
        return "macbook"
    return mode


def ensure_windows_warning():
    if platform.system().lower() != "windows":
        print("[INFO] You're not on Windows. This script is Windows-friendly but still cross-platform.")


def ensure_unique_dir(path: Path) -> Path:
    """
    If path exists and is non-empty, create an auto-suffixed directory:
    <name>_001, <name>_002, ...
    """
    if (not path.exists()) or (path.exists() and not any(path.iterdir())):
        return path
    base = path
    for i in range(1, 1000):
        cand = base.with_name(base.name + f"_{i:03d}")
        if not cand.exists():
            print(f"[SAFE] Output dir exists and is non-empty. Using: {cand}")
            return cand
    raise SystemExit(f"Could not find unique directory name based on {base}")


def next_index_and_width(label_dir: Path, label: str) -> Tuple[int, int]:
    """
    Scan '{label}_NNN.png' files and return (next_index, pad_width).
    - If files exist, preserves the max width seen (e.g., 3 for NNN).
    - Defaults to width=3 when none exist.
    """
    max_idx = -1
    max_width = 3
    for p in label_dir.glob(f"{label}_*.png"):
        m = re.match(rf"^{re.escape(label)}_(\d+)\.png$", p.name)
        if m:
            s = m.group(1)
            max_width = max(max_width, len(s))
            try:
                max_idx = max(max_idx, int(s))
            except ValueError:
                pass
    return max_idx + 1, max_width


def get_labels_writer(out_dir: Path, append: bool):
    """
    Open labels.csv in append-safe mode. Writes header if file is new.
    Returns (writer, file_handle). Remember to close the file_handle.
    """
    path = out_dir / "labels.csv"
    exists = path.exists()
    mode = "a" if (append and exists) else "w"
    f = open(path, mode, newline="")
    w = csv.writer(f)
    if not exists or mode == "w":
        w.writerow(["path", "label"])
    return w, f


# -----------------------
# I/O helpers
# -----------------------
def list_audio_files(input_dir: Path) -> List[Path]:
    # Support nested per-class folders (e.g., A/1_foo.wav) and flat files
    files = []
    for p in sorted(input_dir.rglob("*")):
        if p.suffix.lower() in VALID_EXTS and p.is_file():
            files.append(p)
    return files


def label_from_filename(path: Path, input_root: Path) -> str:
    """
    Prefer parent folder name if it looks like a key (0-9, a-z), else use stem.
    """
    parent = path.parent
    cand = parent.name.lower()
    if parent != input_root and len(cand) == 1 and (("a" <= cand <= "z") or ("0" <= cand <= "9")):
        return cand
    return path.stem.lower()


def load_keystroke_csv(csv_path: Path) -> List[Tuple[float, str]]:
    stamps = []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        # Accept either with or without a header
        if header:
            for row in r:
                try:
                    t = float(row[0])
                    k = row[1].strip().lower()
                    stamps.append((t, k))
                except Exception:
                    continue
        else:
            f.seek(0)
            for row in csv.reader(f):
                try:
                    t = float(row[0])
                    k = row[1].strip().lower()
                    stamps.append((t, k))
                except Exception:
                    continue
    return stamps


# --------------------------------
# Energy envelope via STFT (FFT)
# --------------------------------
def energy_envelope_fft(y: np.ndarray, sr: int, win_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=win_length, hop_length=hop_length, win_length=win_length))
    energy = S.sum(axis=0)
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    # light smoothing
    if energy.size >= 3:
        kernel = np.ones(3) / 3.0
        energy = np.convolve(energy, kernel, mode="same")
    return energy


def detect_onsets_phone_or_mac(energy: np.ndarray, threshold: float, distance_frames: int = 5) -> np.ndarray:
    peaks, _ = find_peaks(energy, height=threshold, distance=distance_frames)
    return peaks


def detect_onsets_zoom_adaptive(energy: np.ndarray,
                                target: int = 25,
                                init_threshold: float = 0.2,
                                distance_frames: int = 5,
                                step: float = 0.05,
                                min_threshold: float = 0.01,
                                max_threshold: float = 0.99,
                                patience: int = 200) -> Tuple[np.ndarray, float]:
    t = init_threshold
    s = step
    best = (np.array([], dtype=int), t)
    best_diff = 10 ** 9

    for _ in range(patience):
        t = float(np.clip(t, min_threshold, max_threshold))
        peaks, _ = find_peaks(energy, height=t, distance=distance_frames)
        diff = abs(len(peaks) - target)
        if diff < best_diff:
            best = (peaks, t)
            best_diff = diff
            if diff == 0:
                return peaks, t
        if len(peaks) < target:
            t -= s
        else:
            t += s
        s *= 0.99
    return best[0], best[1]


# ----------------------------------------------------------
# Extract fixed-length segments (from frames or timestamps)
# ----------------------------------------------------------
def extract_segments_from_frames(y: np.ndarray,
                                 onsets_frames: np.ndarray,
                                 stft_hop: int,
                                 segment_len: int) -> List[np.ndarray]:
    out = []
    for f in onsets_frames:
        onset_sample = int(f * stft_hop)
        start = onset_sample - segment_len // 2
        end = start + segment_len
        seg = pad_safe_slice(y, start, end, segment_len)
        if len(seg) == segment_len:
            out.append(seg.astype(np.float32))
    return out


def extract_segments_from_timestamps(y: np.ndarray,
                                     sr: int,
                                     stamps_sec: List[float],
                                     segment_len: int) -> List[np.ndarray]:
    out = []
    for t in stamps_sec:
        center = int(round(t * sr))
        start = center - segment_len // 2
        end = start + segment_len
        seg = pad_safe_slice(y, start, end, segment_len)
        if len(seg) == segment_len:
            out.append(seg.astype(np.float32))
    return out


def pad_safe_slice(y: np.ndarray, start: int, end: int, segment_len: int) -> np.ndarray:
    if start < 0:
        pad_left = -start
        seg = np.pad(y[:max(end, 0)], (pad_left, 0))[:segment_len]
    elif end > len(y):
        pad_right = end - len(y)
        seg = np.pad(y[start:], (0, pad_right))[:segment_len]
    else:
        seg = y[start:end]
    return seg


# ---------------------------------------------------
# Time-shift augmentation & mel-spectrogram creation
# ---------------------------------------------------
def time_shift(y: np.ndarray, max_frac: float = 0.4, mode: str = "zero") -> np.ndarray:
    n = len(y)
    max_shift = int(max_frac * n)
    if max_shift <= 0:
        return y.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return y.copy()
    out = np.zeros_like(y)
    if mode == "zero":
        if shift > 0:
            out[shift:] = y[:n - shift]
        else:
            out[:n + shift] = y[-shift:]
        return out
    else:
        return np.roll(y, shift)


def compute_mel(y: np.ndarray, sr: int, n_mels: int, win_length: int, hop_length: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=win_length, hop_length=hop_length, win_length=win_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def resize_to_square(img: np.ndarray, target: int = 64) -> np.ndarray:
    mn, mx = float(np.min(img)), float(np.max(img))
    norm = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
    pil = Image.fromarray((norm * 255.0).astype(np.uint8))
    pil = pil.resize((target, target), resample=Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32) / 255.0
    return out


def spec_augment(spec: np.ndarray, time_mask_frac: float = 0.10, freq_mask_frac: float = 0.10) -> np.ndarray:
    H, W = spec.shape
    mean_val = float(spec.mean())
    # time
    t_width = max(1, int(W * time_mask_frac))
    t0 = np.random.randint(0, max(1, W - t_width + 1))
    spec[:, t0:t0 + t_width] = mean_val
    # freq
    f_height = max(1, int(H * freq_mask_frac))
    f0 = np.random.randint(0, max(1, H - f_height + 1))
    spec[f0:f0 + f_height, :] = mean_val
    return spec


def save_png(spec64: np.ndarray, out_png: Path):
    img8 = (np.clip(spec64, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img8).save(out_png)


# -------------------------------
# RECORDING with keystroke logging
# -------------------------------
def list_devices():
    if sd is None:
        print("sounddevice not available. Install with: pip install sounddevice")
        return
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []
    devs = sd.query_devices()
    print("Available audio devices:")
    for idx, d in enumerate(devs):
        host = d.get("hostapi", None)
        host_name = f"{hostapis[host]['name']}" if host is not None and host < len(hostapis) else "UnknownHostAPI"
        print(f"[{idx:>2}] {d['name']} | inputs={d['max_input_channels']} | host={host_name}")


def record_session(out_dir: Path,
                   sr: int = 44100,
                   channels: int = 1,
                   device: Optional[Union[int, str]] = None,
                   listener_backend: str = "pynput"):
    if sd is None:
        raise SystemExit("sounddevice not available. Install with: pip install sounddevice")
    if listener_backend == "pynput" and pynput_keyboard is None:
        raise SystemExit("pynput not available. Install with: pip install pynput")
    if listener_backend == "keyboard" and kbd is None:
        raise SystemExit("keyboard not available. Install with: pip install keyboard (run as Admin if needed)")

    ensure_windows_warning()

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "session.wav"
    csv_path = out_dir / "keystrokes.csv"
    meta_path = out_dir / "meta.json"
    counts_path = out_dir / "counts.json"

    print("Recording… Press ESC to stop. Only logs 0–9 and a–z.")
    print(f"Saving audio to: {wav_path}")
    print(f"Saving keystrokes to: {csv_path}")
    if device is not None:
        print(f"Using input device: {device}")
    print("\nLive key counts will update below. Press ESC to finish.\n")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    stop_flag = threading.Event()
    start_time = time.time()
    keystrokes: List[Tuple[float, str]] = []
    counts = Counter()
    last_line_len = 0

    def render_counts(c: Counter) -> str:
        # show only keys that have counts, in 0-9 then a-z order
        parts = [f"{k}:{c[k]}" for k in ALLOWED_KEYS if c[k]]
        return " ".join(parts) if parts else "(none)"

    def print_live(line: str):
        nonlocal last_line_len
        pad = max(0, last_line_len - len(line))
        print("\r" + line + (" " * pad), end="", flush=True)
        last_line_len = len(line)

    # --- keyboard listeners
    def on_keypress_letter_digit(ch: Optional[str]):
        if not ch:
            return
        s = ch.lower()
        if ("a" <= s <= "z") or ("0" <= s <= "9"):
            t = time.time() - start_time
            keystrokes.append((t, s))
            counts[s] += 1
            live = f"[{t:6.2f}s] key='{s}' total={sum(counts.values())} | {render_counts(counts)}"
            print_live(live)

    def start_keyboard_listener():
        if listener_backend == "pynput":
            def on_press(key):
                try:
                    s = key.char
                except AttributeError:
                    s = None
                if key == pynput_keyboard.Key.esc:
                    stop_flag.set()
                    return False
                on_keypress_letter_digit(s)
            listener = pynput_keyboard.Listener(on_press=on_press)
            listener.start()
            return listener
        else:  # keyboard lib
            def handler(event):
                if event.event_type == "down":
                    if event.name == "esc":
                        stop_flag.set()
                    elif len(event.name) == 1:
                        on_keypress_letter_digit(event.name)
            kbd.hook(handler)
            return "keyboard"

    k_listener = start_keyboard_listener()

    # --- audio callback
    def audio_cb(indata, frames, time_info, status):
        if status:
            # print(status)  # uncomment for troubleshooting dropouts
            pass
        audio_q.put(indata.copy())

    stream = sd.InputStream(samplerate=sr, channels=channels, callback=audio_cb, device=device)
    stream.start()

    # collect until ESC
    try:
        while not stop_flag.is_set():
            time.sleep(0.05)
    finally:
        stream.stop()
        stream.close()
        # stop listeners
        if listener_backend == "pynput" and k_listener is not None:
            try:
                k_listener.stop()
            except Exception:
                pass
        elif listener_backend == "keyboard":
            try:
                kbd.unhook_all()
            except Exception:
                pass

    # move to a fresh line after live printing
    print()

    # gather audio
    chunks = []
    while not audio_q.empty():
        chunks.append(audio_q.get())
    if not chunks:
        print("[WARN] No audio captured.")
        return

    audio = np.concatenate(chunks, axis=0).squeeze()
    sf.write(str(wav_path), audio, sr)

    # write keystrokes
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "key"])
        for t, k in keystrokes:
            w.writerow([f"{t:.6f}", k])

    # meta
    with open(meta_path, "w") as f:
        json.dump(
            {"sr": sr, "channels": channels, "start_epoch": int(start_time), "device": device,
             "listener_backend": listener_backend},
            f, indent=2
        )

    # counts summary (also save to JSON)
    counts_dict = {k: counts[k] for k in ALLOWED_KEYS if counts[k]}
    with open(counts_path, "w") as f:
        json.dump({"total": sum(counts.values()), "counts": counts_dict}, f, indent=2)

    print(f"\nKey counts (total={sum(counts.values())}): {render_counts(counts)}")
    print(f"Saved counts to: {counts_path}")
    print(f"Done. Duration: {len(audio)/sr:.2f}s, Keystrokes logged: {len(keystrokes)}")
    print("Next: run the 'process' subcommand on this folder.")


# ---------------
# PROCESSING
# ---------------
def process_path_folder(folder: Path,
                        out_dir: Path,
                        mode: str,
                        target: int,
                        segment_len: int,
                        sr_target: int,
                        fft_threshold: float,
                        stft_win_energy: int,
                        stft_hop_energy: int,
                        augment_time_shift: bool,
                        specaugment: bool,
                        seed: int,
                        write_labels_csv: bool):
    """
    If folder contains a session.wav + keystrokes.csv -> timestamp segmentation per key.
    Else if folder contains many WAVs (one per key) -> file-wise processing.
    """
    # Make output dir safe (auto-suffix if non-empty)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare labels writer (append-safe)
    labels_writer, labels_fh = (None, None)
    if write_labels_csv:
        labels_writer, labels_fh = get_labels_writer(out_dir, append=True)  # append to existing by default

    norm_mode = normalize_mode(mode)

    session_wav = folder / "session.wav"
    ks_csv = folder / "keystrokes.csv"

    mel_hop = 255 if norm_mode == "macbook" else 500

    if session_wav.exists() and ks_csv.exists():
        # 1) load session
        y, sr = sf.read(str(session_wav))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        stamps = load_keystroke_csv(ks_csv)
        # group timestamps by key
        by_key: Dict[str, List[float]] = {}
        for t, k in stamps:
            by_key.setdefault(k, []).append(float(t))

        # deterministic order by key
        for k in sorted(by_key.keys()):
            label = k.lower()
            label_dir = out_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            # take first 'target' stamps for consistency
            tlist = by_key[label][:target]
            segs = extract_segments_from_timestamps(y, sr, tlist, segment_len)

            rng = np.random.default_rng(seed + hash(label) % 10000)
            start_idx, pad_w = next_index_and_width(label_dir, label)

            for i, seg in enumerate(segs):
                seg_proc = seg
                if augment_time_shift and rng.random() < 0.5:
                    seg_proc = time_shift(seg_proc, max_frac=0.4, mode="zero")
                mel = compute_mel(seg_proc, sr, n_mels=64, win_length=1024, hop_length=mel_hop)
                spec64 = resize_to_square(mel, target=64)
                if specaugment and rng.random() < 0.5:
                    spec64 = spec_augment(spec64, 0.10, 0.10)

                out_png = label_dir / f"{label}_{start_idx + i:0{pad_w}d}.png"
                save_png(spec64, out_png)

                if labels_writer is not None:
                    labels_writer.writerow([str(out_png.relative_to(out_dir)), label])

        if labels_fh is not None:
            labels_fh.close()

        print(f"[session] Wrote spectrograms to {out_dir}")
        return

    # Otherwise: treat every WAV as one label file (supports nested class dirs)
    files = list_audio_files(folder)
    if not files:
        raise SystemExit(f"No audio found in {folder} (neither session.wav nor individual WAVs).")

    for idx, f in enumerate(files):
        label = label_from_filename(f, folder)
        rows = process_single_wav(
            in_wav=f,
            input_root=folder,
            out_dir=out_dir,
            label=label,
            mode=norm_mode,
            target=target,
            segment_len=segment_len,
            sr_target=sr_target,
            fft_threshold=fft_threshold,
            stft_win_energy=stft_win_energy,
            stft_hop_energy=stft_hop_energy,
            mel_hop=mel_hop,
            augment_time_shift=augment_time_shift,
            specaugment=specaugment,
            seed=seed + idx,
            labels_writer=labels_writer
        )

    if labels_fh is not None:
        labels_fh.close()

    print(f"Wrote spectrograms to {out_dir}")


def process_single_wav(in_wav: Path,
                       input_root: Path,
                       out_dir: Path,
                       label: str,
                       mode: str,
                       target: int,
                       segment_len: int,
                       sr_target: int,
                       fft_threshold: float,
                       stft_win_energy: int,
                       stft_hop_energy: int,
                       mel_hop: int,
                       augment_time_shift: bool,
                       specaugment: bool,
                       seed: int,
                       labels_writer: Optional[csv.writer]) -> List[Tuple[str, str]]:

    y, sr = sf.read(str(in_wav))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    energy = energy_envelope_fft(y, sr, win_length=stft_win_energy, hop_length=stft_hop_energy)

    if mode in ("phone", "macbook"):
        onsets = detect_onsets_phone_or_mac(energy, threshold=fft_threshold, distance_frames=2)
    elif mode == "zoom":
        onsets, _ = detect_onsets_zoom_adaptive(energy, target=target, init_threshold=fft_threshold, distance_frames=2)
    else:
        raise ValueError("mode must be one of: phone, macbook/windows, zoom")

    segs = extract_segments_from_frames(y, onsets, stft_hop_energy, segment_len)
    if len(segs) > target:
        segs = segs[:target]

    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    start_idx, pad_w = next_index_and_width(label_dir, label)

    rows: List[Tuple[str, str]] = []
    for i, seg in enumerate(segs):
        seg_proc = seg
        if augment_time_shift and rng.random() < 0.5:
            seg_proc = time_shift(seg_proc, max_frac=0.4, mode="zero")
        mel = compute_mel(seg_proc, sr, n_mels=64, win_length=1024, hop_length=mel_hop)
        spec64 = resize_to_square(mel, target=64)
        if specaugment and rng.random() < 0.5:
            spec64 = spec_augment(spec64, 0.10, 0.10)

        out_png = label_dir / f"{label}_{start_idx + i:0{pad_w}d}.png"
        save_png(spec64, out_png)

        rel = str(out_png.relative_to(out_dir))
        if labels_writer is not None:
            labels_writer.writerow([rel, label])
        else:
            rows.append((rel, label))

    if len(segs) < target:
        print(f"[WARN] {in_wav.relative_to(input_root)}: only {len(segs)} segments found (< {target}).")

    return rows


# -----------------
# CLI
# -----------------
def main():
    ap = argparse.ArgumentParser(description="Keystroke recording + processing pipeline (Windows-friendly)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # DEVICES
    sub.add_parser("devices", help="List available audio input devices")

    # RECORD
    apr = sub.add_parser("record", help="Record mic and log keypresses (ESC to stop)")
    apr.add_argument("--out_dir", type=str, required=True)
    apr.add_argument("--sr", type=int, default=44100)
    apr.add_argument("--channels", type=int, default=1)
    apr.add_argument("--device", type=str, default=None,
                     help="Input device index or substring name (e.g., 'Microphone (Realtek)')")
    apr.add_argument("--listener", type=str, choices=["pynput", "keyboard"], default="pynput",
                     help="Keyboard listener backend (keyboard may require Admin)")

    # PROCESS
    app = sub.add_parser("process", help="Process a recorded session OR a folder of per-key WAVs")
    app.add_argument("--input_dir", type=str, required=True,
                     help="Folder with session.wav+keystrokes.csv OR WAVs per key (supports nested class dirs)")
    app.add_argument("--output_dir", type=str, required=True)
    app.add_argument("--mode", type=str, choices=["windows", "macbook", "phone", "zoom"], default="windows")
    app.add_argument("--target", type=int, default=25)
    app.add_argument("--segment_len", type=int, default=14400)
    app.add_argument("--sr", type=int, default=44100)
    app.add_argument("--fft_threshold", type=float, default=0.25,
                     help="Fixed (macbook/windows/phone) or initial (zoom)")
    app.add_argument("--stft_hop_energy", type=int, default=256)
    app.add_argument("--stft_win_energy", type=int, default=1024)
    app.add_argument("--no_time_shift", action="store_true")
    app.add_argument("--no_specaugment", action="store_true")
    app.add_argument("--save_labels_csv", action="store_true")
    app.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    if args.cmd == "devices":
        list_devices()
        return

    if args.cmd == "record":
        # resolve device
        dev: Optional[Union[int, str]] = None
        if args.device is not None:
            # try int first, else pass through as name substring
            try:
                dev = int(args.device)
            except ValueError:
                # find first device whose name contains substring
                if sd is None:
                    raise SystemExit("sounddevice required to search devices.")
                candidates = []
                for idx, d in enumerate(sd.query_devices()):
                    if args.device.lower() in d["name"].lower() and d["max_input_channels"] > 0:
                        candidates.append(idx)
                if not candidates:
                    raise SystemExit(f"No input device matching '{args.device}' found.")
                dev = candidates[0]
                print(f"[INFO] Using device index {dev} for '{args.device}'")
        record_session(
            Path(args.out_dir),
            sr=args.sr,
            channels=args.channels,
            device=dev,
            listener_backend=args.listener
        )
        return

    if args.cmd == "process":
        process_path_folder(
            folder=Path(args.input_dir),
            out_dir=Path(args.output_dir),
            mode=args.mode,
            target=args.target,
            segment_len=args.segment_len,
            sr_target=args.sr,
            fft_threshold=args.fft_threshold,
            stft_win_energy=args.stft_win_energy,
            stft_hop_energy=args.stft_hop_energy,
            augment_time_shift=not args.no_time_shift,
            specaugment=not args.no_specaugment,
            seed=args.seed,
            write_labels_csv=args.save_labels_csv
        )
        return


if __name__ == "__main__":
    main()
