import sys
import os
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# ── 1. Restrict GPU Visibility ─────────────────────────────────────────────
# This MUST be set before importing torch or voxcpm.
# It forces the script to only use physical GPUs 4 and 5.
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# Avoid memory spikes during tensor materialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import gc
import torch
import soundfile as sf
from voxcpm import VoxCPM

# ── Paths ──────────────────────────────────────────────────────────────────
BLIND_DATA_DIR = Path("/Data/deepakkumar/ahtisam/clone/iwslt_blind")
REF_AUDIO_DIR  = BLIND_DATA_DIR / "reference_audio" / "crop_ref_audio"
REF_TEXT_DIR   = BLIND_DATA_DIR / "reference_texts"  / "crop_ref_texts"
OUTPUT_DIR     = BLIND_DATA_DIR / "clone_audio" / "voxcpm2-tts"

TARGET_LANGS = {
    "ar": BLIND_DATA_DIR / "text" / "arabic.txt",
    "fr": BLIND_DATA_DIR / "text" / "french.txt",
    "zh": BLIND_DATA_DIR / "text" / "chinese.txt",
}

# ── Load model ─────────────────────────────────────────────────────────────
print("[INFO] Loading VoxCPM 2.0 onto empty GPU...")
# optimize=False prevents Dynamo compiler crashes.
# enable_denoiser=False saves VRAM so it fits on an 11GB RTX 2080 Ti.
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
print("[INFO] Model loaded successfully.")

# ── Load target language lines ─────────────────────────────────────────────
target_lines = {}
for lang_code, path in TARGET_LANGS.items():
    if not path.exists():
        print(f"[WARN] Text file not found for {lang_code}: {path}")
        continue
    lines = [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    target_lines[lang_code] = lines
    print(f"[INFO] {lang_code}: {len(lines)} lines")

# ── Load speakers ──────────────────────────────────────────────────────────
speaker_files = sorted(REF_AUDIO_DIR.glob("*.wav"))
print(f"[INFO] Found {len(speaker_files)} speakers.")

total    = sum(len(lines) for lines in target_lines.values()) * len(speaker_files)
done     = 0
skipped  = 0
errors   = 0

# ── Generation loop ────────────────────────────────────────────────────────
for spk_path in speaker_files:
    spk_id   = spk_path.stem
    txt_path = REF_TEXT_DIR / f"{spk_id}.txt"

    if not txt_path.exists():
        print(f"[WARN] No transcript for {spk_id}, skipping speaker.")
        skipped += sum(len(lines) for lines in target_lines.values())
        continue

    ref_text  = " ".join(
        l.strip()
        for l in txt_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    )
    ref_audio_path = str(spk_path)
    print(f"\n[INFO] Speaker: {spk_id}  (ref_text: {len(ref_text)} chars)")

    for lang_code, lines in target_lines.items():
        out_lang_dir = OUTPUT_DIR / lang_code
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        for line_idx, target_text in enumerate(lines):
            out_path = out_lang_dir / f"{spk_id}_line{line_idx + 1:03d}.wav"

            if out_path.exists():
                print(f"  [SKIP] {out_path.name}")
                skipped += 1
                done    += 1
                continue

            print(f"  [{lang_code}] line {line_idx + 1}/{len(lines)} → {out_path.name}")

            wav = None

            try:
                # ── Inference (VoxCPM 2.0 Hi-Fi Cloning) ──
                wav = model.generate(
                    text=target_text,
                    normalize=True,
                    prompt_wav_path=ref_audio_path,       
                    prompt_text=ref_text,                 
                    reference_wav_path=ref_audio_path,    
                    cfg_value=2.0,
                    inference_timesteps=10
                )

                if wav is not None:
                    sf.write(str(out_path), wav, model.tts_model.sample_rate)
                    print(f"  [OK] Saved → {out_path.name} (sr={model.tts_model.sample_rate})")
                    done += 1
                else:
                    print(f"  [ERROR] Model returned None for {spk_id} {lang_code} line {line_idx + 1}")
                    errors += 1

            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                errors += 1

            except Exception as e:
                print(f"  [ERROR] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                errors += 1

            finally:
                # Aggressive cleanup to keep VRAM flat over long runs
                try:
                    del wav
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

            progress = f"{done}/{total}"
            print(f"  [PROGRESS] {progress}  errors={errors}  skipped={skipped}")

print(f"\n{'='*60}")
print(f"[DONE] Total={total}  Done={done}  Errors={errors}  Skipped={skipped}")
print(f"Output → {OUTPUT_DIR.resolve()}")