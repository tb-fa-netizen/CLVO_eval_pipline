import sys
import os
from pathlib import Path
import gc


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# ── Ensure CosyVoice paths are loaded properly using absolute paths ────────
COSYVOICE_REPO_PATH = "/Data/deepakkumar/ahtisam/cosy-voice/CosyVoice"
sys.path.insert(0, COSYVOICE_REPO_PATH)
sys.path.insert(0, os.path.join(COSYVOICE_REPO_PATH, "third_party", "Matcha-TTS"))

# Avoid memory spikes during tensor materialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice3

# ── Paths ──────────────────────────────────────────────────────────────────
BLIND_DATA_DIR = Path("/Data/deepakkumar/ahtisam/clone/iwslt_blind")
REF_AUDIO_DIR  = BLIND_DATA_DIR / "reference_audio" / "crop_ref_audio"
OUTPUT_DIR     = BLIND_DATA_DIR / "clone_audio" / "cosyvoice-tts"

TARGET_LANGS = {
    "fr": BLIND_DATA_DIR / "text" / "french.txt",
    "zh": BLIND_DATA_DIR / "text" / "chinese.txt",
}
# CosyVoice3 requires this delimiter token in text/prompt_text.
# Keep only the control token to avoid spoken English prompt leakage.
CV3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>[breath] "

# ── Load model ─────────────────────────────────────────────────────────────
# Use the explicit model path provided
model_dir = "/Data/deepakkumar/ahtisam/cosy-voice/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"

if not os.path.exists(model_dir):
    raise FileNotFoundError(f"[ERROR] Model directory not found at {model_dir}. Please verify the path.")

print("[INFO] Loading CosyVoice3 0.5B from local directory...")
cosyvoice = CosyVoice3(model_dir)
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
    spk_id = spk_path.stem
    ref_audio_path = str(spk_path)
    print(f"\n[INFO] Speaker: {spk_id}")

    # CosyVoice cross-lingual frontend also enforces prompt audio <= 30s.
    try:
        info = torchaudio.info(ref_audio_path)
        duration_sec = info.num_frames / info.sample_rate
        if duration_sec > 30.0:
            print(
                f"[WARN] {spk_id} prompt audio is {duration_sec:.1f}s (>30s). "
                "Skipping speaker for CosyVoice cross-lingual."
            )
            skipped += sum(len(lines) for lines in target_lines.values())
            continue
    except Exception as e:
        print(f"[ERROR] Failed to inspect reference audio {spk_id}: {e}")
        errors += sum(len(lines) for lines in target_lines.values())
        continue

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

            chunks = []
            final_audio = None

            try:
                # ── Inference (cross-lingual mode) ──
                # CosyVoice3 requires <|endofprompt|> token in text/prompt_text.
                tts_text = f"{CV3_PROMPT_PREFIX}{target_text}"
                for chunk in cosyvoice.inference_cross_lingual(
                    tts_text,
                    ref_audio_path,
                    stream=False,
                    text_frontend=False,
                ):
                    chunks.append(chunk['tts_speech'])

                if chunks:
                    final_audio = torch.concat(chunks, dim=1)
                    torchaudio.save(str(out_path), final_audio, cosyvoice.sample_rate)
                    print(f"  [OK] Saved → {out_path.name} (sr={cosyvoice.sample_rate})")
                    done += 1
                else:
                    print(f"  [ERROR] Model returned no chunks for {spk_id} {lang_code} line {line_idx + 1}")
                    errors += 1

            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                errors += 1

            except Exception as e:
                print(f"  [ERROR] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                errors += 1

            finally:
                # Aggressive cleanup matching the other pipelines
                try:
                    del chunks, final_audio
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

            progress = f"{done}/{total}"
            print(f"  [PROGRESS] {progress}  errors={errors}  skipped={skipped}")

print(f"\n{'='*60}")
print(f"[DONE] Total={total}  Done={done}  Errors={errors}  Skipped={skipped}")
print(f"Output → {OUTPUT_DIR.resolve()}")
