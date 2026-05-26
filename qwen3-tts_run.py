import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import gc
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# ── Paths ──────────────────────────────────────────────────────────────────
BLIND_DATA_DIR = Path("/Data/deepakkumar/ahtisam/clone/iwslt_blind")
REF_AUDIO_DIR  = BLIND_DATA_DIR / "reference_audio" / "crop_ref_audio1"
REF_TEXT_DIR   = BLIND_DATA_DIR / "reference_texts"  / "crop_ref_texts1"
OUTPUT_DIR     = BLIND_DATA_DIR / "clone_audio" / "qwen3-tts"

TARGET_LANGS = {
    "fr": (BLIND_DATA_DIR / "text" / "french.txt",  "French"),
    "zh": (BLIND_DATA_DIR / "text" / "chinese.txt", "Chinese"),
}
# clone/iwslt_blind/text/french.txt
# ── Model config ───────────────────────────────────────────────────────────
PRETRAINED = "/mnt/storage/hf_cache/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE     = "cuda:3"
DTYPE      = torch.bfloat16

# ── Load model ─────────────────────────────────────────────────────────────
print("[INFO] Loading Qwen3-TTS model ...")
model = Qwen3TTSModel.from_pretrained(
    PRETRAINED,
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="sdpa",
)
print("[INFO] Model loaded.")

# ── Load target language lines ─────────────────────────────────────────────
target_lines: dict[str, tuple[list[str], str]] = {}
for lang_code, (path, lang_name) in TARGET_LANGS.items():
    lines = [
        l.strip()
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    target_lines[lang_code] = (lines, lang_name)
    print(f"[INFO] {lang_code} ({lang_name}): {len(lines)} lines")

# ── Load speakers ──────────────────────────────────────────────────────────
speaker_files = sorted(REF_AUDIO_DIR.glob("*.wav"))
print(f"[INFO] Found {len(speaker_files)} speakers.")

total    = sum(len(lines) for lines, _ in target_lines.values()) * len(speaker_files)
done     = 0
skipped  = 0
errors   = 0

# ── Generation loop ────────────────────────────────────────────────────────
for spk_path in speaker_files:
    spk_id   = spk_path.stem
    txt_path = REF_TEXT_DIR / f"{spk_id}.txt"

    if not txt_path.exists():
        print(f"[WARN] No transcript for {spk_id}, skipping speaker.")
        skipped += len(target_lines) * sum(
            len(lines) for lines, _ in target_lines.values()
        )
        continue

    ref_text  = " ".join(
        l.strip()
        for l in txt_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    )
    ref_audio = str(spk_path)
    print(f"\n[INFO] Speaker: {spk_id}  (ref_text: {len(ref_text)} chars)")

    for lang_code, (lines, lang_name) in target_lines.items():
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

            wavs = None
            try:
                wavs, sr = model.generate_voice_clone(
                    text=target_text,
                    language=lang_name,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=True,
                )
                sf.write(str(out_path), wavs[0], sr)
                print(f"  [OK] Saved → {out_path.name}  (sr={sr})")
                done += 1

            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                errors += 1

            except Exception as e:
                print(f"  [ERROR] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                errors += 1

            finally:
                try:
                    del wavs
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

            progress = f"{done}/{total}"
            print(f"  [PROGRESS] {progress}  errors={errors}  skipped={skipped}")

print(f"\n{'='*60}")
print(f"[DONE] Total={total}  Done={done}  Errors={errors}  Skipped={skipped}")
print(f"Output → {OUTPUT_DIR.resolve()}")