import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Avoid concurrent tensor materialization spikes during from_pretrained().
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"

from pathlib import Path
import importlib.util
import types
import gc
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ── Paths ──────────────────────────────────────────────────────────────────
BLIND_DATA_DIR = Path("clone/iwslt_blind")
REF_AUDIO_DIR  = BLIND_DATA_DIR / "reference_audio" / "crop_ref_audio"
REF_TEXT_DIR   = BLIND_DATA_DIR / "reference_texts" / "crop_ref_texts"
OUTPUT_DIR     = BLIND_DATA_DIR / "clone_audio" / "moss-tts"

TARGET_LANGS = {
    "ar": BLIND_DATA_DIR / "text" / "arabic.txt",
    "fr": BLIND_DATA_DIR / "text" / "french.txt",
    "zh": BLIND_DATA_DIR / "text" / "chinese.txt",
}

# ── Model config ───────────────────────────────────────────────────────────
PRETRAINED    = "OpenMOSS-Team/MOSS-TTS"
TOKENIZER_DEV = "cuda:1"                  # GPU 1 — 9.7 GiB free, isolated from model
_MAX_MEMORY = {
    0: "0GiB",
    1: "0GiB",
    2: "0GiB",
    3: "3GiB",
    4: "7GiB",
    5: "10GiB",
    6: "0GiB",
}
PRIMARY_DEV   = torch.device("cuda:3")    # keep first-shard input placement
DTYPE         = torch.bfloat16

# ── Load model ─────────────────────────────────────────────────────────────
print("[INFO] Loading processor ...")
processor = AutoProcessor.from_pretrained(PRETRAINED, trust_remote_code=True)
processor.audio_tokenizer = processor.audio_tokenizer.to(TOKENIZER_DEV)
print("[INFO] Processor loaded.")

print("[INFO] Loading model ...")
model = AutoModel.from_pretrained(
    PRETRAINED,
    trust_remote_code=True,
    attn_implementation="sdpa",
    dtype=DTYPE,
    device_map="sequential",
    max_memory=_MAX_MEMORY,
)
model.eval()


def _patched_get_input_embeddings(self, input_ids):
    inputs_embeds = self.language_model.embed_tokens(input_ids[..., 0])
    target_device = inputs_embeds.device
    for i, embed_layer in enumerate(self.emb_ext):
        inputs_embeds = inputs_embeds + embed_layer(
            input_ids[..., i + 1]
        ).to(target_device)
    return inputs_embeds


model.get_input_embeddings = types.MethodType(_patched_get_input_embeddings, model)
print("[INFO] Model loaded.")

# ── Load target language lines ─────────────────────────────────────────────
target_lines = {}
for lang_code, path in TARGET_LANGS.items():
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    target_lines[lang_code] = lines
    print(f"[INFO] {lang_code}: {len(lines)} lines")

# ── Load speakers ──────────────────────────────────────────────────────────
speaker_files = sorted(REF_AUDIO_DIR.glob("*.wav"))
print(f"[INFO] Found {len(speaker_files)} speakers.")

# ── Generation loop ────────────────────────────────────────────────────────
for spk_path in speaker_files:
    spk_id   = spk_path.stem          # e.g. "2023.acl-long.12"
    txt_path = REF_TEXT_DIR / f"{spk_id}.txt"

    if not txt_path.exists():
        print(f"[WARN] No transcript for {spk_id}, skipping.")
        continue

    ref_text  = " ".join(
        l.strip()
        for l in txt_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    )
    ref_audio = str(spk_path)
    print(f"\n[INFO] Speaker: {spk_id}  (ref_text: {len(ref_text)} chars)")

    for lang_code, lines in target_lines.items():
        out_lang_dir = OUTPUT_DIR / lang_code
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        for line_idx, target_text in enumerate(lines):
            out_path = out_lang_dir / f"{spk_id}_line{line_idx + 1:03d}.wav"

            if out_path.exists():
                print(f"  [SKIP] {out_path.name} already exists.")
                continue

            print(f"  [{lang_code}] line {line_idx + 1}/{len(lines)} → {out_path.name}")

            try:
                conversation = [
                    processor.build_user_message(
                        text=ref_text + " " + target_text,
                        reference=[ref_audio],
                    ),
                    processor.build_assistant_message(
                        audio_codes_list=[ref_audio],
                    ),
                ]

                batch          = processor([conversation], mode="continuation")
                input_ids      = batch["input_ids"].to(PRIMARY_DEV)
                attention_mask = batch["attention_mask"].to(PRIMARY_DEV)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        audio_temperature=1.0,
                        audio_top_p=0.8,
                        audio_top_k=25,
                        audio_repetition_penalty=1.0,
                    )

                messages = processor.decode(outputs)
                audio    = messages[0].audio_codes_list[0]
                torchaudio.save(
                    str(out_path),
                    audio.unsqueeze(0),
                    processor.model_config.sampling_rate,
                )
                print(f"  [OK] Saved → {out_path}")

            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {spk_id} {lang_code} line {line_idx + 1}: {e}")
                print("  [OOM] Clearing cache and continuing ...")
                continue

            finally:
                # Always clean up after every clip regardless of success or failure.
                try:
                    del batch, input_ids, attention_mask, outputs, messages
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

print("\n[DONE] All generation complete.")
