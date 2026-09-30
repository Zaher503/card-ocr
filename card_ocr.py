"""Digitize a folder of card photos into a JSON deck using a local vision model."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

register_heif_opener()

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "cards"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SECTIONS = ("hers", "his", "both")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
# 1024 keeps the 8B model inside 8 GB VRAM; larger images spill into system RAM and run ~10x slower
MAX_SIDE = 1024

PROMPT = """This photo shows one or more printed cards from a couples board game.
Transcribe the text of every card exactly as written. Keep the original wording and spelling.
Some cards are split into a "Hers" part and a "His" part. Output each part as its own entry with "for" set to "hers" or "his", and leave the heading word itself out of the text.
Cards without those headings are one entry with "for" set to "both".
"title" is the short name that starts the entry's text, like "Ring Fence!", "Now You See Me, Now You Don't." or "Helpless Victim". It is never the word "Hers" or "His". Leave it out of "text". Use "" if there is none.
Join lines that wrap mid-sentence into one line. Only keep a line break between separate paragraphs or list items.
If a word is unclear, write your best guess followed by [?].
Ignore anything that is not card text: logos, icons, decorative borders, card numbers, the table, fingers, background.
Reply with JSON only, in this shape:
{"cards": [{"for": "hers", "title": "...", "text": "..."}, {"for": "his", "title": "...", "text": "..."}]}"""


def load_model(model_id, use_4bit):
    # Load model and processor
    kwargs = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def load_image(path, max_side=MAX_SIDE):
    # Fix phone rotation and shrink big photos
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def read_cards(model, processor, image):
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return parse_reply(reply)


def parse_reply(reply):
    # Pull (section, title, text) entries out of the reply, fall back to raw text
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            cards = json.loads(match.group(0)).get("cards", [])
            entries = []
            for c in cards:
                if not isinstance(c, dict):
                    continue
                section = str(c.get("for", "")).strip().lower()
                title = str(c.get("title", "")).strip()
                text = str(c.get("text", "")).strip()
                # The Hers/His heading is not a title
                if title.lower().strip(":. ") in ("hers", "his"):
                    title = ""
                if text:
                    entries.append((section if section in SECTIONS else "both", title, text))
            return entries, False
        except (json.JSONDecodeError, AttributeError):
            pass
    return [("both", "", reply.strip())], True


def guess_tier(folder_name):
    match = re.search(r"\d+", folder_name)
    return int(match.group(0)) if match else None


def load_deck(path, folder, tier):
    deck = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"deck": folder.name, "tier": tier}
    # Older decks kept everything in one "cards" list
    old = deck.pop("cards", [])
    for section in SECTIONS:
        deck.setdefault(section, [])
    deck["both"].extend(old)
    return deck


def all_cards(deck):
    return [c for section in SECTIONS for c in deck[section]]


def save_deck(path, deck):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(deck, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def next_id(deck):
    # Ids look like <deck>_001 or <deck>_hers_001 / <deck>_his_001, one counter per deck
    nums = [int(m.group(1)) for c in all_cards(deck) if (m := re.search(r"_(\d+)$", c.get("id", "")))]
    return max(nums, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder of card photos")
    parser.add_argument("--out", type=Path, help="output JSON path (default: data/cards/<folder>.json)")
    parser.add_argument("--tier", type=int, help="tier number (default: first number in the folder name)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE, help=f"shrink photos to this many pixels on the long side (default: {MAX_SIDE})")
    parser.add_argument("--no-4bit", action="store_true", help="load full precision (needs more VRAM)")
    parser.add_argument("--redo", action="store_true", help="rescan images already in the JSON (replaces their cards)")
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No images found in {folder}")

    out = args.out or DEFAULT_OUT_DIR / f"{folder.name}.json"
    tier = args.tier if args.tier is not None else guess_tier(folder.name)
    deck = load_deck(out, folder, tier)
    deck["tier"] = tier

    # Skip images that were already scanned so manual edits survive reruns
    if args.redo:
        redo_names = {p.name for p in images}
        for section in SECTIONS:
            deck[section] = [c for c in deck[section] if c.get("source_image") not in redo_names]
    done = {c.get("source_image") for c in all_cards(deck)}
    todo = [p for p in images if p.name not in done]
    print(f"{len(images)} images, {len(todo)} to scan -> {out}")
    if not todo:
        return

    print(f"Loading {args.model} ...")
    model, processor = load_model(args.model, not args.no_4bit)

    prefix = re.sub(r"\W+", "_", folder.name).strip("_").lower()
    n = next_id(deck)
    for i, path in enumerate(todo, 1):
        try:
            entries, raw = read_cards(model, processor, load_image(path, args.max_side))
        except Exception as e:
            print(f"[{i}/{len(todo)}] {path.name}: FAILED ({e})")
            continue
        # A Hers part without a matching His part usually means a misread
        sections = [s for s, _, _ in entries]
        unpaired = sections.count("hers") != sections.count("his")
        # The nth Hers and nth His on a photo share a number, "both" cards come after
        pairs = max(sections.count("hers"), sections.count("his"))
        seen = {"hers": 0, "his": 0, "both": 0}
        for section, title, text in entries:
            if section == "both":
                sid = f"{prefix}_{n + pairs + seen['both']:03d}"
            else:
                sid = f"{prefix}_{section}_{n + seen[section]:03d}"
            seen[section] += 1
            deck[section].append({
                "id": sid,
                "tier": tier,
                "title": title,
                "text": text,
                "category": "",
                "tags": [],
                "source_image": path.name,
                "needs_review": raw or unpaired or "[?]" in title + text,
            })
        n += pairs + seen["both"]
        save_deck(out, deck)
        preview = " | ".join(f"{s}: {(t or x).replace(chr(10), ' ')[:40]}" for s, t, x in entries)
        print(f"[{i}/{len(todo)}] {path.name}: {len(entries)} part(s) {'(REVIEW) ' if raw or unpaired else ''}{preview}")

    counts = ", ".join(f"{len(deck[s])} {s}" for s in SECTIONS)
    flagged = sum(c["needs_review"] for c in all_cards(deck))
    print(f"Done. {counts} in {out}, {flagged} flagged for review.")


if __name__ == "__main__":
    main()
