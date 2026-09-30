# card_ocr

Turns a folder of printed card photos into a JSON deck using a local vision model
(Qwen3-VL-8B, 4-bit, runs on the RTX 3070).

## Setup

```
cd tools/card_ocr
uv sync
```

The first run downloads the model (~17 GB) into the Hugging Face cache.

## Usage

```
uv run python card_ocr.py "path/to/Tier 1"
```

- Writes `data/cards/<folder name>.json` at the repo root (change with `--out`).
- The tier is the first number in the folder name (override with `--tier 2`).
- A photo can hold one or several cards.
- Reruns only scan new photos, so edits you make to the JSON are kept.
  `--redo` rescans every photo and replaces those cards.
- Unclear words are marked `[?]` and the card gets `"needs_review": true`.
  Search for that after each run.
- Photos are shrunk to 1024 px on the long side (about 12 s per photo). Bigger sizes
  overflow the 8 GB GPU and run about 10x slower. If you put many cards in one photo
  and small text gets misread, try `--max-side 1280`.
- `--model Qwen/Qwen3-VL-4B-Instruct` is a faster, less accurate option.

## Deck format

Cards split into a Hers and a His part go into `hers` and `his`. The two parts of one
printed card share a number (`tier_1_hers_001` and `tier_1_his_001`). Cards without
that split go into `both`. Decks made before this split are moved into `both` the next
time the script runs on them.

```json
{
  "deck": "Tier 1",
  "tier": 1,
  "hers": [
    {
      "id": "tier_1_hers_001",
      "tier": 1,
      "title": "Ring Fence!",
      "text": "Ask your partner a thoughtful question ...",
      "category": "",
      "tags": [],
      "source_image": "card_006.jpg",
      "needs_review": false
    }
  ],
  "his": [
    {"id": "tier_1_his_001", "title": "Daily challenge.", "text": "Try a small teamwork challenge ...", "...": "..."}
  ],
  "both": []
}
```

A photo that gives a Hers part without a His part (or the other way round) is flagged
`needs_review`, since one of them was probably missed.
