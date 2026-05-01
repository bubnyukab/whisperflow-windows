"""Generate v3 targeted training data — 15K pairs to fix v2 regressions.

Composition:
  5K spelling pairs   - each input contains >=1 of:
      definately, recieve, occured, seperate, teh, becuz,
      beleive, freind, wierd, untill
  5K cardinal pairs   - clean cardinal-word sequences (one two three...)
                        that must pass through as-is
  5K mixed pairs      - resampled from training_data_v2.jsonl
                        (anti-forgetting anchor)

Cardinal guard: a corrupt() call on text whose word content is
3+ cardinal words and *only* cardinal words skips number-word and
homophone corruptions. Prevents the v2 regression where the model
learned to convert "one two three four five" -> "1 to 3 to 4 to 5".

Run from the project root:
    .venv/Scripts/python.exe tools/generate_training_data_v3.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

random.seed(43)  # Different seed from v2 (which used 42).

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_PATH = Path(__file__).parent / "training_data_v3.jsonl"
V2_DATA_PATH = Path(__file__).parent / "training_data_v2.jsonl"

N_SPELLING = 5_000
N_CARDINAL = 5_000
N_MIXED = 5_000

CARDINAL_WORDS: list[str] = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]
_CARDINAL_SET = set(CARDINAL_WORDS)

FILLERS = [
    "um", "uh", "like", "so", "you know", "I mean",
    "basically", "literally", "actually", "right",
]

# Spelling targets: (correct, misspelled) — case variants included so
# regex \b matching works on real text.
SPELLING_TARGETS: list[tuple[str, str]] = [
    ("definitely", "definately"), ("Definitely", "Definately"),
    ("receive",    "recieve"),    ("Receive",    "Recieve"),
    ("received",   "recieved"),   ("Received",   "Recieved"),
    ("occurred",   "occured"),    ("Occurred",   "Occured"),
    ("separate",   "seperate"),   ("Separate",   "Seperate"),
    ("the",        "teh"),        ("The",        "Teh"),
    ("because",    "becuz"),      ("Because",    "Becuz"),
    ("believe",    "beleive"),    ("Believe",    "Beleive"),
    ("believed",   "beleived"),   ("Believed",   "Beleived"),
    ("friend",     "freind"),     ("Friend",     "Freind"),
    ("friends",    "freinds"),    ("Friends",    "Freinds"),
    ("weird",      "wierd"),      ("Weird",      "Wierd"),
    ("until",      "untill"),     ("Until",      "Untill"),
]

# Group by lowercase root so we can balance the dataset 500/word x 10.
_SPELLING_GROUPS: dict[str, list[tuple[str, str]]] = {}
for correct, misspelled in SPELLING_TARGETS:
    key = correct.lower().rstrip("d").rstrip("s")  # collapse received->receive, friends->friend
    _SPELLING_GROUPS.setdefault(key, []).append((correct, misspelled))

# Verification list: what a successful spelling-pair input must contain.
SPELLING_TYPOS_LOWER: tuple[str, ...] = tuple({m.lower() for _, m in SPELLING_TARGETS})


# ── Cardinal-only guard ───────────────────────────────────────────────────────

def is_cardinal_only_sequence(text: str) -> bool:
    """Return True iff text's word content is >=3 cardinal words and ONLY
    cardinal words. Used to short-circuit number/homophone corruptions."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) < 3:
        return False
    return all(w in _CARDINAL_SET for w in words)


# ── Light corruption ops (used by spelling + cardinal generators) ─────────────

def insert_fillers(text: str) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    out: list[str] = []
    for i, w in enumerate(words):
        out.append(w)
        if 0 < i < len(words) - 1 and random.random() < 0.18:
            out.append(random.choice(FILLERS))
    if random.random() < 0.30:
        out.insert(0, random.choice(FILLERS))
    return " ".join(out)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[.,?!;:]", "", text)


def lowercase_all(text: str) -> str:
    return text.lower()


def force_typo(text: str, correct: str, misspelled: str) -> str:
    """Replace one whole-word occurrence of correct with misspelled."""
    return re.sub(rf"\b{re.escape(correct)}\b", misspelled, text, count=1)


def corrupt_with_guard(text: str) -> str:
    """Apply 1-3 light corruptions; skip number/homophone rules entirely
    if input is a cardinal-only sequence (the v2-regression guard)."""
    skip_dangerous = is_cardinal_only_sequence(text)

    safe_ops: list[tuple[float, callable]] = [
        (0.55, lowercase_all),
        (0.55, remove_punctuation),
        (0.45, insert_fillers),
    ]
    if not skip_dangerous:
        # Number/homophone ops are intentionally unused in v3 generation —
        # the regressions came from the model over-applying them. We keep
        # the conditional structure so future generators can opt back in.
        pass

    chosen = [op for prob, op in safe_ops if random.random() < prob]
    if not chosen:
        chosen = [insert_fillers]

    result = text
    for op in chosen:
        result = op(result)
    return result


# ── Real-text source for spelling pairs ───────────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(paragraph: str) -> list[str]:
    paragraph = paragraph.replace("\n", " ").strip()
    if not paragraph:
        return []
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def good_sentence(s: str) -> bool:
    n = len(s.split())
    if not (8 <= n <= 80):
        return False
    if "@" in s or "http" in s.lower() or "www." in s.lower():
        return False
    if s.lstrip().startswith("="):
        return False
    alpha = sum(c.isalpha() for c in s)
    if alpha < len(s) * 0.55:
        return False
    return True


def collect_spelling_sources(per_group: int) -> dict[str, list[str]]:
    """Stream wikitext until we have `per_group` sentences for each spelling group."""
    from datasets import load_dataset
    print(f"  loading wikitext-103-raw-v1 (streaming) for spelling sources ...")

    sources: dict[str, list[str]] = {k: [] for k in _SPELLING_GROUPS}
    seen: set[str] = set()

    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1", split="train", streaming=True,
    )

    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        for sent in split_sentences(text):
            if sent in seen or not good_sentence(sent):
                continue
            seen.add(sent)
            for group_key, variants in _SPELLING_GROUPS.items():
                if len(sources[group_key]) >= per_group:
                    continue
                for correct, _ in variants:
                    if re.search(rf"\b{re.escape(correct)}\b", sent):
                        sources[group_key].append(sent)
                        break
        if all(len(v) >= per_group for v in sources.values()):
            break

    return sources


def build_spelling_pairs(target: int) -> list[dict]:
    """Build pairs each containing >=1 targeted misspelling in the input."""
    n_groups = len(_SPELLING_GROUPS)
    per_group = target // n_groups + 50  # buffer

    print(f"\n[1/3] Spelling pairs ({target}, {per_group}/group across {n_groups} groups)")
    sources = collect_spelling_sources(per_group)
    for k, v in sources.items():
        print(f"  group {k:>10s}: {len(v):,} source sentences")

    pairs: list[dict] = []
    for group_key, sents in sources.items():
        random.shuffle(sents)
        variants = _SPELLING_GROUPS[group_key]
        per_group_target = target // n_groups
        added = 0
        for sent in sents:
            if added >= per_group_target:
                break
            # Find a variant whose correct form actually appears in this sentence.
            usable = [(c, m) for c, m in variants
                      if re.search(rf"\b{re.escape(c)}\b", sent)]
            if not usable:
                continue
            correct, misspelled = random.choice(usable)
            dirty = force_typo(sent, correct, misspelled)
            dirty = corrupt_with_guard(dirty)
            if dirty.strip() == sent.strip():
                continue
            if misspelled.lower() not in dirty.lower():
                # Lower/strip step might have eaten the typo (it shouldn't
                # but be defensive); skip rather than emit an unverified pair.
                continue
            pairs.append({"input": dirty, "output": sent})
            added += 1

    random.shuffle(pairs)
    return pairs[:target]


# ── Cardinal pass-through pairs ───────────────────────────────────────────────

def _cardinal_sequence(min_len: int = 3, max_len: int = 10) -> list[str]:
    """Return a 3-10-word cardinal sequence, varied between monotonic/subset/random."""
    n = random.randint(min_len, min(max_len, 10))
    r = random.random()
    if r < 0.45:
        # Monotonic from 'one'
        return CARDINAL_WORDS[:n]
    if r < 0.75:
        # Sorted random subset
        chosen = random.sample(CARDINAL_WORDS, k=min(n, len(CARDINAL_WORDS)))
        return sorted(chosen, key=CARDINAL_WORDS.index)
    # Free-form (allows repeats)
    return [random.choice(CARDINAL_WORDS) for _ in range(n)]


def build_cardinal_pairs(target: int) -> list[dict]:
    """Generate clean cardinal sequences with light corruption on the input
    side (lowercase, fillers, missing punctuation). Output is always the
    canonical 'Cap rest. ' form. The guard ensures no number/homophone
    corruption is applied — input == lightly-noised cardinal-only sequence."""
    print(f"\n[2/3] Cardinal pass-through pairs ({target})")
    pairs: list[dict] = []
    attempts = 0
    while len(pairs) < target and attempts < target * 3:
        attempts += 1
        words = _cardinal_sequence()
        clean = " ".join(words)
        # Canonical clean form: capitalised first letter, trailing period.
        clean_out = clean[0].upper() + clean[1:] + "."

        # Construct the dirty input via guard-aware corruption.
        dirty_seed = clean[0].upper() + clean[1:] if random.random() < 0.25 else clean
        dirty = corrupt_with_guard(dirty_seed)

        # If corruption stripped fillers and re-cased to identical canonical
        # form, the pair has no learning signal -> skip.
        if dirty.strip() == clean_out.strip():
            continue

        pairs.append({"input": dirty, "output": clean_out})

    return pairs[:target]


# ── Mixed pairs (anti-forgetting) ─────────────────────────────────────────────

def load_v2_sample(target: int) -> list[dict]:
    print(f"\n[3/3] Mixed pairs from v2 ({target})")
    if not V2_DATA_PATH.exists():
        raise FileNotFoundError(f"v2 dataset not found: {V2_DATA_PATH}")

    all_pairs: list[dict] = []
    with V2_DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))

    print(f"  {len(all_pairs):,} v2 pairs available")
    return random.sample(all_pairs, min(target, len(all_pairs)))


# ── Verification + reporting ──────────────────────────────────────────────────

def verify(spelling: list[dict], cardinal: list[dict], mixed: list[dict]) -> None:
    print(f"\n{'='*64}")
    print(f"  Verification")
    print(f"{'='*64}")

    n_with_typo = sum(
        1 for p in spelling
        if any(t in p["input"].lower() for t in SPELLING_TYPOS_LOWER)
    )
    print(f"  spelling pairs with target typo : {n_with_typo:,} / {len(spelling):,}")

    n_cardinal_guarded = sum(1 for p in cardinal if is_cardinal_only_sequence(p["input"]))
    print(f"  cardinal pairs detected by guard: {n_cardinal_guarded:,} / {len(cardinal):,}")

    n_mixed = len(mixed)
    print(f"  mixed-from-v2 pairs             : {n_mixed:,}")

    # Sanity: the user's exact regression input must be representable.
    sample = "one two three four five"
    print(f"  guard fires on '{sample}': {is_cardinal_only_sequence(sample)}")


def print_samples(pairs: list[dict], n: int = 12) -> None:
    print(f"\n{'='*64}")
    print(f"  {n} Random Examples")
    print(f"{'='*64}")
    for i, p in enumerate(random.sample(pairs, min(n, len(pairs))), 1):
        print(f"\n[{i}]")
        print(f"  IN : {p['input']}")
        print(f"  OUT: {p['output']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating v3 training data: "
          f"{N_SPELLING + N_CARDINAL + N_MIXED:,} pairs total")

    spelling = build_spelling_pairs(N_SPELLING)
    cardinal = build_cardinal_pairs(N_CARDINAL)
    mixed = load_v2_sample(N_MIXED)

    pairs = spelling + cardinal + mixed
    random.shuffle(pairs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pairs):,} pairs -> {OUTPUT_PATH}")
    verify(spelling, cardinal, mixed)
    print_samples(pairs, n=12)


if __name__ == "__main__":
    main()
