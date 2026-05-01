"""Generate (dirty, clean) training pairs from real text + targeted templates.

Pulls clean source sentences from HuggingFace datasets (wikitext, bookcorpus
or ag_news fallback) and adds three pass-through categories the model must
*not* answer (math questions, instructions, technical jargon). Applies a mix
of corruptions per pair: fillers, homophones, lowercasing, missing punctuation,
spelling errors, number-word substitution, duplicate words, and audio dropouts.

Run from project root:
    .venv/Scripts/python.exe tools/generate_training_data.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

random.seed(42)

# Ensure UTF-8 stdout so non-ASCII characters from real-text datasets
# (e.g. Japanese romanization "ō") don't crash the stats display on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_PATH = Path(__file__).parent / "training_data_v2.jsonl"
TARGET_PAIRS = 50_000

# Category quotas for pass-through (model must clean, not answer).
N_MATH = 500
N_INSTRUCTION = 500
N_TECHNICAL = 500
# The remaining slots come from real-text datasets and corruption-only pairs.

MIN_WORDS, MAX_WORDS = 8, 120

# ── Filler words ───────────────────────────────────────────────────────────────

FILLERS = [
    "um", "uh", "like", "you know", "so", "basically", "literally",
    "kind of", "I mean", "right", "actually", "sort of",
]

# ── Homophone pairs (case-insensitive whole-word swap, randomised direction) ──
# Each tuple is a homophone group; corruption picks any *other* member of the
# group as the swap target, so "their"→"there", "there"→"they're", etc.
HOMOPHONE_GROUPS: list[tuple[str, ...]] = [
    ("their", "there", "they're"),
    ("to", "too", "two"),
    ("your", "you're"),
    ("write", "right"),
    ("by", "buy", "bye"),
    ("hear", "here"),
    ("no", "know"),
    ("whether", "weather"),
    ("its", "it's"),
    ("then", "than"),
]

# Build word→[swap candidates] for fast lookup.
_HOMOPHONE_LOOKUP: dict[str, list[str]] = {}
for group in HOMOPHONE_GROUPS:
    for w in group:
        _HOMOPHONE_LOOKUP[w.lower()] = [x for x in group if x.lower() != w.lower()]


# ── Number-word maps ──────────────────────────────────────────────────────────
# Inverse direction (digit→word) is what STT actually does, so we corrupt
# clean digit text into spoken form.

_DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "21": "twenty one",
    "22": "twenty two", "23": "twenty three", "24": "twenty four",
    "25": "twenty five", "30": "thirty", "40": "forty", "50": "fifty",
    "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "one hundred", "1000": "one thousand",
}

_ORDINAL_TO_WORD = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "20th": "twentieth", "21st": "twenty first",
}


# ── Spelling-error map ────────────────────────────────────────────────────────

SPELLING_SUBS = [
    (r"\bdefinitely\b",  "definately"),
    (r"\bDefinitely\b",  "Definately"),
    (r"\breceive\b",     "recieve"),
    (r"\bReceive\b",     "Recieve"),
    (r"\boccurred\b",    "occured"),
    (r"\boccurrence\b",  "occurence"),
    (r"\bseparate\b",    "seperate"),
    (r"\bSeparate\b",    "Seperate"),
    (r"\bthe\b",         "teh"),
    (r"\bThe\b",         "Teh"),
    (r"\bbecause\b",     "becuz"),
    (r"\bBecause\b",     "Becuz"),
    (r"\bbecause\b",     "cuz"),
    (r"\bI'm\b",         "im"),
    (r"\bI am\b",        "im"),
    (r"\bdon't\b",       "dont"),
    (r"\bDon't\b",       "Dont"),
    (r"\bcan't\b",       "cant"),
    (r"\bCan't\b",       "Cant"),
    (r"\bwon't\b",       "wont"),
    (r"\bWon't\b",       "Wont"),
    (r"\byou are\b",     "ur"),
    (r"\byour\b",        "ur"),
    (r"\btheir\b",       "thier"),
    (r"\bweird\b",       "wierd"),
    (r"\baccommodate\b", "accomodate"),
    (r"\bcommittee\b",   "commitee"),
    (r"\bexistence\b",   "existance"),
    (r"\bconsistent\b",  "consistant"),
]
_COMPILED_SUBS = [(re.compile(p), r) for p, r in SPELLING_SUBS]


# ── Pass-through templates ────────────────────────────────────────────────────
# These are sentences where output == cleaned(input). The model must NOT answer.

MATH_OPS = [("plus", "+"), ("minus", "-"), ("times", "×"),
            ("divided by", "÷"), ("multiplied by", "×")]

MATH_NUMS_SMALL = list(range(1, 21))
MATH_NUMS_LARGE = [25, 30, 50, 75, 100, 125, 150, 200, 500, 1000]


def _math_question() -> str:
    op_word, _ = random.choice(MATH_OPS)
    a = random.choice(MATH_NUMS_SMALL + MATH_NUMS_LARGE)
    b = random.choice(MATH_NUMS_SMALL)
    template = random.choice([
        f"What is {a} {op_word} {b}?",
        f"Can you tell me what {a} {op_word} {b} is?",
        f"How much is {a} {op_word} {b}?",
        f"Calculate {a} {op_word} {b}.",
        f"What's {a} {op_word} {b}?",
    ])
    return template


INSTRUCTION_TEMPLATES = [
    "Write me an email to {target} about {topic}.",
    "Draft a message to {target} regarding {topic}.",
    "Compose a short note for {target} on {topic}.",
    "Send {target} an update about {topic}.",
    "Remind me to {action} {when}.",
    "Add {item} to my shopping list.",
    "Schedule a meeting with {target} for {when}.",
    "Set a reminder for {when} to {action}.",
    "Book {item} for {when}.",
    "Cancel my {item} for {when}.",
    "Find me a {item} near {target}.",
    "Look up {topic} for me.",
    "Search for the best {item} in {target}.",
    "Order {item} from {target}.",
    "Translate this paragraph to {target}.",
    "Summarize the notes on {topic} for me.",
    "Create a {item} for {topic}.",
    "Write a short summary about {topic}.",
    "Help me draft notes on {topic} for {target}.",
]
INSTR_TARGETS = ["John", "Sarah", "the team", "my manager", "the client",
                 "marketing", "legal", "the board", "Alex", "engineering",
                 "Spanish", "French", "German", "the office", "everyone"]
INSTR_TOPICS = ["the deadline", "next quarter's plan", "the budget review",
                "the new feature", "the launch date", "today's outage",
                "Friday's all-hands", "the contract terms", "the demo",
                "the timeline", "Q3 priorities", "the project update",
                "compliance", "onboarding", "vendor selection"]
INSTR_ACTIONS = ["call the dentist", "buy groceries", "pay the rent",
                 "follow up with HR", "review the PR", "submit the report",
                 "send the invoice", "renew my passport", "water the plants",
                 "take the medication", "pick up the package"]
INSTR_WHEN = ["tomorrow at 9", "Friday afternoon", "next Monday", "tonight",
              "in an hour", "at 3pm", "before the weekend", "by end of day",
              "next week", "this evening"]
INSTR_ITEMS = ["doc", "spreadsheet", "playlist", "folder", "report",
               "appointment", "ticket", "table for two", "Uber", "flight",
               "milk", "bread", "coffee", "running shoes", "book",
               "Italian restaurant", "coffee shop", "doctor", "barber"]


def _instruction_sentence() -> str:
    template = random.choice(INSTRUCTION_TEMPLATES)
    return template.format(
        target=random.choice(INSTR_TARGETS),
        topic=random.choice(INSTR_TOPICS),
        action=random.choice(INSTR_ACTIONS),
        when=random.choice(INSTR_WHEN),
        item=random.choice(INSTR_ITEMS),
    )


# A grab-bag of jargon: code identifiers, drug names, legal/medical terms,
# acronyms — things that make Whisper lowercase or hallucinate.
TECHNICAL_TEMPLATES = [
    "We need to refactor the {token} module to use {token} for {purpose}.",
    "The {token} endpoint returns a {token} response with status code {num}.",
    "Patient was prescribed {token} {num} milligrams twice daily for {purpose}.",
    "The court ruled in {token} v. {token}, citing {token} precedent.",
    "Deploy {token} to the {token} environment after the {token} pipeline passes.",
    "Update the {token} package to fix CVE-{num}-{num}.",
    "The {token} algorithm has time complexity O of n {token}.",
    "Run kubectl apply -f {token}.yaml in the {token} namespace.",
    "Diagnosis: {token}, refer to {token} for follow-up.",
    "The {token} contract specifies {token} arbitration under {token} law.",
    "Initialize the {token} with {token} authentication and TLS {num} point {num}.",
    "Connect to {token}://{token}:{num}/{token} with the API key in the header.",
]
TECH_TOKENS = [
    "Kubernetes", "Docker", "PostgreSQL", "Redis", "Kafka", "GraphQL",
    "OAuth", "JWT", "Lambda", "Terraform", "Ansible", "Helm",
    "amoxicillin", "ibuprofen", "lisinopril", "metformin", "atorvastatin",
    "Brown", "Smith", "Doe", "Roe",
    "antitrust", "GDPR", "HIPAA", "SOC2", "ISO27001",
    "production", "staging", "dev", "qa",
    "binary search", "quicksort", "Dijkstra", "BFS", "DFS",
    "main", "feature", "release", "hotfix",
    "https", "grpc", "wss", "redis", "postgres",
    "auth-service", "api-gateway", "user-svc", "billing", "telemetry",
    "us-east-1", "eu-west-2", "ap-south-1",
    "TypeScript", "Rust", "Python", "Go", "Java",
]


def _technical_sentence() -> str:
    template = random.choice(TECHNICAL_TEMPLATES)
    out = template
    while "{token}" in out:
        out = out.replace("{token}", random.choice(TECH_TOKENS), 1)
    while "{num}" in out:
        out = out.replace("{num}", str(random.randint(1, 999)), 1)
    while "{purpose}" in out:
        out = out.replace("{purpose}", random.choice([
            "load balancing", "session management", "rate limiting",
            "high blood pressure", "infection control", "preventive care",
        ]), 1)
    return out


# ── Real-text loader (HuggingFace datasets) ───────────────────────────────────

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(paragraph: str) -> list[str]:
    paragraph = paragraph.replace("\n", " ").strip()
    if not paragraph:
        return []
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def _good_sentence(s: str) -> bool:
    n = len(s.split())
    if not (MIN_WORDS <= n <= MAX_WORDS):
        return False
    if "@" in s or "http" in s.lower() or "www." in s.lower():
        return False
    # Drop wikitext heading markers like "= = Title = ="
    if s.lstrip().startswith("="):
        return False
    # Skip obviously broken lines (lots of digits/symbols)
    alpha = sum(c.isalpha() for c in s)
    if alpha < len(s) * 0.55:
        return False
    return True


def load_real_sentences(target: int) -> list[str]:
    """Pull `target` clean sentences from HuggingFace datasets.

    Strategy: stream wikitext-103 first; if more are still needed, try
    bookcorpus, falling back to ag_news. Streaming avoids downloading
    the full ~500 MB if we only need 30K sentences.
    """
    from datasets import load_dataset

    out: list[str] = []
    seen: set[str] = set()

    def _ingest(stream, text_key: str, cap: int) -> None:
        for row in stream:
            if len(out) >= cap:
                return
            text = row.get(text_key, "")
            if not text:
                continue
            for sent in _split_sentences(text):
                if not _good_sentence(sent):
                    continue
                if sent in seen:
                    continue
                seen.add(sent)
                out.append(sent)
                if len(out) >= cap:
                    return

    print(f"  loading wikitext-103-raw-v1 (streaming) ...")
    try:
        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", streaming=True,
        )
        _ingest(ds, "text", target)
    except Exception as exc:
        print(f"  wikitext failed: {exc}")

    if len(out) < target:
        print(f"  have {len(out):,}, trying bookcorpus ...")
        try:
            ds = load_dataset(
                "bookcorpus", split="train", streaming=True,
                trust_remote_code=True,
            )
            _ingest(ds, "text", target)
        except Exception as exc:
            print(f"  bookcorpus failed ({exc}); falling back to ag_news ...")
            try:
                ds = load_dataset("ag_news", split="train", streaming=True)
                _ingest(ds, "text", target)
            except Exception as exc2:
                print(f"  ag_news also failed: {exc2}")

    random.shuffle(out)
    return out


# ── Corruption ops ────────────────────────────────────────────────────────────

def insert_fillers(text: str) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    result: list[str] = []
    for i, word in enumerate(words):
        result.append(word)
        if 0 < i < len(words) - 1 and random.random() < 0.15:
            result.append(random.choice(FILLERS))
    if random.random() < 0.30:
        result.insert(0, random.choice(FILLERS))
    return " ".join(result)


def remove_punctuation(text: str) -> str:
    return re.sub(r"[.,?!;:]", "", text)


def apply_spelling_errors(text: str) -> str:
    for pattern, replacement in _COMPILED_SUBS:
        if pattern.search(text) and random.random() < 0.40:
            text = pattern.sub(replacement, text, count=1)
    return text


def apply_homophone_swaps(text: str) -> str:
    """Swap individual words with another member of their homophone group."""
    def _swap(match: re.Match) -> str:
        w = match.group(0)
        candidates = _HOMOPHONE_LOOKUP.get(w.lower())
        if not candidates or random.random() > 0.55:
            return w
        repl = random.choice(candidates)
        # Preserve sentence-initial capitalisation only.
        if w[0].isupper() and not w.isupper():
            return repl[0].upper() + repl[1:]
        return repl
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in _HOMOPHONE_LOOKUP.keys()) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(_swap, text)


def apply_number_words(text: str) -> str:
    """Replace digit numbers and ordinals with their spoken-word forms."""
    def _digit_repl(match: re.Match) -> str:
        digits = match.group(0)
        return _DIGIT_TO_WORD.get(digits, digits)

    def _ord_repl(match: re.Match) -> str:
        return _ORDINAL_TO_WORD.get(match.group(0).lower(), match.group(0))

    text = re.sub(
        r"\b(?:1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|20th|21st)\b",
        _ord_repl, text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{1,4}\b", _digit_repl, text)
    return text


def duplicate_words(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    result: list[str] = []
    for word in words:
        result.append(word)
        if random.random() < 0.05:
            result.append(word)
    return " ".join(result)


def drop_words(text: str) -> str:
    """Drop 1-2 mid-sentence words to simulate audio dropout."""
    words = text.split()
    if len(words) < 6:
        return text
    n_drop = 1 if random.random() < 0.7 else 2
    # Avoid dropping the first or last word so the sentence still parses.
    drop_idxs = set()
    for _ in range(n_drop):
        if len(words) - len(drop_idxs) <= 4:
            break
        drop_idxs.add(random.randint(1, len(words) - 2))
    return " ".join(w for i, w in enumerate(words) if i not in drop_idxs)


def corrupt(text: str) -> str:
    """Apply a random mix of 2-4 corruptions per sample."""
    ops = [
        (0.65, insert_fillers),
        (0.55, lambda t: t.lower()),
        (0.55, remove_punctuation),
        (0.40, apply_spelling_errors),
        (0.30, apply_homophone_swaps),
        (0.25, apply_number_words),
        (0.15, duplicate_words),
        (0.20, drop_words),
    ]

    chosen = [op for prob, op in ops if random.random() < prob]
    if len(chosen) < 2:
        # Force at least 2 corruptions
        remaining = [op for prob, op in ops if op not in chosen]
        random.shuffle(remaining)
        chosen.extend(remaining[: 2 - len(chosen)])
    if len(chosen) > 4:
        chosen = chosen[:4]

    result = text
    # Spelling errors before lowercasing so case-sensitive patterns match.
    for op in sorted(chosen, key=lambda o: 0 if o is apply_spelling_errors else 1):
        result = op(result)
    return result


def corrupt_light(text: str) -> str:
    """Pass-through corruption — never drops content words.

    Math questions, short instructions, and technical jargon must keep
    every meaningful token so the model isn't trained to hallucinate the
    missing number / target / drug name.
    """
    ops = [
        (0.55, insert_fillers),
        (0.55, lambda t: t.lower()),
        (0.55, remove_punctuation),
        (0.30, apply_spelling_errors),
        (0.10, duplicate_words),
    ]

    chosen = [op for prob, op in ops if random.random() < prob]
    if not chosen:
        chosen = [insert_fillers]

    result = text
    for op in sorted(chosen, key=lambda o: 0 if o is apply_spelling_errors else 1):
        result = op(result)
    return result


# ── Pair generation ───────────────────────────────────────────────────────────

def _build_pass_through_pairs() -> list[dict]:
    """Build pairs where output is the cleaned-form of the question/instruction.

    The model must learn: "even when the input is a question, do not answer
    — produce a properly capitalised, punctuated version of the question."
    """
    pairs: list[dict] = []

    for _ in range(N_MATH):
        clean = _math_question()
        pairs.append({"input": corrupt_light(clean), "output": clean})

    for _ in range(N_INSTRUCTION):
        clean = _instruction_sentence()
        pairs.append({"input": corrupt_light(clean), "output": clean})

    for _ in range(N_TECHNICAL):
        clean = _technical_sentence()
        pairs.append({"input": corrupt_light(clean), "output": clean})

    return pairs


def generate_pairs(target: int) -> list[dict]:
    pass_through = _build_pass_through_pairs()
    print(f"  built {len(pass_through):,} pass-through pairs "
          f"(math + instructions + technical)")

    needed_real = target - len(pass_through)
    # Pull more sentences than we need so we can drop ones whose corruption
    # collides with the clean form.
    sentences = load_real_sentences(int(needed_real * 1.25))
    print(f"  loaded {len(sentences):,} real-text sentences")

    real_pairs: list[dict] = []
    attempts = 0
    max_attempts = needed_real * 6
    while len(real_pairs) < needed_real and attempts < max_attempts:
        clean = sentences[attempts % len(sentences)] if sentences else None
        attempts += 1
        if not clean:
            break
        dirty = corrupt(clean)
        if dirty.strip() == clean.strip():
            continue
        real_pairs.append({"input": dirty, "output": clean})

    pairs = pass_through + real_pairs
    random.shuffle(pairs)
    return pairs[:target]


# ── Stats & display ────────────────────────────────────────────────────────────

def print_stats(pairs: list[dict]) -> None:
    total = len(pairs)
    avg_in  = sum(len(p["input"])  for p in pairs) / total
    avg_out = sum(len(p["output"]) for p in pairs) / total

    print(f"\n{'='*64}")
    print(f"  Dataset Stats")
    print(f"{'='*64}")
    print(f"  Total pairs        : {total:,}")
    print(f"  Avg input length   : {avg_in:.1f} chars")
    print(f"  Avg output length  : {avg_out:.1f} chars")
    print(f"  Avg dirty overhead : {avg_in - avg_out:+.1f} chars")

    print(f"\n{'='*64}")
    print(f"  15 Random Examples")
    print(f"{'='*64}")
    for i, pair in enumerate(random.sample(pairs, min(15, total)), 1):
        print(f"\n[{i}]")
        print(f"  INPUT : {pair['input']}")
        print(f"  OUTPUT: {pair['output']}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating {TARGET_PAIRS:,} (dirty, clean) training pairs ...\n")
    pairs = generate_pairs(TARGET_PAIRS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(pairs):,} pairs -> {OUTPUT_PATH}")
    print_stats(pairs)


if __name__ == "__main__":
    main()
