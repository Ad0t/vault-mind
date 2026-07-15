"""
Pre-generation data quality audit.
Run from the project root:
    python src/audit_chunks.py
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
chunks_path = ROOT / "data" / "processed" / "chunks.json"

with chunks_path.open(encoding="utf-8") as f:
    chunks = json.load(f)

sep = "-" * 55

print(sep)
print("CHUNK CORPUS AUDIT")
print(sep)

# --- Basic counts ---
print(f"\nTotal chunks       : {len(chunks)}")
type_counts = Counter(c["type"] for c in chunks)
for t, n in type_counts.items():
    print(f"  {t:<12}: {n}")

# --- Metadata completeness ---
required_fields = [
    "chunk_id", "type", "start_page", "end_page",
    "source_doc", "section_title", "heading_level",
    "chunk_index", "text",
]
bad_meta = []
for c in chunks:
    missing = [k for k in required_fields if c.get(k) is None]
    if missing:
        bad_meta.append((c["chunk_id"], missing))

print(f"\nChunks with missing required metadata : {len(bad_meta)}")
for cid, fields in bad_meta[:5]:
    print(f"  {cid}: missing {fields}")

# --- Text length stats ---
lengths = [len(c["text"]) for c in chunks]
print(f"\nText length (chars)")
print(f"  min    : {min(lengths)}")
print(f"  max    : {max(lengths)}")
print(f"  mean   : {statistics.mean(lengths):.0f}")
print(f"  median : {statistics.median(lengths):.0f}")

# --- Very short chunks (likely parsing noise) ---
short = [c for c in chunks if len(c["text"]) < 50]
print(f"\nVery short chunks (<50 chars) : {len(short)}")
for c in short[:5]:
    print(f"  {c['chunk_id']} | {repr(c['text'])}")

# --- Chunks with blank section title ---
no_title = [c for c in chunks if not str(c.get("section_title", "")).strip()]
print(f"\nChunks with blank section title : {len(no_title)}")

# --- Private-use Unicode (PDF bullet artifacts like \uf0b7) ---
weird = [c for c in chunks if re.search(r"[\uf000-\uf8ff]", c["text"])]
print(f"\nChunks with private-use Unicode : {len(weird)}")
if weird:
    # Show the distinct codepoints found
    found = set()
    for c in weird:
        found.update(re.findall(r"[\uf000-\uf8ff]", c["text"]))
    print(f"  Codepoints : {[hex(ord(ch)) for ch in sorted(found)]}")
    print(f"  Example    : {repr(weird[0]['text'][:120])}")

# --- Approximate token count (rough: chars / 4) ---
# GPT-4o-mini context: 128k tokens. Claude Haiku: 200k tokens.
# A typical prompt + query overhead is ~500 tokens.
# Each retrieved chunk should stay well under 1000 tokens to leave room
# for multiple chunks + the generated answer.
approx_tokens = [len(c["text"]) // 4 for c in chunks]
over_1k = [c for c, t in zip(chunks, approx_tokens) if t > 1000]
print(f"\nApprox token count (chars/4)")
print(f"  mean   : {statistics.mean(approx_tokens):.0f}")
print(f"  max    : {max(approx_tokens)}")
print(f"  chunks >1000 tokens : {len(over_1k)}")
if over_1k:
    for c in over_1k[:3]:
        print(f"  {c['chunk_id']} | {len(c['text'])//4} tokens | {c['section_title'][:50]}")

# --- Duplicate text check ---
text_counts = Counter(c["text"] for c in chunks)
dupes = {t: n for t, n in text_counts.items() if n > 1}
print(f"\nExact duplicate chunk texts : {len(dupes)}")
if dupes:
    for text, n in list(dupes.items())[:3]:
        print(f"  x{n}: {repr(text[:80])}")

# --- Out-of-scope retrieval simulation ---
# A good guardrail needs to know what "no good answer" looks like.
# Print the section title distribution to spot if the corpus is well-labelled.
titled = Counter(c["section_title"] for c in chunks)
print(f"\nUnique section titles : {len(titled)}")
top_n = 5
print(f"Top {top_n} most common:")
for title, n in titled.most_common(top_n):
    print(f"  x{n} | {title[:60]}")

print(f"\n{sep}")
print("READY FOR GENERATION?")
print(sep)
issues = []
if bad_meta:
    issues.append(f"  - {len(bad_meta)} chunks have missing metadata fields (citations may break)")
if short:
    issues.append(f"  - {len(short)} very short chunks (<50 chars) may add noise to LLM context")
if weird:
    issues.append(f"  - {len(weird)} chunks contain private-use Unicode (clean before sending to LLM)")
if over_1k:
    issues.append(f"  - {len(over_1k)} chunks exceed ~1000 tokens (may crowd out other chunks in context)")
if dupes:
    issues.append(f"  - {len(dupes)} duplicate chunk texts (wastes context window tokens)")

if issues:
    print("Issues to address:")
    for i in issues:
        print(i)
else:
    print("No blocking issues found. Good to proceed with generation.")
