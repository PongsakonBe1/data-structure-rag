"""Test retrieval and chunk selection logic directly"""
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load BM25 index
index_path = Path('indexes/bm25_index.pkl')
if not index_path.exists():
    print(f"ERROR: {index_path} not found!")
    sys.exit(1)

data = pickle.loads(index_path.read_bytes())
chunks = data['chunks']

print(f"Total chunks: {len(chunks)}\n")

# Show chunks from page 1
print("=== Chunks from page 1 ===")
page1_chunks = [c for c in chunks if c.metadata.get('page') == '1']
print(f"Found {len(page1_chunks)} chunks on page 1\n")

for i, doc in enumerate(page1_chunks[:5]):
    meta = doc.metadata or {}
    print(f"--- Chunk {i} ({meta.get('chunk_id', 'N/A')}) ---")
    print(f"  H3: {meta.get('H3', 'N/A')}")
    print(f"  section: {meta.get('section', 'N/A')}")
    print(f"  Content preview: {doc.page_content[:150]}...")
    print()

# Test section matching for 1.1.1
print("\n=== Test section matching for 1.1.1 ===")
target_section_id = "1.1.1"
patterns = [target_section_id, f"### {target_section_id}", "1.1.1 ความหมาย", "ความหมายโครงสร้างข้อมูล"]

def _has_section(d, sid, patterns):
    meta = d.metadata or {}
    check = " ".join([
        str(meta.get("H1", "")), str(meta.get("H2", "")), str(meta.get("H3", "")),
        str(meta.get("section", "")), str(meta.get("best_topic_id", "")),
        str(d.page_content[:200]),
    ])
    return any(p in check for p in patterns)

section_docs = [d for d in page1_chunks if _has_section(d, target_section_id, patterns)]
other_docs = [d for d in page1_chunks if not _has_section(d, target_section_id, patterns)]

print(f"Section docs (1.1.1): {len(section_docs)}")
for d in section_docs:
    meta = d.metadata or {}
    print(f"  - {meta.get('chunk_id')}: H3={meta.get('H3', 'N/A')}")

print(f"\nOther docs: {len(other_docs)}")
for d in other_docs[:3]:
    meta = d.metadata or {}
    print(f"  - {meta.get('chunk_id')}: H3={meta.get('H3', 'N/A')}")

# Simulate select_context_docs_diverse with target_section_id
print("\n=== Simulate select_context_docs_diverse ===")
context_doc_limit = 12
min_unique_pages = 2

# Reorder: section docs first
docs = section_docs + other_docs
print(f"After reorder: {len(docs)} docs")
print(f"First 3 docs: {[d.metadata.get('chunk_id') for d in docs[:3]]}")

# Simulate diversity selection
selected = []
used_keys = set()
seen_pages = set()

def _doc_key(d):
    meta = d.metadata or {}
    src = str(meta.get('source', '')).strip()
    page = str(meta.get('page', '')).strip()
    chunk = str(meta.get('chunk_id', '')).strip()
    return f"{src}:{page}:{chunk}"

# Phase 1: Page diversity
for d in docs:
    if len(selected) >= context_doc_limit:
        break
    meta = d.metadata or {}
    page = str(meta.get('page', '')).strip()
    key = _doc_key(d)
    if key in used_keys:
        continue
    if page and page in seen_pages:
        continue
    selected.append(d)
    used_keys.add(key)
    if page:
        seen_pages.add(page)

# Phase 2: Fill remaining
for d in docs:
    if len(selected) >= context_doc_limit:
        break
    key = _doc_key(d)
    if key in used_keys:
        continue
    selected.append(d)
    used_keys.add(key)

selected = selected[:context_doc_limit]

print(f"\nSelected {len(selected)} docs for context:")
for i, d in enumerate(selected):
    meta = d.metadata or {}
    print(f"  {i+1}. {meta.get('chunk_id')} (page {meta.get('page')}, H3={meta.get('H3', 'N/A')})")

# Check if we got section 1.1.1
has_111 = any(d.metadata.get('H3', '').startswith('1.1.1') for d in selected)
print(f"\n✅ Has 1.1.1 in context: {has_111}")
