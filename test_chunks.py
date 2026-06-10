"""Test script to verify chunk selection logic"""
import pickle
import sys
from pathlib import Path

# Load BM25 index
index_path = Path('indexes/bm25_index.pkl')
if not index_path.exists():
    print(f"ERROR: {index_path} not found!")
    sys.exit(1)

data = pickle.loads(index_path.read_bytes())
chunks = data['chunks']

print(f"Total chunks: {len(chunks)}\n")

# Show first 5 chunks from page 1
print("=== Chunks from page 1 ===")
page1_chunks = [c for c in chunks if c.metadata.get('page') == '1']
for i, doc in enumerate(page1_chunks[:5]):
    meta = doc.metadata or {}
    print(f"\n--- Chunk {i} ---")
    print(f"  chunk_id: {meta.get('chunk_id', 'N/A')}")
    print(f"  page: {meta.get('page', 'N/A')}")
    print(f"  H1: {meta.get('H1', 'N/A')}")
    print(f"  H2: {meta.get('H2', 'N/A')}")
    print(f"  H3: {meta.get('H3', 'N/A')}")
    print(f"  section: {meta.get('section', 'N/A')}")
    print(f"  Content preview: {doc.page_content[:200]}...")

# Test section matching for 1.1.1
print("\n\n=== Test section matching for 1.1.1 ===")
target_section_id = "1.1.1"
patterns = [target_section_id, f"### {target_section_id}", "1.1.1 ความหมาย", "ความหมายโครงสร้างข้อมูล"]

section_docs = []
other_docs = []
for d in page1_chunks:
    meta = d.metadata or {}
    check = " ".join([
        str(meta.get("H1", "")), str(meta.get("H2", "")), str(meta.get("H3", "")),
        str(meta.get("section", "")), str(meta.get("best_topic_id", "")),
        str(d.page_content[:200]),
    ])
    has_section = any(p in check for p in patterns)
    if has_section:
        section_docs.append(d)
    else:
        other_docs.append(d)

print(f"Section docs (1.1.1): {len(section_docs)}")
for d in section_docs:
    meta = d.metadata or {}
    print(f"  - {meta.get('chunk_id')}: {meta.get('H3', 'N/A')}")

print(f"\nOther docs: {len(other_docs)}")
for d in other_docs[:3]:
    meta = d.metadata or {}
    print(f"  - {meta.get('chunk_id')}: {meta.get('H3', 'N/A')}")

print("\n=== VERIFICATION ===")
if section_docs:
    print("✅ Section filtering works! Chunks with 1.1.1 are detected.")
else:
    print("❌ Section filtering NOT working! No chunks with 1.1.1 detected.")
