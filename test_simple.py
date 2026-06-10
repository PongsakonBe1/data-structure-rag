import pickle
from pathlib import Path

# Load BM25 index
data = pickle.loads(Path('indexes/bm25_index.pkl').read_bytes())
chunks = data['chunks']

output = []
output.append(f"Total chunks: {len(chunks)}\n")

# Show chunks from page 1
page1_chunks = [c for c in chunks if c.metadata.get('page') == '1']
output.append(f"Chunks on page 1: {len(page1_chunks)}\n")

for i, doc in enumerate(page1_chunks[:5]):
    meta = doc.metadata or {}
    output.append(f"Chunk {i}: {meta.get('chunk_id')}, H3={meta.get('H3')}\n")

# Test section matching
target = "1.1.1"
patterns = [target, "1.1.1 ความหมาย"]
section_docs = []
for d in page1_chunks:
    meta = d.metadata or {}
    check = f"{meta.get('H3', '')} {meta.get('section', '')} {d.page_content[:100]}"
    if any(p in check for p in patterns):
        section_docs.append(d)

output.append(f"\nSection 1.1.1 docs: {len(section_docs)}\n")
for d in section_docs:
    meta = d.metadata or {}
    output.append(f"  - {meta.get('chunk_id')}: {meta.get('H3')}\n")

# Write to file
Path('test_result.txt').write_text(''.join(output), encoding='utf-8')
print("Done! Check test_result.txt")
