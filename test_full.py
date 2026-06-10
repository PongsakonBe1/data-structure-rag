import pickle
from pathlib import Path

# Load BM25 index
data = pickle.loads(Path('indexes/bm25_index.pkl').read_bytes())
chunks = data['chunks']

output = []
output.append("=== ทดสอบการเลือก Context Docs ===\n\n")

# จำลองการทำงานของ select_context_docs_diverse
target_section_id = "1.1.1"
context_doc_limit = 12
min_unique_pages = 2

# 1. หา chunks ทั้งหมดจาก page 1-3 (simulating retrieved docs)
retrieved_docs = [c for c in chunks if int(c.metadata.get('page', 0)) <= 3]
output.append(f"Retrieved docs (page 1-3): {len(retrieved_docs)}\n")

# 2. ทดสอบการจัดลำดับตาม section
patterns = [target_section_id, "1.1.1 ความหมาย"]

def _has_section(d):
    meta = d.metadata or {}
    check = f"{meta.get('H3', '')} {meta.get('section', '')} {d.page_content[:200]}"
    return any(p in check for p in patterns)

section_docs = [d for d in retrieved_docs if _has_section(d)]
other_docs = [d for d in retrieved_docs if not _has_section(d)]

output.append(f"\nSection 1.1.1 docs: {len(section_docs)}\n")
for d in section_docs:
    meta = d.metadata
    output.append(f"  - {meta.get('chunk_id')}: page {meta.get('page')}, H3={meta.get('H3')}\n")

output.append(f"\nOther docs: {len(other_docs)}\n")
for d in other_docs[:5]:
    meta = d.metadata
    output.append(f"  - {meta.get('chunk_id')}: page {meta.get('page')}, H3={meta.get('H3')}\n")

# 3. จำลองการเลือกแบบ diversity (แบบเดิม - ไม่มี prioritize)
output.append("\n=== แบบเดิม (ไม่ prioritize) ===\n")

def _doc_key(d):
    meta = d.metadata
    return f"{meta.get('source')}:{meta.get('page')}:{meta.get('chunk_id')}"

# Old way: no reordering
docs_old = retrieved_docs[:context_doc_limit]
output.append(f"Selected {len(docs_old)} docs:\n")
for d in docs_old:
    meta = d.metadata
    output.append(f"  - {meta.get('chunk_id')}: page {meta.get('page')}, H3={meta.get('H3')}\n")

# 4. จำลองการเลือกแบบใหม่ (prioritize section)
output.append("\n=== แบบใหม่ (prioritize section 1.1.1) ===\n")
docs_new = (section_docs + other_docs)[:context_doc_limit]
output.append(f"Selected {len(docs_new)} docs:\n")
for d in docs_new:
    meta = d.metadata
    output.append(f"  - {meta.get('chunk_id')}: page {meta.get('page')}, H3={meta.get('H3')}\n")

# 5. ตรวจสอบ content
output.append("\n=== Content ของ chunk-00002 (1.1.1) ===\n")
chunk_111 = [c for c in chunks if c.metadata.get('chunk_id') == 'chunk-00002'][0]
output.append(chunk_111.page_content[:500])
output.append("\n...\n")

# Write to file
Path('test_full_result.txt').write_text(''.join(output), encoding='utf-8')
print("Done! Check test_full_result.txt")
