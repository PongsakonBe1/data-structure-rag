"""
auto_gen_expert_eval.py
สร้าง CSV สำหรับผู้เชี่ยวชาญประเมิน โดยส่งคำถามผ่านระบบ RAG จริง (retrieval + LLM)
แล้ว export ออกมาให้ผู้เชี่ยวชาญกรอก eval_score เอง
"""

import os
import sys
import re
import csv
import time
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from huggingface_hub import InferenceClient
from retriever import RAGSystem

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = (
    os.getenv("HUGGINGFACE_READ_TOKEN")
    or os.getenv("HUGGINGFACE_API_KEY")
    or os.getenv("HF_TOKEN")
    or ""
).strip()
if not HF_TOKEN:
    print("ERROR: ไม่พบ HF_TOKEN ใน .env")
    sys.exit(1)

CHAT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "1400"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "20"))
RETRIEVE_TOP_N = int(os.getenv("RETRIEVE_TOP_N", "8"))
CONTEXT_DOC_LIMIT = int(os.getenv("CONTEXT_DOC_LIMIT", "5"))

EVALUATOR_NAME = "ธิติกา"
OUTPUT_PATH = PROJECT_ROOT / "logs" / f"expert_eva_{EVALUATOR_NAME}.csv"

hf_client = InferenceClient(api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
QUESTIONS = [
    # IS (In-Scope) questions
    "ความหมายของโครงสร้างข้อมูล (Data Structure)",
    "ความหมายของอัลกอริทึม (Algorithm)",
    "บิต (Bit)",
    "ไบต์ (Byte)",
    "ฟิลด์ (Field)",
    "เรคอร์ด (Record)",
    "ไฟล์ (File)",
    "ฐานข้อมูล (Database)",
    "ประเภทโครงสร้างข้อมูล",
    "ประเภทอัลกอริทึม",
    "ผังงาน",
    "รหัสเทียม",
    "โครงสร้างควบคุมแบบเรียงลำดับ (Sequence)",
    "โครงสร้างควบคุมแบบเลือกการทำงาน (Selection)",
    "โครงสร้างควบคุมแบบทำซ้ำ (Repetition)",
    "ความหมายของอาร์เรย์",
    "โครงสร้างของอาร์เรย์",
    "อาร์เรย์ 1 มิติ",
    "อาร์เรย์ 2 มิติ",
    "ความหมายของลิงค์ลิสต์",
    "โครงสร้างของลิงค์ลิสต์",
    "โครงสร้างลิงค์ลิสต์แบบทิศทางเดียว",
    "การทำงานของลิงค์ลิสต์แบบทิศทางเดียว",
    "โครงสร้างคิว",
    "การนำข้อมูลเข้า (Enqueue)",
    "การนำข้อมูลออก (Dequeue)",
    "โครงสร้างของการแทนคิวด้วยอาร์เรย์",
    "การดำเนินการแทนคิวด้วยอาร์เรย์",
    "การดำเนินการแทนคิวด้วยวงกลม",
    "โครงสร้างสแตก (Stack)",
    "การนำข้อมูลเข้า (Push)",
    "การดึงข้อมูลออก (Pop)",
    "ตำแหน่งบนสุด (top)",
    "การดำเนินการแทนสแตกด้วยอาร์เรย์",
    "การประยุกต์ใช้งานสแตกในการเปลงรูปนิพจน์ทางคณิตศาสตร์",
    "โครงสร้างทรี (Tree)",
    "ทรีทั่วไป (General Tree)",
    "ไบนารีทรี (Binary Tree)",
    "ไบนารีทรีแบบสมบูรณ์ (Complete Binary Tree)",
    "แบบพรีออร์เดอร์ (Preorder Traversal)",
    "แบบอินออร์เดอร์ (Inorder Traversal)",
    "แบบโพสต์ออร์เดอร์ (Postorder Traversal)",
    "ทรีกับการดำเนินการด้านนิพจน์ (Expression Tree)",
    "การแทนโครงสร้างไบนารีทรีแบบซีเควนเชียล (Sequential Representation)",
    "การแปลงทรีเป็นไบนารีทรี",
    # OOS (Out-of-Scope) questions
    "Browser History Stack กับ Forward Stack ใน Chrome",
    "Recursion Stack Frame ใน Memory มี Stack Pointer อย่างไร",
    "Call Stack Trace ในการ Debug โปรแกรม Crash",
    "Min Stack ที่เก็บค่าต่ำสุดใน O(1) เวลา Push/Pop",
    "Stack Overflow Attack ใน Buffer Overflow อันตรายอย่างไร",
    "Git Push ใน Version Control กับ Stack Push ต่างกันอย่างไร",
    "Tower of Hanoi ใช้ Stack ใน Recursive Solution อย่างไร",
    "Stack Segment ใน Process Memory Layout เก็บอะไร",
    "AVL Tree ที่เป็น Self-balancing Binary Search Tree ทำงานอย่างไร",
    "Red-Black Tree ที่ใช้ใน TreeMap และ TreeSet ของ Java",
    "Binary Heap ในการทำ Priority Queue และ Heap Sort",
    "N-ary Tree ที่มี Children มากกว่า 2 โหนด ใช้ทำอะไร",
    "Adjacency Matrix กับ Adjacency List ใน Graph Representation",
    "B-Tree ใน File System (NTFS, ext4) ทำงานอย่างไร",
    "Suffix Tree ใน String Matching และ Bioinformatics",
    "Merkle Tree ใน Blockchain ใช้ตรวจสอบ Integrity อย่างไร",
    "Huffman Tree ในการบีบอัดข้อมูล (Data Compression)",
    "Leaf Node ใน Decision Tree ของ Machine Learning",
    "Neural Network ใน Deep Learning มี Layers อะไรบ้าง",
    "Index ใน Relational Database (B+ Tree Index) ทำงานอย่างไร",
    "DOM Tree ใน HTML Document Object Model คืออะไร",
    "Greedy Algorithm กับ Dynamic Programming ต่างกันอย่างไร",
    "Rust Programming กับ Ownership และ Borrowing",
    "Disjoint Set Union (Union-Find) ใน Graph Algorithm",
    "Spatial Data Structure ใน Computer Graphics (Quadtree)",
    "Topological Sort ใน Directed Acyclic Graph (DAG)",
    "Distributed Queue ใน Message Broker เช่น RabbitMQ",
    "Backtracking Algorithm ในการแก้ N-Queens Problem",
    "Fenwick Tree (Binary Indexed Tree) ใน Range Query",
    "Abstract Syntax Tree (AST) ใน Compiler คืออะไร",
    "Molecular Structure ในเคมีกับ Data Structure ต่างกันอย่างไร",
    "Kadane Algorithm ในการหา Maximum Subarray Sum",
    "Data Structure ที่ใช้ใน High-frequency Trading",
    "Kotlin Coroutines กับ Channel ใน Asynchronous Programming",
    "Explicit Stack ในการแปลง Recursive เป็น Iterative",
    "Genealogy Tree ในการศึกษาพันธุกรรมครอบครัว",
    "Hash Function ใน Cryptography (SHA-256) ต่างจาก Hash Table",
    "Array Formula ใน Excel/Spreadsheet ทำงานอย่างไร",
    "Event-driven Architecture กับ Event Queue ในระบบ Real-time",
    "Undo/Redo Stack ในการ Implement ฟีเจอร์ Undo ของแอพ",
    "Cartesian Tree ในการแก้ Range Minimum Query",
    "WebAssembly (Wasm) กับ Stack-based VM ทำงานอย่างไร",
    "Bitwise Operations ในการ Optimize โปรแกรม (Bitmask DP)",
    "I/O Scheduling Queue ใน Disk Scheduling Algorithms",
    "Shunting Yard Algorithm ของ Dijkstra ในการ Parse Expression",
]


# ---------------------------------------------------------------------------
# RAG Pipeline (simplified non-streaming version)
# ---------------------------------------------------------------------------
def call_hf_api_sync(messages, max_tokens=None, temperature=0.2):
    """Non-streaming HF API call."""
    token_budget = max_tokens or CHAT_MAX_TOKENS
    try:
        res = hf_client.chat_completion(
            model=CHAT_MODEL_ID,
            messages=messages,
            stream=False,
            max_tokens=token_budget,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except TypeError:
        res = hf_client.chat_completion(
            model=CHAT_MODEL_ID,
            messages=messages,
            stream=False,
            max_tokens=token_budget,
            temperature=temperature,
        )
    if res and getattr(res, "choices", None):
        text = (res.choices[0].message.content or "").strip()
        # Strip thinking blocks
        if "<think>" in text and "</think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text
    return ""


def generate_answer(rag_system, question):
    """Run retrieval + generation for a single question, return answer string."""
    try:
        docs = rag_system.retrieve(question, RETRIEVE_TOP_K, RETRIEVE_TOP_N)
    except Exception as e:
        return f"ERROR: retrieval failed: {e}"

    if not docs:
        return "ABSTAIN: ไม่พบหลักฐานจากเอกสารที่ตรงคำถาม จึงยังไม่สามารถตอบได้อย่างน่าเชื่อถือ"

    # Build context from top docs
    context_parts = []
    for i, d in enumerate(docs[:CONTEXT_DOC_LIMIT]):
        meta = getattr(d, "metadata", {}) or {}
        src = str(meta.get("source", "reference_document"))
        page = str(meta.get("page", "")).strip()
        citation = f"{src}:{page}" if page else src
        chunk_id = str(meta.get("chunk_id", f"chunk-{i+1}")).strip()
        content = (d.page_content or "").strip()
        context_parts.append(f"Context {i+1} [{citation} | {chunk_id}]: {content}")

    context_text = "\n".join(context_parts)

    system_instr = (
        "You are an AI tutor for Data Structure. "
        "Respond in Thai only. "
        "Answer using ONLY evidence from the provided context. "
        "Do not introduce facts from outside the context. "
        "Use the original wording from context as much as possible, preserving the author's language. "
        "Include inline citation [หน้า X] for key claims. "
        "Do NOT generate Mermaid, flowcharts, code blocks, or ASCII diagrams unless explicitly present in context. "
        "If evidence is insufficient, say you do not know. /no_think"
    )

    messages = [
        {"role": "system", "content": system_instr},
        {"role": "user", "content": f"Provided context:\n{context_text}\n\nStudent question: {question}"},
    ]

    answer = call_hf_api_sync(messages, max_tokens=CHAT_MAX_TOKENS, temperature=0.2)
    if not answer:
        return "ERROR: LLM returned empty response"
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading RAG system...")
    rag_system = RAGSystem()
    print(f"RAG loaded. Dense: {rag_system.vectorstore is not None}, Reranker: {rag_system.reranker is not None}")
    print(f"Total questions: {len(QUESTIONS)}")
    print(f"Evaluator: {EVALUATOR_NAME}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q[:60]}...", end=" ", flush=True)
        t0 = time.time()
        answer = generate_answer(rag_system, q)
        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s, {len(answer)} chars)")

        results.append({
            "timestamp": datetime.utcnow().isoformat(),
            "evaluator_name": EVALUATOR_NAME,
            "question": q,
            "answer": answer,
            "eval_score": "",  # ผู้เชี่ยวชาญกรอกเอง
        })

        # Rate limit: avoid hammering API
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)

    # Write CSV
    fieldnames = ["timestamp", "evaluator_name", "question", "answer", "eval_score"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 60)
    print(f"Done! {len(results)} rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
