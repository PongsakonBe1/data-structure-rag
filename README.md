# 🤖 RAG Chatbot for Data Structures Course

ระบบถาม-ตอบอัตโนมัติ (Chatbot) สำหรับรายวิชาโครงสร้างข้อมูล โดยใช้เทคนิค **RAG (Retrieval-Augmented Generation)** — ค้นหาข้อมูลจากเอกสารตำราเรียนก่อน แล้วจึงสร้างคำตอบ พร้อมนโยบาย Grounded Only เพื่อป้องกันการตอบมั่ว (Hallucination)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## � กรอบแนวคิดการวิจัย (Conceptual Framework)

```
┌─────────────────────────────────────────────┐
│  ตัวแปรต้น (Independent Variables)           │
│                                             │
│  • ระบบ RAG                                  │
│    (Retrieval-Augmented Generation)          │
│  • เทคนิคการค้นหา:                           │
│    BM25 + FAISS + RRF + Cross-Encoder       │
│  • โมเดลสร้างคำตอบ:                          │
│    Qwen3-4B-Instruct                        │
│  • นโยบาย Grounded Only:                     │
│    ตอบเฉพาะจากหลักฐาน / ปฏิเสธเมื่อไม่แน่ใจ  │
└──────────────────┬──────────────────────────┘
                   │ ระบบที่พัฒนา
                   ▼
┌─────────────────────────────────────────────┐
│  กระบวนการทดสอบ (Process)                    │
│                                             │
│  • คำถาม In-Scope (45 ข้อ/คน)               │
│  • คำถาม Out-of-Scope (45 ข้อ/คน)           │
│         ↓                                   │
│  ระบบ RAG ประมวลผลคำถาม                      │
│         ↓                                   │
│  ผู้เชี่ยวชาญ 3 คน ประเมินผล                  │
│  (90 ข้อ × 3 = 270 ข้อ)                     │
└──────────────────┬──────────────────────────┘
                   │ ผลประเมิน
                   ▼
┌─────────────────────────────────────────────┐
│  ตัวแปรตาม (Dependent Variables)             │
│                                             │
│  Confusion Matrix (TP, TN, FP, FN)          │
│         ↓                                   │
│  Accuracy = (TP + TN) / Total               │
└─────────────────────────────────────────────┘
```

---

## �📋 คุณสมบัติหลัก

- **🔍 Hybrid Retrieval:** BM25 (ค้นตามคำ) + FAISS (ค้นตามความหมาย) + RRF Fusion
- **🏆 Reranking:** Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
- **🖼️ Visual Evidence:** แสดงภาพหน้าเอกสารประกอบคำตอบ
- **📚 Citation Required:** ทุกคำตอบมีการอ้างอิง [หน้า X]
- **🛡️ Grounded Only Policy:** ปฏิเสธคำถามที่ไม่มีในเอกสาร (Abstain)
- **📊 Expert Evaluation:** ระบบประเมินด้วยผู้เชี่ยวชาญ 3 คน + Confusion Matrix

---

## 📊 ผลการประเมิน (Expert Evaluation)

จากการประเมิน **270 ข้อ** (90 ข้อ × 3 ผู้เชี่ยวชาญ):

| ผู้ประเมิน | จำนวนคำถาม | TP | TN | FP | FN | Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **พงศกร** | 90 | 41 | 45 | 0 | 4 | **95.6%** |
| **เพียงธาร** | 90 | 40 | 44 | 1 | 5 | **93.3%** |
| **ธิติกา** | 90 | 31 | 41 | 4 | 14 | **80.0%** |
| **📊 รวมทุกคน** | **270** | **112** | **130** | **5** | **23** | **89.6%** |

### เกณฑ์การให้คะแนน

| คะแนน | ประเภทคำถาม | พฤติกรรมของระบบ | ผลลัพธ์ |
|:---:|---|---|---|
| 1 | In-Scope | ตอบได้ถูกต้อง | ✅ TP |
| 2 | In-Scope | ตอบไม่ได้ / ตอบไม่สมบูรณ์ | ❌ FN |
| 3 | Out-of-Scope | ตอบไปทั้งที่ไม่ควรตอบ | ❌ FP (Hallucination) |
| 4 | Out-of-Scope | ปฏิเสธการตอบถูกต้อง | ✅ TN |

---

## 🔬 เทคโนโลยีที่ใช้

| ขั้นตอน | หน้าที่ | เทคโนโลยี |
|---------|--------|-----------|
| เตรียมข้อมูล | อ่าน PDF แปลงเป็นข้อความ | Qwen2.5-VL-72B-Instruct (AI OCR) |
| สร้างดัชนี (Dense) | ค้นตามความหมาย | BAAI/bge-m3 + FAISS |
| สร้างดัชนี (Sparse) | ค้นตามคำ | BM25 + pythainlp (ตัดคำไทย) |
| รวมผลค้นหา | รวมคะแนนจาก 2 วิธี | Reciprocal Rank Fusion (RRF) |
| จัดอันดับใหม่ | เลือกข้อมูลที่ตรงที่สุด | BAAI/bge-reranker-v2-m3 |
| สร้างคำตอบ | AI สร้างคำตอบภาษาไทย | Qwen3-4B-Instruct (HuggingFace API) |
| หน้าเว็บ | แสดงผลให้ผู้ใช้ | Streamlit |

---

## 🚀 การ Deploy บน Streamlit Cloud

### ขั้นตอนที่ 1: Push ขึ้น GitHub

```bash
git init
git add .
git commit -m "Initial commit: RAG Chatbot for Data Structures"
git remote add origin https://github.com/YOUR_USERNAME/data-structure-rag.git
git push -u origin main
```

### ขั้นตอนที่ 2: Deploy บน Streamlit Cloud

1. ไปที่ [share.streamlit.io](https://share.streamlit.io)
2. Sign in ด้วย GitHub account
3. Click "New app"
4. เลือก Repository: `YOUR_USERNAME/data-structure-rag`
5. Branch: `main`
6. Main file path: `src/app.py`
7. Click "Deploy"

### ขั้นตอนที่ 3: ตั้งค่า Secrets

ใน Streamlit Cloud → Settings → Secrets:

```toml
[huggingface]
hf_token = "YOUR_HF_TOKEN"

[general]
environment = "production"
```

---

## 🛠️ การติดตั้งในเครื่อง (Local Development)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/data-structure-rag.git
cd data-structure-rag

# 2. Create virtual environment
python -m venv venv

# 3. Activate (Windows)
venv\Scripts\activate
# หรือ (Mac/Linux)
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file
echo "HF_TOKEN=your_hf_token_here" > .env

# 6. Run
streamlit run src/app.py
```

---

## 📁 โครงสร้างโปรเจกต์

```
data_structure_rag/
├── 📁 data/                 # ข้อมูลต้นฉบับ (PDF ตำราเรียน 5 บท, 66 หน้า)
├── 📁 docs/                 # เอกสารวิจัย
├── 📁 eval/                 # ชุดคำถาม OOS (Out-of-Scope)
├── 📁 indexes/              # Index files (FAISS, BM25)
├── 📁 logs/                 # Logs และผลการประเมินผู้เชี่ยวชาญ
├── 📁 scripts/              # Scripts (auto_gen_expert_eval.py)
├── 📁 src/                  # Source code
│   ├── app.py              # 🎯 Main Streamlit App (RAG + UI + Evaluation)
│   └── retriever.py        # Retrieval pipeline
├── 📄 .gitignore
├── 📄 requirements.txt     # Dependencies
└── 📄 README.md            # ไฟล์นี้
```

---

## 👨‍💻 ผู้พัฒนา

**นายพงศกร ระวังวงศ์**  
มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)  
รายวิชา: โครงสร้างข้อมูล (Data Structures)

## 📄 License

MIT License - สำหรับการศึกษาและวิจัย

---

💡 **หมายเหตุ:** ไฟล์ PDF ตำรา และ Index files มีขนาดใหญ่ จึงไม่รวมใน repository นี้ ต้องเตรียมเองตามคำแนะนำในเอกสาร
