# 🤖 RAG Chatbot for Data Structures Course

ระบบแชตบอตอัจฉริยะสำหรับตอบคำถามรายวิชาโครงสร้างข้อมูล โดยใช้ Retrieval-Augmented Generation (RAG) กับ Visual Grounding

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📋 คุณสมบัติหลัก

- **🔍 Hybrid Retrieval:** ColPali (Visual) + SPLADE (Lexical)
- **🖼️ Visual Grounding:** อ้างอิงภาพประกอบจากตำรา
- **📚 Citation Required:** ทุกคำตอบมีแหล่งที่มา
- **🛡️ Abstain Policy:** ปฏิเสธคำถามนอกเนื้อหา
- **📊 IOC Evaluation:** ระบบประเมินคุณภาพด้วย Expert Evaluation

## 🚀 การ Deploy บน Streamlit Cloud

### ขั้นตอนที่ 1: Push ขึ้น GitHub

```bash
# 1. Initialize git
git init

# 2. Add files
git add .

# 3. Commit
git commit -m "Initial commit: RAG Chatbot for Data Structures"

# 4. Add remote (แทนที่ YOUR_USERNAME ด้วย username GitHub ของคุณ)
git remote add origin https://github.com/YOUR_USERNAME/data-structure-rag.git

# 5. Push
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

### ขั้นตอนที่ 3: ตั้งค่า Secrets (สำคัญ!)

ใน Streamlit Cloud, ไปที่:
- Settings → Secrets

เพิ่ม secrets ตามนี้:

```toml
[gemini]
api_key = "YOUR_GEMINI_API_KEY"

[general]
environment = "production"
```

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
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 6. Run
streamlit run src/app.py
```

## 📁 โครงสร้างโปรเจกต์

```
data_structure_rag/
├── 📁 data/                 # ข้อมูลต้นฉบับ (PDF)
├── 📁 docs/                 # เอกสารวิจัย
├── 📁 eval/                 # ชุดข้อมูลทดสอบ
├── 📁 indexes/              # Index files (ไม่ขึ้น GitHub)
├── 📁 logs/                 # Logs และผลการประเมิน
├── 📁 scripts/              # Scripts สำหรับ data preparation
├── 📁 src/                  # Source code
│   ├── app.py              # 🎯 Main Streamlit App
│   └── ...
├── 📄 .gitignore           # ไฟล์ที่ไม่ขึ้น GitHub
├── 📄 requirements.txt     # Dependencies
└── 📄 README.md            # ไฟล์นี้
```

## 📊 ผลการประเมิน IOC

จากการประเมิน **276 รายการ** โดยผู้เชี่ยวชาญ 3 ท่าน:

| Metric | ค่า | สถานะ |
|--------|-----|--------|
| **IOC (Accuracy)** | 87.0% | ✅ ผ่าน (≥70%) |
| **Precision** | 96.4% | ✅ ผ่าน |
| **Recall** | 76.8% | ⚠️ ต่ำกว่าเป้า (80%) |
| **F1-Score** | 85.5% | ✅ |

## 🔬 เทคโนโลยีที่ใช้

- **LLM:** Google Gemini Pro
- **Visual Retrieval:** ColPali (Vision Language Model)
- **Text Retrieval:** SPLADE++ & BM25
- **UI Framework:** Streamlit
- **Embeddings:** Sentence Transformers
- **Vector Store:** FAISS

## 📝 เอกสารอ้างอิง

- [Research Development Documentation](docs/research_development.md)
- [Thesis Report Ch2-Ch5](docs/thesis_report_ch2_ch5_th.md)
- [Visual RAG Workflow](docs/visual_rag_workflow_research_th.md)

## 👨‍💻 ผู้พัฒนา

**พงศกร** - มหาวิทยาลัยขอนแก่น  
รายวิชา: โครงสร้างข้อมูล (Data Structures)

## 📄 License

MIT License - สำหรับการศึกษาและวิจัย

---

💡 **หมายเหตุ:** ไฟล์ PDF ตำรา และ Index files มีขนาดใหญ่ จึงไม่รวมใน repository นี้ ต้องเตรียมเองตามคำแนะนำในเอกสาร
