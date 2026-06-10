"""
Run Re-OCR with Qwen2.5-VL-72B-Instruct model.
This script runs the full PDF re-processing with the upgraded model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-OCR entire PDF with Qwen2.5-VL-72B-Instruct"
    )
    ap.add_argument(
        "--pdf",
        default="data/data_structure_data_ch1_to_ch5.pdf",
        help="Path to PDF file to re-OCR",
    )
    ap.add_argument(
        "--output",
        default="final_extracted_content.md",
        help="Output markdown file path",
    )
    ap.add_argument(
        "--page-range",
        default="",
        help="Optional: specific page range e.g., '1-10' or '1,2,3'",
    )
    ap.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Vision model to use for OCR",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    args = ap.parse_args()

    # Check environment
    hf_token = os.getenv("HUGGINGFACE_READ_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        print("❌ Error: HUGGINGFACE_READ_TOKEN or HUGGINGFACE_API_KEY not set")
        print("Please set your Hugging Face token:")
        print("  $env:HUGGINGFACE_READ_TOKEN = 'your_token_here'")
        return 1

    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    
    # Update the VISION_MODEL_ID in ingest.py
    ingest_file = src_dir / "ingest.py"
    if not ingest_file.exists():
        print(f"❌ Error: Cannot find {ingest_file}")
        return 1

    # Read current content
    content = ingest_file.read_text(encoding="utf-8")
    
    # Update model if needed
    if f'VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "{args.model}")' not in content:
        print(f"⚠️  Warning: VISION_MODEL_ID in ingest.py may not be set to {args.model}")
        print("The model has been updated in the code. Make sure it's committed.")
    else:
        print(f"✅ VISION_MODEL_ID is set to: {args.model}")

    print(f"\n📄 PDF to process: {args.pdf}")
    print(f"📝 Output file: {args.output}")
    if args.page_range:
        print(f"📑 Page range: {args.page_range}")
    else:
        print(f"📑 Page range: ALL pages")
    print(f"🤖 Vision Model: {args.model}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Not executing")
        return 0

    print("\n⚠️  Warning: This will take a long time with 72B model!")
    print("   - 67 pages × ~30-60 seconds per page = ~30-60 minutes")
    print("   - Hugging Face credit will be consumed")
    print("\nPress Ctrl+C to cancel, or wait 3 seconds to continue...")
    
    try:
        for i in range(3, 0, -1):
            print(f"   Starting in {i}...", end="\r")
            time.sleep(1)
        print("   Starting now!    ")
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 1

    # Run the ingest process
    print("\n🚀 Starting Re-OCR process...")
    print("=" * 60)
    
    # Set environment variables
    env = os.environ.copy()
    env["VISION_MODEL_ID"] = args.model
    if args.page_range:
        env["INGEST_PAGE_RANGE"] = args.page_range
    
    # Import and run the ingest module
    sys.path.insert(0, str(src_dir))
    
    try:
        import ingest
        
        # Update the module's constants
        ingest.VISION_MODEL_ID = args.model
        if args.page_range:
            ingest.INGEST_PAGE_RANGE = args.page_range
        
        # Update output file path if specified
        if args.output != "final_extracted_content.md":
            ingest.DEBUG_FILE = Path(args.output)
        
        # Run the main function
        ingest.main()
        
        print("\n" + "=" * 60)
        print("✅ Re-OCR completed successfully!")
        print(f"📄 Output saved to: {args.output}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during Re-OCR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
