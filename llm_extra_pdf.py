# pdf_ie_with_checkpoint.py
import json
import time
import re
from pathlib import Path
from typing import List, Set
import PyPDF2
from openai import OpenAI

# ====================
SCRIPT_DIR = Path(__file__).parent


with open(SCRIPT_DIR / 'config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

INPUT_DIR = Path(config['input_dir'])            
OUTPUT_JSONL = Path(config['output_jsonl'])       
CHUNK_CHARS = int(config.get('chunk_chars', 2000))
MAX_RETRY = int(config.get('max_retry', 6))
TEMP = float(config.get('temperature', 0.2))
CLEAR_PROGRESS = bool(config.get('clear_progress_on_complete', True))

with open(SCRIPT_DIR / "prompts.py", "r", encoding="utf-8") as f:
    PROMPT_DISCOVERY = f.read()


client = OpenAI(api_key=config['api_key'], base_url=config['api_base'])

# ===================
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from PDF; return an empty string on failure."""
    try:
        text = []
        with pdf_path.open('rb') as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            print(f" PDF total pages: {total}")
            for i in range(total):
                page = reader.pages[i]
                page_txt = page.extract_text() or ""
                text.append(page_txt)
        return ("\n".join(text)).strip()
    except Exception as e:
        print(f"  Error reading PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$", re.DOTALL)

def _strip_code_fences(s: str) -> str:
    """Remove possible code block fences such as json,cypher, etc."""
    return _CODE_FENCE_RE.sub("", s).strip()

def call_llm(prompt: str, max_retry: int = MAX_RETRY) -> str | None:
    """Call the LLM; ensure the response is a parseable JSON string."""
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP
            )
            result = (resp.choices[0].message.content or "").strip()
            result = _strip_code_fences(result)
            json.loads(result)
            return result
        except Exception as e:
            print(f"try {attempt}/{max_retry} fail: {e}")
            if attempt < max_retry:
                time.sleep(2)
    return None

def load_progress(progress_path: Path) -> Set[str]:
    """Read the checkpoint file (stores completed doc_id set)."""
    if progress_path.exists():
        try:
            data = json.loads(progress_path.read_text(encoding='utf-8'))
            return set(data) if isinstance(data, list) else set()
        except Exception:
            return set()
    return set()

def save_progress(progress_path: Path, processed: Set[str]) -> None:
    progress_path.write_text(
        json.dumps(sorted(processed), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

# ========== 主处理 ==========
def process_pdf(pdf_path: Path, processed_chunks: Set[str]) -> None:
    print(f"settle file: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"  Unable to extract text from file {pdf_path.name}, skipping.")
        return

    chunks = chunk_text(text, CHUNK_CHARS)
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        doc_id = f"{pdf_path.stem}_chunk_{idx}"

        if doc_id in processed_chunks:
            print(f"  Skip already processed block: {doc_id}")
            continue

        prompt = PROMPT_DISCOVERY.replace('{DOC_ID}', doc_id).replace('{TEXT}', chunk)
        print(f"  Processing block {idx+1}/{total}...")

        result = call_llm(prompt, MAX_RETRY)
        if result:
            with OUTPUT_JSONL.open('a', encoding='utf-8') as out:
                out.write(result + '\n')

            processed_chunks.add(doc_id)
            save_progress(SCRIPT_DIR / 'progress.json', processed_chunks)
        else:
            print(f"  Failed to extract block {idx+1} (checkpoint will be kept for retry).")

def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.touch(exist_ok=True)

    progress_path = SCRIPT_DIR / 'progress.json'
    processed_chunks = load_progress(progress_path)
    if processed_chunks:
        print(f"✅ Resume from checkpoint, processed {len(processed_chunks)} blocks.")

    pdf_files = sorted(INPUT_DIR.glob('*.pdf'))
    print(f"find {len(pdf_files)} PDF file")
    if not pdf_files:
        print("Warning: No PDF files found!")
        return

    for pdf in pdf_files:
        process_pdf(pdf, processed_chunks)

    if CLEAR_PROGRESS and progress_path.exists():
        try:
            progress_path.unlink()
            print("🧹 Cleared progress.json (all completed).")
        except Exception as e:
            print(f"Failed to clean progress.json: {e}")

    print("\n✅ All done, results saved to: OUTPUT_JSONL.resolve())

if __name__ == "__main__":
    main()
