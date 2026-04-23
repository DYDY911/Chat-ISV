# pdf_ie_with_checkpoint.py
import json
import time
import re
from pathlib import Path
from typing import List, Set
import PyPDF2
from openai import OpenAI

# ========== 基础配置 ==========
SCRIPT_DIR = Path(__file__).parent

# 读取配置
with open(SCRIPT_DIR / 'config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

INPUT_DIR = Path(config['input_dir'])             # PDF 输入目录
OUTPUT_JSONL = Path(config['output_jsonl'])       # 输出 jsonl 文件
CHUNK_CHARS = int(config.get('chunk_chars', 2000))
MAX_RETRY = int(config.get('max_retry', 6))
TEMP = float(config.get('temperature', 0.2))
CLEAR_PROGRESS = bool(config.get('clear_progress_on_complete', True))

# 读取 Prompt 模板
with open(SCRIPT_DIR / "prompts.py", "r", encoding="utf-8") as f:
    PROMPT_DISCOVERY = f.read()


# 初始化 OpenAI 客户端
client = OpenAI(api_key=config['api_key'], base_url=config['api_base'])

# ========== 工具函数 ==========
def extract_text_from_pdf(pdf_path: Path) -> str:
    """从 PDF 提取纯文本；失败返回空串"""
    try:
        text = []
        with pdf_path.open('rb') as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            print(f"  PDF共 {total} 页")
            for i in range(total):
                page = reader.pages[i]
                # PyPDF2 的 extract_text 可能返回 None
                page_txt = page.extract_text() or ""
                text.append(page_txt)
        return ("\n".join(text)).strip()
    except Exception as e:
        print(f"  PDF读取错误: {e}")
        return ""

def chunk_text(text: str, chunk_size: int) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$", re.DOTALL)

def _strip_code_fences(s: str) -> str:
    """去掉可能的 ```json / ```cypher 等代码块围栏"""
    return _CODE_FENCE_RE.sub("", s).strip()

def call_llm(prompt: str, max_retry: int = MAX_RETRY) -> str | None:
    """调用 LLM；确保返回为可解析的 JSON 字符串"""
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=config['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP
            )
            result = (resp.choices[0].message.content or "").strip()
            # 去围栏，常见导致 “Invalid control character” 的源头
            result = _strip_code_fences(result)
            # 尝试解析 JSON；若失败会进入 except 重试
            json.loads(result)
            return result
        except Exception as e:
            print(f"尝试 {attempt}/{max_retry} 失败: {e}")
            if attempt < max_retry:
                time.sleep(2)
    return None

def load_progress(progress_path: Path) -> Set[str]:
    """读取断点文件（存储已完成的 doc_id 集合）"""
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
    print(f"处理文件: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"  文件 {pdf_path.name} 无法提取文本，跳过")
        return

    chunks = chunk_text(text, CHUNK_CHARS)
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        doc_id = f"{pdf_path.stem}_chunk_{idx}"

        if doc_id in processed_chunks:
            print(f"  跳过已处理块: {doc_id}")
            continue

        prompt = PROMPT_DISCOVERY.replace('{DOC_ID}', doc_id).replace('{TEXT}', chunk)
        print(f"  处理块 {idx+1}/{total}...")

        result = call_llm(prompt, MAX_RETRY)
        if result:
            # 实时追加写出
            with OUTPUT_JSONL.open('a', encoding='utf-8') as out:
                out.write(result + '\n')

            processed_chunks.add(doc_id)
            save_progress(SCRIPT_DIR / 'progress.json', processed_chunks)
        else:
            print(f"  块 {idx+1} 提取失败（将保留断点以便重试）")

def main():
    # 准备输出
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL.touch(exist_ok=True)

    # 读取断点
    progress_path = SCRIPT_DIR / 'progress.json'
    processed_chunks = load_progress(progress_path)
    if processed_chunks:
        print(f"✅ 从断点恢复，已处理 {len(processed_chunks)} 个块")

    # 收集 PDF
    pdf_files = sorted(INPUT_DIR.glob('*.pdf'))
    print(f"找到 {len(pdf_files)} 个PDF文件")
    if not pdf_files:
        print("警告: 未找到任何PDF文件!")
        return

    # 逐个处理
    for pdf in pdf_files:
        process_pdf(pdf, processed_chunks)

    # 全部完成后可选清理断点
    if CLEAR_PROGRESS and progress_path.exists():
        try:
            progress_path.unlink()
            print("🧹 已清理 progress.json（全部完成）")
        except Exception as e:
            print(f"清理 progress.json 失败：{e}")

    print("\n✅ 全部完成，结果已保存到:", OUTPUT_JSONL.resolve())

if __name__ == "__main__":
    main()