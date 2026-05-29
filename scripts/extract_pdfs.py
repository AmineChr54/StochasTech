from pathlib import Path
from pypdf import PdfReader

BASE = Path(__file__).resolve().parents[1]
pdfs = [BASE / "docs" / "zezwa pipeline.pdf", BASE / "docs" / "zezwa math modeling.pdf"]

for pdf in pdfs:
    out = BASE / "docs" / (pdf.stem.replace(' ', '_') + ".txt")
    try:
        reader = PdfReader(str(pdf))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        text = "\n\n".join(text_parts)
        out.write_text(text, encoding="utf-8")
        print(f"WROTE: {out}")
    except Exception as e:
        print(f"ERROR processing {pdf}: {e}")
