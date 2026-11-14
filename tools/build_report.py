"""Build docs/TECHNICAL_REPORT.pdf from docs/TECHNICAL_REPORT.md (markdown -> HTML -> headless Chrome PDF)."""
import pathlib, subprocess
import markdown
D = pathlib.Path(__file__).resolve().parents[1] / "docs"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6px; } h2 { font-size: 13pt; margin: 16px 0 6px; border-bottom: 1px solid #ddd; padding-bottom: 2px; }
h3 { font-size: 11pt; margin: 12px 0 4px; } li { margin: 1px 0; } h2, h3 { break-after: avoid; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6px 0 10px; font-size: 8.6pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3px 5px; vertical-align: top; text-align: left; } th { background: #f2f2f0; }
pre { background: #f6f6f4; padding: 6px 8px; font-size: 7.4pt; line-height: 1.3; white-space: pre-wrap; page-break-inside: avoid; }
code { font-family: Menlo, monospace; font-size: 8.4pt; } pre code { font-size: 7.4pt; }
img { max-width: 100%; display: block; margin: 6px auto; page-break-inside: avoid; }
img[src*="examples_grid"] { max-height: 225mm; } img[src*="roc_holdout"] { max-height: 95mm; width: auto; }
hr { border: 0; border-top: 1px solid #ddd; }
"""
body = markdown.markdown((D / "TECHNICAL_REPORT.md").read_text(), extensions=["tables", "fenced_code"])
(D / "TECHNICAL_REPORT.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Technical Report</title><style>{CSS}</style></head><body>{body}</body></html>")
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={D / 'TECHNICAL_REPORT.pdf'}", f"file://{D / 'TECHNICAL_REPORT.html'}"], check=True, capture_output=True)
import pymupdf
doc = pymupdf.open(D / "TECHNICAL_REPORT.pdf"); print("pages:", len(doc))
