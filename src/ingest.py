import fitz

PDF_PATH = r"data\raw\Database Management Systems.pdf"

doc = fitz.open(PDF_PATH)

pages = []

for page_num, page in enumerate(doc, start=1):
    text = page.get_text()

    if text.strip():
        pages.append(
            {
                "page": page_num,
                "text": text
            }
        )

print(f"Pages extracted: {len(pages)}")

print("\nFirst page preview:\n")
print(pages[0]["text"][:500])