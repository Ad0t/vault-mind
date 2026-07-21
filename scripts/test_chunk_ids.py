import sys
sys.path.insert(0, "src")
from chunk import chunk_sections

def fake_section(src, text):
    return {
        "type": "text", "start_page": 1, "end_page": 2,
        "source_doc": src, "section_title": "Test", "heading_level": 1,
        "text": text, "bbox": None, "table_index": None, "caption": None,
    }

# Simulate two PDFs processed in separate calls (Gradio UI path)
chunks_a = chunk_sections([fake_section("Database Management Systems.pdf", "Transaction is a unit of work. " * 50)])
chunks_b = chunk_sections([fake_section("Solutions Manual.pdf", "A transaction is any one execution. " * 50)])

ids_a = {c["chunk_id"] for c in chunks_a}
ids_b = {c["chunk_id"] for c in chunks_b}
collision = ids_a & ids_b

print(f"PDF A sample ID : {chunks_a[0]['chunk_id']}  (source: {chunks_a[0]['source_doc']})")
print(f"PDF B sample ID : {chunks_b[0]['chunk_id']}  (source: {chunks_b[0]['source_doc']})")
print(f"Colliding IDs   : {len(collision)} (must be 0)")
print("PASS" if not collision else "FAIL — IDs still collide!")

# Also verify same PDF produces same IDs (deterministic)
chunks_a2 = chunk_sections([fake_section("Database Management Systems.pdf", "Transaction is a unit of work. " * 50)])
ids_a2 = {c["chunk_id"] for c in chunks_a2}
print(f"Same PDF re-ingested produces same IDs: {'YES' if ids_a == ids_a2 else 'NO'}")
