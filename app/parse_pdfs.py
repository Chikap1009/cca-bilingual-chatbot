import json, os, re, uuid, glob
import fitz  # pymupdf
from tqdm import tqdm

RAW_DIR = "data_raw_pdfs"
OUT_PATH = "data_clean/chunks.jsonl"
os.makedirs("data_clean", exist_ok=True)

# ---------------------------------------------------------
# Metadata mapping for all known PDFs
# ---------------------------------------------------------
PDF_METADATA = {
    "Centre for Protected Cultivation Technology": {"title": "Centre for Protected Cultivation Technology (India) – Brochure", "org": "ICAR-IARI, NHB", "year": "2019"},
    "Good Agricultural Practices (GAP)": {"title": "Good Agricultural Practices for IPM in Protected Cultivation", "org": "National Centre for Integrated Pest Management (ICAR)", "year": "2010"},
    "Low Cost Green Houses": {"title": "Low-Cost Greenhouses for Vegetable Production", "org": "Tamil Nadu Agricultural University (TNAU)", "year": "2020"},
    "Protected Cultivation & Post-Harvest": {"title": "Protected Cultivation & Post-Harvest Technology", "org": "Tamil Nadu Agricultural University (TNAU)", "year": "2020"},
    "Protected Cultivation (Methods & Techniques)": {"title": "Protected Cultivation (Methods & Techniques)", "org": "University Academic Reference", "year": "2018"},
    "Protected Cultivation of Horticultural Crops": {"title": "Protected Cultivation of Horticultural Crops", "org": "NABARD", "year": "2020"},
    "Hydroponic Farming": {"title": "Study on Hydroponic Farming", "org": "IJFRM", "year": "2022"},
    "AIoT-based Hydroponic System": {"title": "AIoT-Based Hydroponic System for Crop Recommendation", "org": "IoT & Agriculture Domain", "year": "2021"},
    "Climate-smart Agriculture": {"title": "Climate-Smart Agriculture: Strategies for Resilient Farming Systems", "org": "IJRA", "year": "2022"},
    "Evaluating Climate-Smart Agriculture": {"title": "Evaluating Climate-Smart Agriculture Effects on Productivity and Sustainability", "org": "IJIR", "year": "2023"},
    "FAO_GAP_Greenhouse_Veg_Crops": {"title": "Good Agricultural Practices for Greenhouse Vegetable Crops", "org": "FAO", "year": "2013"},
    "Hydroponics — Sustainable Farming": {"title": "Hydroponics: Sustainable Farming", "org": "IJSAT", "year": "2021"},
    "Hydroponics as an Advanced Technique": {"title": "Hydroponics as an Advanced Technique for Vegetable Production", "org": "Journal of Soil and Water Conservation", "year": "2022"},
    "Hydroponics Exploring": {"title": "Exploring Innovative Sustainable Hydroponic Technologies", "org": "Heliyon (Elsevier)", "year": "2021"},
    "NIH Published Paper": {"title": "Hydroponic System: Nutrient Management Research", "org": "National Institute of Horticulture (NIH)", "year": "2019"},
    "Optimizing Hydroponic Systems": {"title": "Optimizing Hydroponic Systems for High-Density Cultivation", "org": "IJAN", "year": "2022"},
    "Protected cultivation of vegetable crops": {"title": "Protected Cultivation of Vegetable Crops for Sustainable Food Production", "org": "ICAR IndHort", "year": "2023"},
    "Research on Hydroponics Farming": {"title": "Research on Hydroponics Farming", "org": "IJRPR", "year": "2022"},
}

def clean_text(t):
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def chunk_text(text, chunk_tokens=800, overlap_tokens=150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + chunk_tokens)
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap_tokens)
    return chunks

def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for pno in range(len(doc)):
        txt = doc.load_page(pno).get_text("text")
        if txt:
            pages.append(txt)
    return "\n".join(pages)

def get_metadata_for_pdf(fname):
    """Return metadata dict for a given PDF filename based on partial match."""
    for key, meta in PDF_METADATA.items():
        if key.lower() in fname.lower():
            return meta
    return {"title": fname, "org": "UNKNOWN", "year": "NA"}

def main():
    with open(OUT_PATH, "w", encoding="utf-8") as w:
        for pdf in tqdm(glob.glob(os.path.join(RAW_DIR, "*.pdf"))):
            try:
                full = clean_text(extract_pdf(pdf))
                if not full or len(full) < 400:
                    continue

                fname = os.path.basename(pdf)
                meta = get_metadata_for_pdf(fname)
                meta["file"] = fname
                meta["lang"] = "en"

                for ch in chunk_text(full):
                    rec = {"id": str(uuid.uuid4()), "text": ch, "meta": meta}
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

            except Exception as e:
                print("❌ Failed:", pdf, e)

if __name__ == "__main__":
    main()