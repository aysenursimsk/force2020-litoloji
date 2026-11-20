"""Projenin tüm sabitleri tek yerde: yollar, veri adresleri, özellikler, sınıf adları."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "sonuclar"

_BASE = ("https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/"
         "master/lithology_competition/data/")
URLS = {
    "train.zip": _BASE + "train.zip",          # 98 kuyu, ~1,17M satır (eğitim + doğrulama)
    "hidden_test.csv": _BASE + "hidden_test.csv",  # 10 ayrı kuyu, etiketli resmi test seti
}

SEED = 42
TARGET = "FORCE_2020_LITHOFACIES_LITHOLOGY"

# %70'ten fazla eksik loglar (DTS, SGR, RMIC, ROPA, DCAL, MUDWEIGHT, RXO) ve konum/formasyon bilgisi kullanılmıyor.
FEATURES = ["DEPTH_MD", "CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR",
            "NPHI", "PEF", "DTC", "SP", "BS", "ROP", "DRHO"]
RESISTIVITY = ["RSHA", "RMED", "RDEP"]   # log10 dönüşümü uygulanır

# Fiziksel olarak mümkün aralıklar. Dışındaki değerler ölçüm hatasıdır → NaN (ölçülmemiş) sayılır.
PHYSICAL_BOUNDS = {
    "CALI": (3, 30),        # kuyu çapı, inç
    "RHOB": (1.0, 3.2),     # yoğunluk, g/cm³ (kömür ~1,2 … anhidrit ~3,0)
    "NPHI": (0.0, 1.0),     # nötron gözenekliliği, oran
    "DTC":  (40, 240),      # sonik, µs/ft (dolomit ~43 … su ~189)
    "DRHO": (-1.0, 1.0),    # yoğunluk düzeltmesi, g/cm³
    "GR":   (0, 1000),      # gamma ışını, API
}

LITHOLOGY = {
    30000: "Kumtaşı", 65030: "Kumtaşı/Şeyl", 65000: "Şeyl", 80000: "Marn",
    74000: "Dolomit", 70000: "Kireçtaşı", 70032: "Tebeşir", 88000: "Halit",
    86000: "Anhidrit", 99000: "Tüf", 90000: "Kömür", 93000: "Temel kaya",
}
CLASS_CODES = sorted(LITHOLOGY)                       # model çıktısı 0..11 bu sırayla
CLASS_NAMES = [LITHOLOGY[c] for c in CLASS_CODES]
N_CLASSES = len(CLASS_CODES)

N_VAL_WELLS = 14   # eğitim dosyasındaki 98 kuyudan doğrulamaya ayrılan kuyu sayısı
