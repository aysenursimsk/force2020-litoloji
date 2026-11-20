"""Temizleme seçeneklerini aynı XGBoost ile karşılaştırır.

Çalıştırma (repo kökünden):  python -m scripts.temizleme_karsilastir
Hızlı deneme için:           python -m scripts.temizleme_karsilastir --kuyu 30
Tam veride ~6 dk (CPU). Sonuçlar sonuclar/deneyler_<kim>.csv'ye de yazılır.
"""
import argparse
import time

from xgboost import XGBClassifier

from src import config as C, degerlendir as D, veri

JEOLOJIK = [f for f in C.FEATURES if f not in ("ROP", "BS")]   # sondaj parametreleri hariç

VARYANTLAR = {
    "A_temel":            dict(fiziksel=False, doldurma="global"),
    "B_fiziksel_sinir":   dict(fiziksel=True,  doldurma="global"),
    "C_kuyu_medyani":     dict(fiziksel=True,  doldurma="kuyu"),
    "D_rop_bs_yok":       dict(fiziksel=True,  doldurma="global", features=JEOLOJIK),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuyu", type=int, default=None, help="sadece ilk N eğitim kuyusu (hızlı deneme)")
    ap.add_argument("--kim", default="", help="deney kaydında görünecek ad")
    args = ap.parse_args()

    print(f"{'varyant':20s} {'doğrulama F1':>13s} {'test F1':>8s} {'test acc':>9s}")
    for ad, kw in VARYANTLAR.items():
        t = time.time()
        d = veri.hazirla(n_wells=args.kuyu, **kw)
        m = XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.1,
                          tree_method="hist", n_jobs=-1, random_state=C.SEED)
        m.fit(d.X_tr, d.y_tr)
        va, te = D.skorlar(d.y_va, m.predict(d.X_va)), D.skorlar(d.y_te, m.predict(d.X_te))
        print(f"{ad:20s} {va['macro_f1']:13.3f} {te['macro_f1']:8.3f} {te['acc']:9.3f}  ({time.time()-t:.0f} sn)")
        D.kaydet(f"temizleme_{ad}", "XGBoost",
                 {k: (v if k != "features" else "jeolojik") for k, v in kw.items()},
                 te, kim=args.kim, not_=f"doğrulama macro_f1={va['macro_f1']}")


if __name__ == "__main__":
    main()
