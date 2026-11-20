"""Değerlendirme, grafikler ve deney kaydı."""
import csv
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

from . import config as C

_LABELS = np.arange(C.N_CLASSES)


def skorlar(y_true, y_pred) -> dict:
    """Ana metrik macro F1 (dengesiz veride accuracy yanıltıcı)."""
    return {"acc": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4)}


def rapor(y_true, y_pred):
    print(classification_report(y_true, y_pred, labels=_LABELS, target_names=C.CLASS_NAMES,
                                zero_division=0))


def confusion(y_true, y_pred, title: str = ""):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=_LABELS, display_labels=C.CLASS_NAMES, normalize="true",
        xticks_rotation=90, values_format=".2f", cmap="Blues", colorbar=False)
    disp.figure_.set_size_inches(9, 8)
    disp.ax_.set_title(title or "Normalize confusion matrix (satır = gerçek sınıf)")
    plt.tight_layout(); plt.show()


def ogrenme_egrisi(history, title: str = ""):
    """Train ve validation eğrileri yan yana. Aradaki fark açılıyorsa: overfitting."""
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for i, k in enumerate(("loss", "accuracy")):
        ax[i].plot(h[k], label="train"); ax[i].plot(h["val_" + k], label="validation")
        ax[i].set_title(f"{title} — {k}"); ax[i].set_xlabel("epoch"); ax[i].legend()
    plt.tight_layout(); plt.show()


def kuyu_grafigi(d, well: str, y_pred=None, logs=("GR", "RHOB", "NPHI", "RDEP")):
    """Sektörün standart görseli: bir kuyunun logları derinlik boyunca + gerçek (ve varsa tahmin)
    litoloji sütunu. d = temizlenmiş tablo (veri.temizle çıktısı veya Veri.tr/va/te).
    y_pred verilirse d ile aynı satır sırasında olmalı."""
    from matplotlib.colors import ListedColormap
    mask = (d.WELL == well).to_numpy()
    w = d[mask]
    depth = w.DEPTH_MD.to_numpy()
    cols = [w.y.to_numpy()] + ([np.asarray(y_pred)[mask]] if y_pred is not None else [])
    titles = ["Gerçek"] + (["Tahmin"] if y_pred is not None else [])
    cmap = ListedColormap(plt.get_cmap("tab20").colors[:C.N_CLASSES])

    n = len(logs) + len(cols)
    fig, ax = plt.subplots(1, n, figsize=(2.2 * n, 9), sharey=True)
    for i, lg in enumerate(logs):
        ax[i].plot(w[lg], depth, lw=0.6)
        ax[i].set_title(lg + (" (log10)" if lg in C.RESISTIVITY else ""))
        ax[i].grid(alpha=0.3)
    for j, (yy, t) in enumerate(zip(cols, titles)):
        a = ax[len(logs) + j]
        a.imshow(yy[:, None], aspect="auto", cmap=cmap, vmin=0, vmax=C.N_CLASSES - 1,
                 extent=[0, 1, depth.max(), depth.min()], interpolation="nearest")
        a.set_title(t); a.set_xticks([])
    ax[0].set_ylim(depth.max(), depth.min()); ax[0].set_ylabel("Derinlik (m)")   # sığ üstte
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(C.N_CLASSES)]
    fig.legend(handles, C.CLASS_NAMES, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(f"Kuyu {well}"); plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()


def kaydet(deney: str, model: str, ayarlar: dict, skor: dict, kim: str = "", not_: str = "",
           epochs: int | None = None):
    """Sonucu sonuclar/deneyler_<kim>.csv'ye bir satır olarak ekler. Herkesin ayrı dosyası olduğu
    için git'te birleştirme çakışması çıkmaz; hepsini birlikte görmek için tum_sonuclar()."""
    path = C.RESULTS_DIR / f"deneyler_{(kim or 'anonim').lower().replace(' ', '_')}.csv"
    path.parent.mkdir(exist_ok=True)
    yeni = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if yeni:
            w.writerow(["tarih", "kim", "deney", "model", "ayarlar", "epoch",
                        "acc", "macro_f1", "weighted_f1", "not"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), kim, deney, model,
                    json.dumps(ayarlar, ensure_ascii=False), epochs or "",
                    skor["acc"], skor["macro_f1"], skor["weighted_f1"], not_])
    print("Kaydedildi →", path.relative_to(C.ROOT), skor)


def tum_sonuclar():
    """Ekibin bütün deney dosyalarını tek tabloda birleştirir (en iyi macro F1 üstte)."""
    import pandas as pd
    files = sorted(C.RESULTS_DIR.glob("deneyler_*.csv"))
    if not files:
        print("Henüz kayıtlı deney yok."); return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files]).sort_values("macro_f1", ascending=False) \
             .reset_index(drop=True)
