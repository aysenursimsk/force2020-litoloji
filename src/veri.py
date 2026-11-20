"""Veri: indirme, yükleme, temizleme, kuyu bazlı bölme, özellik matrisi ve 1D-CNN pencereleri.

Hızlı kullanım:
    from src import veri
    d = veri.hazirla()            # indirir (gerekirse), temizler, böler, ölçekler
    d.X_tr, d.y_tr, d.X_va, d.y_va, d.X_te, d.y_te
"""
from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C


# ---------------------------------------------------------------- indirme / yükleme
def indir(force: bool = False) -> None:
    """train.csv ve hidden_test.csv yoksa GitHub'dan indirir (~120 MB, bir kez)."""
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in C.URLS.items():
        target = C.DATA_DIR / name
        final = C.DATA_DIR / name.replace(".zip", ".csv")
        if final.exists() and not force:
            continue
        print(f"İndiriliyor: {name}")
        urllib.request.urlretrieve(url, target)
        if name.endswith(".zip"):
            with zipfile.ZipFile(target) as z:
                z.extractall(C.DATA_DIR)
            target.unlink()


def yukle(n_wells: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(eğitim, test) DataFrame'leri. n_wells verilirse hızlı deney için ilk n eğitim kuyusu alınır."""
    indir()
    train = pd.read_csv(C.DATA_DIR / "train.csv", sep=";")
    test = pd.read_csv(C.DATA_DIR / "hidden_test.csv", sep=";")
    if n_wells:
        keep = train.WELL.drop_duplicates().iloc[:n_wells]
        train = train[train.WELL.isin(keep)]
    return train, test


# ---------------------------------------------------------------- temizleme
def temizle(df: pd.DataFrame, fiziksel: bool = True) -> pd.DataFrame:
    """1) Gerekli sütunları seçer  2) fiziksel aralık dışı değerleri NaN yapar (fiziksel=True)
    3) sıfır/negatif rezistiviteyi NaN yapar ve log10'a çevirir  4) etiketi 0..11'e kodlar.
    Satırlar kuyu ve derinliğe göre sıralı döner (1D-CNN pencereleri için gerekli)."""
    d = df[["WELL"] + C.FEATURES + [C.TARGET]].copy()
    if fiziksel:
        for c, (lo, hi) in C.PHYSICAL_BOUNDS.items():
            if c in d:
                d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
    for c in C.RESISTIVITY:
        d.loc[d[c] <= 0, c] = np.nan          # negatif/sıfır rezistivite fiziksel değil
        d[c] = np.log10(d[c])
    d["y"] = d[C.TARGET].map({code: i for i, code in enumerate(C.CLASS_CODES)})
    return d.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)


def kuyu_bol(d: pd.DataFrame, n_val: int | None = None, seed: int = C.SEED):
    """Kuyu bazında eğitim/doğrulama bölmesi (aynı kuyu iki tarafa düşmez → veri sızıntısı yok).
    Eğitim kuyularında görülen sınıf sayısı en fazla olan ilk tohum seçilir; tekrar üretilebilir."""
    wells = d.WELL.unique()
    if n_val is None:   # tam veride 14/98; alt kümede aynı oran
        n_val = C.N_VAL_WELLS if len(wells) >= 98 else max(2, round(len(wells) * C.N_VAL_WELLS / 98))
    n_cls = d.y.nunique()
    for s in range(seed, seed + 1000):
        perm = np.random.default_rng(s).permutation(wells)
        tr_w, va_w = perm[n_val:], perm[:n_val]
        if d[d.WELL.isin(tr_w)].y.nunique() == n_cls:
            break
    return d[d.WELL.isin(tr_w)].copy(), d[d.WELL.isin(va_w)].copy()


def rastgele_bol(d: pd.DataFrame, val_frac: float = 0.15, seed: int = C.SEED):
    """YANLIŞ ama yaygın yöntem — sadece veri sızıntısını göstermek için.
    Satırları kuyuya bakmadan rastgele böler: aynı kuyunun komşu derinlikleri (0,15 m arayla,
    neredeyse aynı log değerleri) hem eğitime hem doğrulamaya düşer, doğrulama skoru şişer."""
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(d)) < val_frac
    return d[~is_val].copy(), d[is_val].copy()


# ---------------------------------------------------------------- özellik matrisi
class OnIsleyici:
    """Uç değer kırpma + 'eksik mi' bayrakları + eksik doldurma + standardizasyon.
    Kırpma sınırları, genel medyan ve ölçek SADECE eğitim kuyularından öğrenilir.

    doldurma="global": her yerde eğitim verisinin genel medyanı (VARSAYILAN).
    doldurma="kuyu":   önce o kuyunun kendi medyanı, log o kuyuda hiç yoksa genel medyan.
                       Denendi: resmi testte macro F1 0,416 → 0,343 düştü (scripts/temizleme_karsilastir.py)."""

    def __init__(self, features=C.FEATURES, clip_q=(0.001, 0.999), doldurma: str = "global"):
        self.features, self.clip_q, self.doldurma = list(features), clip_q, doldurma

    def _ham(self, d: pd.DataFrame) -> pd.DataFrame:
        X = d[self.features].clip(self.lo, self.hi, axis=1)
        flags = X.isna().astype("float32").add_suffix("_eksik")
        if self.doldurma == "kuyu":
            X = X.fillna(X.groupby(d.WELL.values).transform("median"))
        return pd.concat([X.fillna(self.med), flags], axis=1)

    def fit(self, d: pd.DataFrame):
        X = d[self.features]
        self.lo, self.hi = X.quantile(self.clip_q[0]), X.quantile(self.clip_q[1])
        self.med = X.median()
        self.scaler = StandardScaler().fit(self._ham(d))
        self.columns = list(self._ham(d.head(1)).columns)
        return self

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self._ham(d)).astype("float32")


@dataclass
class Veri:
    X_tr: np.ndarray; y_tr: np.ndarray; w_tr: np.ndarray
    X_va: np.ndarray; y_va: np.ndarray; w_va: np.ndarray
    X_te: np.ndarray; y_te: np.ndarray; w_te: np.ndarray
    columns: list
    tr: pd.DataFrame; va: pd.DataFrame; te: pd.DataFrame   # temizlenmiş ham tablolar (grafikler için)

    def ozet(self):
        for ad, X, w in (("eğitim", self.X_tr, self.w_tr), ("doğrulama", self.X_va, self.w_va),
                         ("test", self.X_te, self.w_te)):
            print(f"{ad:10s} {X.shape[0]:>9,} satır  {len(np.unique(w)):>3} kuyu  {X.shape[1]} özellik")


def hazirla(n_wells: int | None = None, features=C.FEATURES, bolme: str = "kuyu",
            fiziksel: bool = True, doldurma: str = "global") -> Veri:
    """Tek çağrıda: yükle → temizle → böl → ön işle. Test = resmi hidden_test seti (her zaman ayrı kuyular).
    bolme="kuyu"     → doğru yöntem: eğitim/doğrulama kuyu bazında ayrılır.
    bolme="rastgele" → sızıntı deneyi: satırlar rastgele ayrılır (doğrulama skoru şişer, test skoru şişmez)."""
    train, test = yukle(n_wells)
    d, te = temizle(train, fiziksel), temizle(test, fiziksel)
    if bolme == "kuyu":
        tr, va = kuyu_bol(d)
    elif bolme == "rastgele":
        tr, va = rastgele_bol(d)
        tr, va = tr.sort_values(["WELL", "DEPTH_MD"]), va.sort_values(["WELL", "DEPTH_MD"])
    else:
        raise ValueError('bolme "kuyu" veya "rastgele" olmalı')
    pp = OnIsleyici(features, doldurma=doldurma).fit(tr)
    return Veri(pp.transform(tr), tr.y.values, tr.WELL.values,
                pp.transform(va), va.y.values, va.WELL.values,
                pp.transform(te), te.y.values, te.WELL.values,
                pp.columns, tr, va, te)


# ---------------------------------------------------------------- 1D-CNN pencereleri
def pencere_dataset(X, y, wells, window: int = 33, batch_size: int = 1024,
                    shuffle: bool = False, class_weight: dict | None = None):
    """Her satır için, aynı kuyuda o derinliğin altındaki/üstündeki komşulardan oluşan
    (window, n_özellik) pencereyi anında üreten tf.data akışı. Bellekte pencere kopyası tutmaz.
    Kuyu kenarlarında kenar değeri tekrarlanır; pencere asla iki kuyuyu karıştırmaz.
    X, y, wells kuyu+derinlik sıralı olmalı (temizle() bu sırayı garanti eder)."""
    import tensorflow as tf

    half = window // 2
    blocks, centers, start = [], [], 0
    for w in pd.unique(wells):
        idx = np.flatnonzero(wells == w)
        blocks.append(np.pad(X[idx], ((half, half), (0, 0)), mode="edge"))
        centers.append(start + half + np.arange(len(idx)))
        start += len(idx) + 2 * half
    Xp = tf.constant(np.concatenate(blocks))
    centers = np.concatenate(centers).astype("int64")
    offsets = tf.range(-half, half + 1, dtype=tf.int64)

    ds = tf.data.Dataset.from_tensor_slices((centers, y))
    if shuffle:
        ds = ds.shuffle(len(centers), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    def _gather(c, t):
        win = tf.gather(Xp, c[:, None] + offsets[None, :])
        if class_weight is None:
            return win, t
        sw = tf.gather(tf.constant([class_weight[i] for i in range(len(class_weight))], tf.float32), t)
        return win, t, sw
    return ds.map(_gather, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":   # python -m src.veri  → sadece veriyi indirir
    indir()
    print("Veri hazır:", sorted(p.name for p in C.DATA_DIR.iterdir()))
