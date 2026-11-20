# FORCE 2020 — Kuyu Loglarından Litoloji Sınıflandırması

Sakarya Üniversitesi **ISE 469 Derin Öğrenmeye Giriş** dönem projesi.

Petrol kuyularında derinlik boyunca ölçülen loglardan (gamma ışını, yoğunluk, rezistivite…) o derinlikteki
kayaç türünü (12 sınıf: şeyl, kumtaşı, kireçtaşı…) tahmin ediyoruz ve **XGBoost**, **MLP** ve **1D-CNN**
yaklaşımlarını karşılaştırıyoruz.

- **Veri:** [FORCE 2020 Machine Predicted Lithology](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition) (CC BY 4.0)
  - Eğitim: 98 kuyu, 1.170.511 satır → kuyu bazında 84 eğitim / 14 doğrulama kuyusu
  - Test: yarışmanın etiketli `hidden_test` seti, 10 ayrı kuyu, 122.397 satır
- **Ana metrik:** macro F1 (şeyl verinin %62'si; accuracy yanıltıcı)

---

## Kurulum (yerel)

```bash
git clone https://github.com/aysenursimsk/force2020-litoloji.git
cd force2020-litoloji

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

nbstripout --install                 # notebook çıktıları commit'e girmesin (her klonda bir kez)
python -m src.veri                   # veriyi data/ altına indirir (~120 MB, bir kez)
jupyter lab                          # veya VS Code
```

## Kurulum (Colab)

1. Colab → *Dosya › Not defterini aç › GitHub* → repoyu seçip bir notebook açın.
2. İlk hücre (*Kurulum*) repoyu `/content` altına klonlar, `git pull` yapar ve `src`'yi import edilebilir kılar.
   İlk hücredeki `REPO_URL`'yi kendi repo adresinizle değiştirin.
3. Veri ilk `veri.hazirla()` / `veri.yukle()` çağrısında otomatik iner (Colab oturumu başına bir kez, ~1 dk).

Colab'da `tensorflow`, `xgboost`, `scikit-learn` hazır gelir; ek kurulum gerekmez.

**Private repo ise:** GitHub'da *Settings › Developer settings › Fine-grained token* ile bu repoya
*Contents: Read and write* yetkili bir token oluşturun, Colab'da sol menüdeki 🔑 *Secrets*'a `GITHUB_TOKEN`
adıyla ekleyin ve `REPO_URL`'yi şöyle kurun:

```python
from google.colab import userdata
REPO_URL = f"https://{userdata.get('GITHUB_TOKEN')}@github.com/aysenursimsk/force2020-litoloji.git"
```

Bu ayarla Colab'dan doğrudan commit/push da yapılabilir:

```python
!git config user.name "Ad Soyad" && git config user.email "mail@ornek.com"
!git add sonuclar/ && git commit -m "Deney sonuçları: lr denemeleri" && git push
```

---

## VS Code ile çalışma

1. Repo klasörünü VS Code'da açın; **Python** ve **Jupyter** eklentilerini kurun.
2. `Ctrl+Shift+P` → *Python: Select Interpreter* → `.venv` içindeki Python'u seçin.
3. Notebook'lar (`notebooks/*.ipynb`) VS Code içinde açılır; sağ üstteki *Select Kernel* ile yine `.venv`'i seçin.
4. Script'ler repo kökünden çalıştırılır, ör. temizleme seçeneklerinin karşılaştırması:

```bash
python -m scripts.temizleme_karsilastir --kuyu 30 --kim muhammed   # hızlı (~1 dk)
python -m scripts.temizleme_karsilastir --kim muhammed             # tam veri (~6 dk)
```

`src/` içindeki bir fonksiyonu tek başına denemek için VS Code'da *Python Interactive* penceresi
veya terminalde `python -c "from src import veri; d = veri.hazirla(n_wells=10); d.ozet()"` yeterli.

## Proje yapısı

```
force2020-litoloji/
├── src/
│   ├── config.py        # yollar, veri adresleri, özellikler, sınıf adları
│   ├── veri.py          # indirme, temizleme, kuyu bazlı bölme, ön işleme, CNN pencereleri
│   ├── modeller.py      # build_mlp, build_cnn1d (Keras)
│   ├── egitim.py        # sınıf ağırlıkları, EarlyStopping'li fit
│   └── degerlendir.py   # skorlar, rapor, confusion matrix, öğrenme eğrisi, kuyu grafiği, deney kaydı
├── notebooks/
│   ├── 00_kesif.ipynb             # keşifsel analiz
│   ├── 01_baseline_xgboost.ipynb  # baseline
│   ├── 02_mlp.ipynb               # MLP + hiperparametre deney şablonu
│   ├── 03_cnn1d.ipynb             # 1D-CNN (derinlik penceresi)
│   └── 99_sonuclar.ipynb          # ekibin tüm deneyleri tek tabloda
├── scripts/
│   └── temizleme_karsilastir.py   # temizleme seçeneklerini XGBoost ile karşılaştırır
├── sonuclar/            # deneyler_<ad>.csv — herkesin kendi dosyası (commit edilir)
├── data/                # veri buraya iner (git'e girmez)
└── requirements.txt
```

**Kural:** Ortak/tekrar kullanılan kod `src/` içine, deney ve görselleştirme notebook'lara.
Notebook'lar `src`'yi import eder; `%autoreload` açık olduğu için `src`'de yaptığınız değişiklik
notebook'u yeniden başlatmadan geçerli olur.

## Ön işleme özeti (`src/veri.py`)

1. %70'ten fazla eksik loglar ve konum/formasyon sütunları kullanılmaz (`config.FEATURES`).
2. Fiziksel olarak imkânsız değerler (ör. yoğunluk < 1 g/cm³, NPHI > 1) → NaN (`config.PHYSICAL_BOUNDS`);
   negatif rezistivite → NaN; rezistivitelere log10.
3. Kuyu bazında bölme (aynı kuyu eğitim ve doğrulamaya birlikte düşmez → veri sızıntısı yok).
4. Uç değerler eğitim verisinin %0,1–%99,9 aralığına kırpılır, eksikler medyanla doldurulur,
   her özellik için `_eksik` bayrağı eklenir, standardizasyon yapılır. **Hepsi sadece eğitim kuyularına fit edilir.**

## Deney kaydı

Her notebook sonunda:

```python
D.kaydet("lr_1e-4", "MLP", ayar, skor, kim=KIM, epochs=...)
```

→ `sonuclar/deneyler_<kim>.csv`'ye bir satır ekler. Herkesin ayrı dosyası olduğu için merge çakışması olmaz.
Hepsini görmek için `99_sonuclar.ipynb` veya `D.tum_sonuclar()`.

## Git akışı

- `main` her zaman çalışır durumda; doğrudan push yerine kısa ömürlü dal + Pull Request.
- Dal adı: `ad/konu` (ör. `muhammed/cnn-pencere`).
- Aynı notebook'u iki kişi aynı anda düzenlemesin; yeni deney için notebook'u kopyalayın
  (ör. `02_mlp_lr_denemeleri.ipynb`).
- Commit'ten önce notebook çıktıları temizlenmiş olmalı (`nbstripout --install` bunu otomatik yapar).

## Hızlı deneme

Tam veriyle bir MLP eğitimi CPU'da epoch başına ~10 sn sürer. Hiperparametre taramasında hız için
notebook başındaki `N_WELLS = 20` ile sadece ilk 20 eğitim kuyusu kullanılabilir; final sonuçlar `N_WELLS = None` ile alınır.
