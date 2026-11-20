"""Eğitim yardımcıları: sınıf ağırlıkları ve callback'li fit."""
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from . import config as C


def sinif_agirliklari(y, cap: float = 50.0) -> dict:
    """'balanced' sınıf ağırlıkları; çok nadir sınıflarda patlamasın diye cap ile kırpılır.
    Eğitimde hiç görülmeyen sınıfa 1.0 verilir."""
    present = np.unique(y)
    w = compute_class_weight("balanced", classes=present, y=y)
    out = {i: 1.0 for i in range(C.N_CLASSES)}
    out.update({int(c): float(min(v, cap)) for c, v in zip(present, w)})
    return out


def callbacks(patience: int = 5):
    return [keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=max(1, patience // 2), factor=0.5)]


def egit(model, X_tr, y_tr, X_va, y_va, epochs: int = 30, batch_size: int = 1024,
         class_weight: dict | None = None, patience: int = 5, verbose: int = 2):
    """MLP gibi düz (n, özellik) girdili modeller için fit. EarlyStopping ve ReduceLROnPlateau açık."""
    keras.utils.set_random_seed(C.SEED)
    return model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                     batch_size=batch_size, class_weight=class_weight,
                     callbacks=callbacks(patience), verbose=verbose)


def egit_ds(model, ds_tr, ds_va, epochs: int = 30, patience: int = 5, verbose: int = 2):
    """tf.data akışlarıyla fit (1D-CNN pencereleri). Sınıf ağırlığı akışın içinde verilir."""
    keras.utils.set_random_seed(C.SEED)
    return model.fit(ds_tr, validation_data=ds_va, epochs=epochs,
                     callbacks=callbacks(patience), verbose=verbose)
