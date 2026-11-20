"""Keras modelleri. Deneylerde sadece fonksiyon argümanlarını değiştirin."""
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from . import config as C


def _optimizer(name: str, lr: float):
    if name == "adam":
        return keras.optimizers.Adam(lr)
    if name == "sgd":
        return keras.optimizers.SGD(lr, momentum=0.9)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(lr)
    raise ValueError(name)


def _compile(m, optimizer, lr):
    m.compile(optimizer=_optimizer(optimizer, lr),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_mlp(n_features: int, n_layers: int = 3, units: int = 128, act: str = "relu",
              dropout: float = 0.3, l2: float = 1e-4, batchnorm: bool = True,
              optimizer: str = "adam", lr: float = 1e-3, n_classes: int = C.N_CLASSES):
    """Tam bağlı ağ: [Dense → BatchNorm → aktivasyon → Dropout] × n_layers → Softmax."""
    m = keras.Sequential([layers.Input((n_features,))], name="mlp")
    for _ in range(n_layers):
        m.add(layers.Dense(units, kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(l2) if l2 else None))
        if batchnorm:
            m.add(layers.BatchNormalization())
        m.add(layers.Activation(act))
        if dropout:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(n_classes, activation="softmax"))
    return _compile(m, optimizer, lr)


def build_cnn1d(n_features: int, window: int = 33, filters=(64, 128), kernel: int = 5,
                act: str = "relu", dropout: float = 0.3, l2: float = 1e-4,
                optimizer: str = "adam", lr: float = 1e-3, n_classes: int = C.N_CLASSES):
    """Derinlik penceresi üzerinde 1D-CNN: [Conv1D → BatchNorm → aktivasyon → MaxPool] × len(filters)
    → GlobalAveragePooling → Dropout → Dense → Softmax. Pencerenin ortasındaki derinliği sınıflandırır."""
    reg = regularizers.l2(l2) if l2 else None
    m = keras.Sequential([layers.Input((window, n_features))], name="cnn1d")
    for f in filters:
        m.add(layers.Conv1D(f, kernel, padding="same", kernel_initializer="he_normal",
                            kernel_regularizer=reg))
        m.add(layers.BatchNormalization())
        m.add(layers.Activation(act))
        m.add(layers.MaxPooling1D(2))
    m.add(layers.GlobalAveragePooling1D())
    if dropout:
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(64, activation=act, kernel_regularizer=reg))
    m.add(layers.Dense(n_classes, activation="softmax"))
    return _compile(m, optimizer, lr)
