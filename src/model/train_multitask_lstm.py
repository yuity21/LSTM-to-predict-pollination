import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Configuration des dossiers
input_dir = "data/splits/lstm_splits_multitask"
output_dir = "models/lstm_models_multitask"
os.makedirs(output_dir, exist_ok=True)

cultures = ["colza", "tournesol", "lavande", "pommiers"]

# Chargement des données
X_train = np.load(os.path.join(input_dir, "X_train_multi.npy"))
y_train = np.load(os.path.join(input_dir, "y_train_multi.npy"))
X_val = np.load(os.path.join(input_dir, "X_val_multi.npy"))
y_val = np.load(os.path.join(input_dir, "y_val_multi.npy"))
X_test = np.load(os.path.join(input_dir, "X_test_multi.npy"))
y_test = np.load(os.path.join(input_dir, "y_test_multi.npy"))

# Vérification GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU détecté :", gpus[0].name)
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("Aucun GPU détecté, entraînement sur CPU.")

# Calcul des poids de classes et définition des fonctions de perte
losses = {}
class_weights_dicts = {}

for i, culture in enumerate(cultures):
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.unique(y_train[:, i]),
                                   y=y_train[:, i])
    class_weights_dicts[culture] = dict(enumerate(weights))
    print(f"📏 Poids de classe pour {culture} :", class_weights_dicts[culture])
    
    def make_loss_fn(index, weights):
        def weighted_loss(y_true, y_pred):
            y_true_int = tf.cast(y_true, tf.int32)
            y_true_oh = tf.one_hot(y_true_int, depth=3)
            weight_tensor = tf.constant(weights, dtype=tf.float32)
            sample_weights = tf.reduce_sum(y_true_oh * weight_tensor, axis=-1)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return loss * sample_weights
        return weighted_loss
    
    losses[f"{culture}_out"] = make_loss_fn(i, weights)

# Nouvelle architecture du modèle avec attention
input_shape = X_train.shape[1:]
inputs = tf.keras.Input(shape=input_shape)

# Bloc 1 : Stacked Bidirectional LSTM
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Mécanisme Multi-Head
# On applique l'attention sur la sortie de LSTM, en utilisant x comme query, key et value.
attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
# Application d'un skip connection
x = tf.keras.layers.Add()([x, attention_output])
# Agrégation temporelle pour obtenir une représentation fixe
x = tf.keras.layers.GlobalAveragePooling1D()(x)

# Dense fully connected
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)

# Sorties multi-tâches (une sortie par culture)
outputs = [
    tf.keras.layers.Dense(3, activation="softmax", name=f"{culture}_out")(x)
    for culture in cultures
]

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Utilisation d'un taux d'apprentissage réduit
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss=losses,
    loss_weights={
        'colza_out': 1.0,
        'tournesol_out': 1.0,
        'lavande_out': 1.0,
        'pommiers_out': 1.0
    },
    metrics={f"{c}_out": "accuracy" for c in cultures}
)

model.summary()

# Callbacks (incluant ReduceLROnPlateau)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, "best_multitask_lstm.keras"),
        save_best_only=True,
        monitor="val_colza_out_accuracy",  # Vous pouvez aussi utiliser une métrique moyenne
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_colza_out_accuracy",
        mode="max",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_colza_out_accuracy",
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )
]

# Entraînement du modèle
history = model.fit(
    X_train,
    {f"{culture}_out": y_train[:, i] for i, culture in enumerate(cultures)},
    validation_data=(X_val, {f"{culture}_out": y_val[:, i] for i, culture in enumerate(cultures)}),
    epochs=150,
    batch_size=32,
    callbacks=callbacks
)

# Sauvegarde des courbes d’accuracy
for culture in cultures:
    plt.figure()
    plt.plot(history.history[f"{culture}_out_accuracy"], label="Train")
    plt.plot(history.history[f"val_{culture}_out_accuracy"], label="Val")
    plt.title(f"Accuracy — {culture}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{culture}_accuracy.png"))
    plt.close()

# Évaluation finale sur le jeu de test
print("\n🎓 ÉVALUATION SUR LE TEST SET\n")
y_pred = model.predict(X_test)
for i, culture in enumerate(cultures):
    print(f"\n📊 {culture.upper()} :")
    y_true = y_test[:, i]
    y_pred_class = np.argmax(y_pred[i], axis=1)
    print(classification_report(y_true, y_pred_class, target_names=["basse", "moyenne", "haute"]))
    print("Matrice de confusion :")
    print(confusion_matrix(y_true, y_pred_class))
