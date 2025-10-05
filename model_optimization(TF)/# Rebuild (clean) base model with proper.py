# Rebuild (clean) base model with proper output for classification + mixed precision safe softmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation

base_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(10),  # logits
    Activation('softmax', dtype='float32')  # ensure numerically stable output
])

import tensorflow as tf
import tensorflow_model_optimization as tfmot

batch_size = 64
epochs = 10
# Assume x_train already loaded
steps_per_epoch = len(x_train) // batch_size
end_step = steps_per_epoch * epochs

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=end_step
    )
}

model_pruned = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)

from tensorflow.keras.optimizers import Adam
model_pruned.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # Optional:
    # tfmot.sparsity.keras.PruningSummaries(log_dir='pruning_logs')
]

history = model_pruned.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# Strip pruning wrappers for final inference/export
model_final = tfmot.sparsity.keras.strip_pruning(model_pruned)
print("Final (stripped) model:")
model_final.summary()