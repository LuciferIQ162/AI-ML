import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
dataset = '/home/shobhit/UNIFIED_mentor_project/Animal Classification/dataset'  # example path

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,      # Add some data augmentation
    horizontal_flip=True
)

print("Setting up data generators...")
train_generator = datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    seed=SEED)

validation_generator = datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    seed=SEED)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Found {train_generator.n} training images belonging to {NUM_CLASSES} classes.")
print(f"Found {validation_generator.n} validation images.")

# 2. Build the Model using Transfer Learning
print("\nBuilding model with MobileNetV2 base...")

# Load the pre-trained model (the "expert") without its top classification layer
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the expert layers so we don't change them during training
base_model.trainable = False

# Add our own custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # A good alternative to Flatten()
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 4. Train the Model
print("\nTraining the model...")
history = model.fit(
    train_generator,
    epochs=5,  # Start with a few epochs
    validation_data=validation_generator
)

print("\n✅ Model training complete.")