from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize CNN
model = Sequential()

# Convolution + Pooling Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=29, activation='softmax'))  # 29 classes for A-Z + 'del', 'nothing', 'space'

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Preparation with validation split
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1  # ✅ 10% of training set used for validation
)

training_set = train_datagen.flow_from_directory(
    'asl_alphabet_train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # ✅ Use 90% for training
)

validation_set = train_datagen.flow_from_directory(
    'asl_alphabet_train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # ✅ Use 10% for validation
)

# Train
model.fit(
    training_set,
    epochs=10,
    validation_data=validation_set
)

# Save model
model.save('CNNmodel.h5')
print("✅ Model training complete and saved as 'CNNmodel.h5'")

