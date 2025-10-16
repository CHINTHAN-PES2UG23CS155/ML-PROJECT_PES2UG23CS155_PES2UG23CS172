import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import joblib

# Load and prepare the data
print("Loading data...")
data = pd.read_csv('D:\ml\projet\pro\extracted_features.csv')

# Separate features and labels
X = data.drop(['filename', 'label'], axis=1)
# Convert string representations of lists to float values
for column in X.columns:
    X[column] = X[column].apply(lambda x: float(str(x).strip('[]')) if isinstance(x, str) else x)
y = data['label']

# Encode the labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder for later use
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'feature_scaler.pkl')

# Build the model
print("Building model...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
# Train the model
history = model.fit(X_train_scaled, y_train,
                   validation_data=(X_test_scaled, y_test),
                   epochs=100,
                   batch_size=32,
                   verbose=1)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# Save the model
print("Saving model...")
model.save('cnn_model.h5')

print("Done! Model has been trained and saved.")

# Print class distribution
print("\nClass distribution in training data:")
for label, count in zip(label_encoder.classes_, np.sum(y_train, axis=0)):
    print(f"{label}: {count}")