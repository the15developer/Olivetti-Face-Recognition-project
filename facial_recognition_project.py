import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns



scaler = StandardScaler()

faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

X = X.reshape(X.shape[0], -1) #nb-classifier & mlp classifier


X_scaled = scaler.fit_transform(X)  #nb-classifier & mlp classifier


X = X.reshape(X.shape[0], 64, 64, 1)   #cnn classifier

X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)  #cnn classifier



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_pred = nb_classifier.predict(X_test)
nb_accuracy=accuracy_score(y_test, nb_pred)


print("\nNaive Bayes Accuracy:", nb_accuracy)

print("\nNaive Bayes Classification Report:")

print(classification_report(y_test, nb_pred))


# Naive Bayes sınıflandırıcısının doğru ve yanlış tahminlerini ayır
correct_predictions = []
incorrect_predictions = []
for true_label, pred_label in zip(y_test, nb_pred):
    if true_label == pred_label:
        correct_predictions.append((true_label, pred_label))
    else:
        incorrect_predictions.append((true_label, pred_label))


# Doğru sınıflandırılan örnekleri görselleştir
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, (true_label, pred_label) in enumerate(correct_predictions[:9]):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(X_test[i].reshape(64, 64), cmap='gray')
    axes[row, col].set_title(f"True: {true_label}, Pred: {pred_label}")
    axes[row, col].axis('off')
plt.suptitle('Doğru Sınıflandırılan Örnekler')
plt.show()

# Yanlış sınıflandırılan örnekleri görselleştir
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, (true_label, pred_label) in enumerate(incorrect_predictions[:9]):
    row = i // 3
    col = i % 3
    axes[row, col].imshow(X_test[i].reshape(64, 64), cmap='gray')
    axes[row, col].set_title(f"True: {true_label}, Pred: {pred_label}")
    axes[row, col].axis('off')
plt.suptitle('Yanlış Sınıflandırılan Örnekler')
plt.show()


class_counts = pd.Series(y).value_counts()
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.title('Olivetti Yüzleri Veri Kümesi Sınıf Dağılımı')
plt.xlabel('Sınıf')
plt.ylabel('Örnek Sayısı')
plt.show()



# Naive Bayes sınıflandırıcısının doğruluğunu görselleştir
plt.figure(figsize=(8, 6))
plt.plot([1], [nb_accuracy], marker='o', label='Naive Bayes')
plt.title('Sınıflandırıcıların Doğruluk Karşılaştırması')
plt.xlabel('Sınıflandırıcı')
plt.ylabel('Doğruluk')
plt.xticks([1], ['Naive Bayes'])
plt.legend()
plt.show()


# Veri setindeki tüm piksellerin değer dağılımını görselleştir
plt.figure(figsize=(8, 6))
plt.hist(X.flatten(), bins=50, edgecolor='k')
plt.title('Olivetti Yüzleri Veri Kümesi Piksel Değer Dağılımı')
plt.xlabel('Piksel Değeri')
plt.ylabel('Frekans')
plt.show()


# Naive Bayes sınıflandırıcısı için çapraz doğrulama
nb_scores = cross_val_score(nb_classifier, X_scaled, y, cv=5)
print("Naive Bayes Çapraz Doğrulama Sonuçları:")
print(nb_scores)
print(f"Ortalama Doğruluk: {np.mean(nb_scores):.2f}")

# Naive Bayes sınıflandırıcısı için konfüzyon matrisi
cm = confusion_matrix(y_test, nb_pred)

# Konfüzyon matrisini görselleştir
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Naive Bayes Sınıflandırıcısı Konfüzyon Matrisi')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show()






# Define the MLP model
model = Sequential()
model.add(Flatten(input_shape=(4096,)))  # Flatten the input data
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(40, activation='softmax'))  # 40 classes (individuals)


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")







# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(40, activation='softmax'))  # 40 classes (individuals)

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")












