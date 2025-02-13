import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# 1. Charger le CSV contenant les chemins d'images et les pseudo‑labels
df = pd.read_csv("pseudo_labels_finetuning.csv")
# Si nécessaire, vérifier que les colonnes sont bien 'filepath' et 'pseudo_label'
print(df.head())

# 2. Diviser le DataFrame en ensemble d'entraînement et de validation
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['pseudo_label'])

# 3. Configurer les générateurs d'images
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='filepath',
    y_col='pseudo_label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',  # Les labels sont numériques (0 ou 1)
    shuffle=True
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='filepath',
    y_col='pseudo_label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# 1. Charger le modèle préalablement fine tuné
model = load_model('chemin')


# 3. Recompiler le modèle avec un taux d'apprentissage réduit pour le fine tuning
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Re-fine tuner le modèle sur le nouvel ensemble
model.fit(train_generator, validation_data=val_generator, epochs=2)

# 5. Sauvegarder le modèle re-fine tuné pour une utilisation ultérieure
model.save('modele_finetuned_2.h5')

# -----------------------------------------------------
# Optionnel : une fois les premières phases de fine tuning effectuées,
# vous pouvez dégeler certaines couches de base_model pour un entraînement plus fin.
#
# Exemple de dégeler les 4 dernières couches :
# for layer in base_model.layers[-4:]:
#     layer.trainable = True
#
# Recompiler et poursuivre l'entraînement :
# model.compile(optimizer=Adam(learning_rate=1e-5), 
#               loss='sparse_categorical_crossentropy', 
#               metrics=['accuracy'])
# model.fit(train_generator, validation_data=val_generator, epochs=5)
