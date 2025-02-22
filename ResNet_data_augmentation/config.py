import os

# Chemin relatif vers le dataset (assurez-vous que le nom du dossier est le même sur GitHub)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'combined_dataset')

# Paramètres d'image et d'entraînement
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 30

# Autres hyperparamètres
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 1e-5
