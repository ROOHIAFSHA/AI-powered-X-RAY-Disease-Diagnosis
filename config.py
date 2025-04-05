

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
import os

# Get the BASE folder (Pneumonia-Detection)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model location (absolute path, so no issues where you run the code from)
MODEL_LOC = os.path.join(BASE_DIR, "model", "pneumonia_detection_cnn_model.h5")

# Data directory (pointing to the actual dataset location)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray', 'chest_xray')

# Paths for train, test, and validation folders
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
VAL_DATA_DIR = os.path.join(DATA_DIR, 'val')

# Detection classes
DETECTION_CLASSES = ('NORMAL', 'PNEUMONIA')

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100

