"""Settings of the Camelyon17 training run."""

TRAINING_HOSPITALS = (0, 3, 4)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LAST_CHECKPOINT = "last"
BEST_CHECKPOINT = "best"
