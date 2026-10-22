"""Settings of the HAM10000 training run."""

# The diagnoses trained as malignant; every other diagnosis is benign.
MALIGNANT = ("mel", "bcc", "akiec")
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LAST_CHECKPOINT = "last"
BEST_CHECKPOINT = "best"
