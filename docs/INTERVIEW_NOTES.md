# Explaining this project

**One-minute walkthrough:** "I compared a TF-IDF baseline with frozen transformer
embeddings for 77 banking support intents. I trained the classifier on one split,
calibrated its confidence on another, and selected a review policy on validation.
The model reached 93.1% overall test accuracy. Routing only confident requests
gave 96.9% accuracy at 90.6% coverage. I also found a meaningful weakness: typos
and unsupported questions can still produce errors, including confident ones."

Trace a request through `intentlab/encoder.py`, `service.py`, and `api.py`. Explain
attention masking, mean pooling, normalization and how the 384-dimensional vector
reaches a logistic classifier. The transformer is pretrained and frozen; do not
describe it as a fine-tuned language model.

**Why a baseline?** TF-IDF is cheap and interpretable. The measured comparison
shows whether the added encoder is worthwhile instead of assuming it is better.

**What does temperature scaling do?** It changes confidence while preserving
the top class. It cannot repair a wrong class or establish that a request belongs
to the known set of categories.

**What does 96.9% mean?** Accuracy on 2,792 accepted test requests, not all 3,080.
The 288 deferred requests are not counted as correct predictions. Neither number
is a measured operational cost saving.

**Why the high-confidence dog example?** Softmax distributes probability among
known labels even when none is suitable. That is why the model needs a separate
out-of-domain evaluation and a new rejection experiment before a real pilot.

**What should happen next?** Collect representative noisy and unsupported text,
define error costs with operations, evaluate a rejection model, and use a new
untouched holdout. Understand and run these steps before describing ownership in
an interview; the repository is evidence you can inspect, not a script to memorize.
