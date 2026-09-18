##  Report on Challenges Faced

**1. Choosing the right evaluation metric for a 4-class problem.**
Since `price_range` has four ordinal classes, plain accuracy can be misleading if classes are imbalanced.
*Technique used:* We first verified the target distribution (Section 1.5) and confirmed it was perfectly
balanced (500 per class), which justified using accuracy as a primary metric while still reporting
macro-averaged precision/recall/F1 as a safety net for any per-class weaknesses that accuracy alone would
hide.

**2. Feature scale disparity across specifications.**
Features like `ram` (up to ~4000) and `m_dep` (0–1) live on wildly different scales, which would bias
distance-based and gradient-based models (KNN, SVM, Logistic Regression) toward large-magnitude features.
*Technique used:* Applied `StandardScaler`, fitting only on the training split to prevent data leakage
into the test set, and reused the same scaled features consistently across all models for a fair,
apples-to-apples comparison.

**3. Mild multicollinearity between related specs.**
Screen height/width (`sc_h`/`sc_w`), pixel height/width (`px_height`/`px_width`), and front/primary camera
(`fc`/`pc`) are naturally correlated with each other (Section 1.8). This can destabilize coefficient
estimates in linear models like Logistic Regression.
*Technique used:* Rather than dropping features (which risks losing signal), we relied on models that are
naturally robust to multicollinearity for the primary comparison (tree ensembles), and used regularized
Logistic Regression (`C` tuned via grid search) to control coefficient instability for the linear model.

**4. Weak individual correlation of several binary features.**
Several one-hot/binary specs (Bluetooth, WiFi, dual-SIM, touch screen, 3G/4G) show near-zero linear
correlation with price on their own (Section 1.8), which could tempt premature feature dropping.
*Technique used:* We kept all original features in the model rather than manually filtering by
correlation, since correlation only captures linear/marginal relationships — tree-based models can still
exploit these features via interactions (e.g., 4G *and* high RAM together). Section 4's feature-importance
step (which captures non-linear, model-based importance) was used as the authoritative signal instead of
raw correlation for the business report.

**5. Balancing model complexity vs. overfitting risk.**
Powerful models like unrestricted Decision Trees and RBF-kernel SVMs can overfit a relatively small
(2000-row) dataset.
*Technique used:* Used 5-fold stratified cross-validation to compare CV accuracy against held-out test
accuracy for every model (Section 3.2) — a large gap between CV/train performance and test performance
was treated as a red flag for overfitting, and `GridSearchCV` hyperparameter tuning (Section 3.4) was used
to regularize the chosen model (e.g., controlling tree depth, `C`, `min_samples_leaf`) rather than trusting
default parameters.

**6. Selecting *one* best model among many close performers.**
Several models often land within a percentage point of each other on this dataset, making "best" somewhat
ambiguous.
*Technique used:* Defined "best" using a combined view — highest test accuracy, small CV/test gap
(generalization), and balanced macro F1 across all four classes — rather than test accuracy alone, so the
selected production model is robust, not just a lucky high scorer on one split.
