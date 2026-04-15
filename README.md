# animal-ml-assignment-



## 📄 README




**Animal Classification & Duplicate Detection**

---

**Objective**

The goal of this assignment was to build a simple image pipeline that can classify animals (cow vs buffalo) and detect near-duplicate images.

---

**1. Image Classification**

I used a pretrained ResNet18 model and applied transfer learning.

Since the dataset is very small (around 30 images), training a model from scratch would lead to overfitting. A pretrained model already understands general visual features like edges and textures, so it works much better for this task.

To keep training stable:

* most of the model was frozen
* only the final classification layer was trained

---

**Data Processing**

* images resized to 224 × 224
* applied basic augmentation:

  * horizontal flip
  * small rotations
  * brightness and contrast changes
* normalized using ImageNet statistics

---

**Handling Class Imbalance**

There were fewer buffalo images compared to cows, so I used a class-weighted loss function to reduce bias toward the cow class.

---

**Results**

* Accuracy: ~0.86
* F1 Score: ~0.86

From the confusion matrix:

* all cow images were classified correctly
* one buffalo image was misclassified as cow

Given the dataset size, this is a reasonable outcome.

---

**Observations / Failure Cases**

* dataset is small and slightly imbalanced
* cows and buffalo look similar under certain lighting conditions
* some images are blurry, which affects confidence

---

**2. Near-Duplicate Detection**

Duplicate detection was done using deep feature embeddings.

---

**Approach**

* extracted features from the second last layer of ResNet18
* generated embeddings for each image
* computed cosine similarity between all image pairs
* grouped images based on similarity threshold

---

**Threshold Selection**

A threshold of 0.92 was selected after inspecting the most similar image pairs.

* above 0.92 → near-duplicate or very similar
* below 0.90 → mostly different images

---

**Output Format**

Final output is saved as:

results.csv

with columns:

image_name, predicted_label, confidence_score, duplicate_group_id

Example:

img1.jpg → cow → 0.91 → group_0
img2.jpg → cow → 0.88 → group_0
img3.jpg → buffalo → 0.85 → None

---

**Scaling Considerations**

To scale this system for large datasets:

* run batch inference on GPU
* store embeddings in a vector database (like FAISS)
* use approximate nearest neighbor search
* process images in parallel
* cache embeddings to avoid recomputation

---

**Improvements**

With more time, I would:

* collect more balanced data
* fine-tune more layers of the model
* try better architectures like EfficientNet or CLIP
* use clustering algorithms for duplicate grouping

---

**Final Thoughts**

This pipeline works end-to-end:

* performs classification with reasonable accuracy
* detects similar images using embeddings
* produces structured output

The main limitation is dataset size, but the overall approach is scalable and practical.

---

**Setup**

pip install torch torchvision pandas scikit-learn matplotlib seaborn

---


