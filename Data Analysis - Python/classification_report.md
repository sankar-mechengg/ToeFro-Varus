# Classification Report - IMU Walking Type Prediction

## Overview

- **Total Models Evaluated:** 8
- **Classes:** FROG, HEEL, KANGAROO, SCISSOR, SIDEKICK, TIPTOE
- **Train/Test Split:** 65% / 35%

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Gradient_Boosting | 0.9592 | 0.9662 | 0.9592 | 0.9592 |
| Logistic_Regression | 0.9442 | 0.9510 | 0.9442 | 0.9437 |
| Random_Forest | 0.9428 | 0.9514 | 0.9428 | 0.9426 |
| XGBoost | 0.9244 | 0.9395 | 0.9244 | 0.9242 |
| SVM | 0.9167 | 0.9375 | 0.9167 | 0.9158 |
| Decision_Tree | 0.8961 | 0.9034 | 0.8961 | 0.8910 |
| KNN | 0.8900 | 0.9128 | 0.8900 | 0.8863 |
| Naive_Bayes | 0.8111 | 0.8626 | 0.8111 | 0.8194 |

## Detailed Classification Reports

### Gradient_Boosting

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 1.0000 | 0.9611 | 0.9802 | 745 |
| HEEL | 0.9985 | 1.0000 | 0.9992 | 655 |
| KANGAROO | 0.9272 | 1.0000 | 0.9622 | 688 |
| SCISSOR | 0.7677 | 1.0000 | 0.8686 | 304 |
| SIDEKICK | 1.0000 | 0.9985 | 0.9992 | 646 |
| TIPTOE | 1.0000 | 0.7918 | 0.8838 | 562 |

**Accuracy:** 0.9592
**Macro Avg:** Precision=0.9489, Recall=0.9586, F1-Score=0.9489
**Weighted Avg:** Precision=0.9662, Recall=0.9592, F1-Score=0.9592

**Sensitivity (Recall) by Class:**

- FROG: 0.9611
- HEEL: 1.0000
- KANGAROO: 1.0000
- SCISSOR: 1.0000
- SIDEKICK: 0.9985
- TIPTOE: 0.7918

---

### Logistic_Regression

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 0.9985 | 0.8658 | 0.9274 | 745 |
| HEEL | 0.9424 | 1.0000 | 0.9704 | 655 |
| KANGAROO | 0.8718 | 0.9985 | 0.9309 | 688 |
| SCISSOR | 0.8375 | 1.0000 | 0.9115 | 304 |
| SIDEKICK | 1.0000 | 1.0000 | 1.0000 | 646 |
| TIPTOE | 1.0000 | 0.8221 | 0.9023 | 562 |

**Accuracy:** 0.9442
**Macro Avg:** Precision=0.9417, Recall=0.9477, F1-Score=0.9404
**Weighted Avg:** Precision=0.9510, Recall=0.9442, F1-Score=0.9437

**Sensitivity (Recall) by Class:**

- FROG: 0.8658
- HEEL: 1.0000
- KANGAROO: 0.9985
- SCISSOR: 1.0000
- SIDEKICK: 1.0000
- TIPTOE: 0.8221

---

### Random_Forest

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 1.0000 | 0.8631 | 0.9265 | 745 |
| HEEL | 0.9820 | 1.0000 | 0.9909 | 655 |
| KANGAROO | 0.8190 | 1.0000 | 0.9005 | 688 |
| SCISSOR | 0.9048 | 1.0000 | 0.9500 | 304 |
| SIDEKICK | 0.9848 | 1.0000 | 0.9923 | 646 |
| TIPTOE | 1.0000 | 0.8149 | 0.8980 | 562 |

**Accuracy:** 0.9428
**Macro Avg:** Precision=0.9484, Recall=0.9463, F1-Score=0.9431
**Weighted Avg:** Precision=0.9514, Recall=0.9428, F1-Score=0.9426

**Sensitivity (Recall) by Class:**

- FROG: 0.8631
- HEEL: 1.0000
- KANGAROO: 1.0000
- SCISSOR: 1.0000
- SIDEKICK: 1.0000
- TIPTOE: 0.8149

---

### XGBoost

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 1.0000 | 0.7772 | 0.8746 | 745 |
| HEEL | 0.9805 | 1.0000 | 0.9902 | 655 |
| KANGAROO | 0.8028 | 1.0000 | 0.8906 | 688 |
| SCISSOR | 0.7716 | 1.0000 | 0.8711 | 304 |
| SIDEKICK | 1.0000 | 1.0000 | 1.0000 | 646 |
| TIPTOE | 1.0000 | 0.8114 | 0.8959 | 562 |

**Accuracy:** 0.9244
**Macro Avg:** Precision=0.9258, Recall=0.9314, F1-Score=0.9204
**Weighted Avg:** Precision=0.9395, Recall=0.9244, F1-Score=0.9242

**Sensitivity (Recall) by Class:**

- FROG: 0.7772
- HEEL: 1.0000
- KANGAROO: 1.0000
- SCISSOR: 1.0000
- SIDEKICK: 1.0000
- TIPTOE: 0.8114

---

### SVM

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 1.0000 | 0.8658 | 0.9281 | 745 |
| HEEL | 0.9562 | 1.0000 | 0.9776 | 655 |
| KANGAROO | 0.8731 | 1.0000 | 0.9322 | 688 |
| SCISSOR | 0.6414 | 1.0000 | 0.7815 | 304 |
| SIDEKICK | 1.0000 | 1.0000 | 1.0000 | 646 |
| TIPTOE | 1.0000 | 0.6441 | 0.7835 | 562 |

**Accuracy:** 0.9167
**Macro Avg:** Precision=0.9118, Recall=0.9183, F1-Score=0.9005
**Weighted Avg:** Precision=0.9375, Recall=0.9167, F1-Score=0.9158

**Sensitivity (Recall) by Class:**

- FROG: 0.8658
- HEEL: 1.0000
- KANGAROO: 1.0000
- SCISSOR: 1.0000
- SIDEKICK: 1.0000
- TIPTOE: 0.6441

---

### Decision_Tree

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 0.8641 | 0.8362 | 0.8499 | 745 |
| HEEL | 0.8649 | 0.9969 | 0.9262 | 655 |
| KANGAROO | 0.8235 | 0.9767 | 0.8936 | 688 |
| SCISSOR | 0.9430 | 0.9803 | 0.9613 | 304 |
| SIDEKICK | 1.0000 | 0.9923 | 0.9961 | 646 |
| TIPTOE | 0.9658 | 0.6032 | 0.7426 | 562 |

**Accuracy:** 0.8961
**Macro Avg:** Precision=0.9102, Recall=0.8976, F1-Score=0.8950
**Weighted Avg:** Precision=0.9034, Recall=0.8961, F1-Score=0.8910

**Sensitivity (Recall) by Class:**

- FROG: 0.8362
- HEEL: 0.9969
- KANGAROO: 0.9767
- SCISSOR: 0.9803
- SIDEKICK: 0.9923
- TIPTOE: 0.6032

---

### KNN

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 1.0000 | 0.7329 | 0.8459 | 745 |
| HEEL | 0.8710 | 1.0000 | 0.9311 | 655 |
| KANGAROO | 0.7765 | 1.0000 | 0.8742 | 688 |
| SCISSOR | 0.7506 | 1.0000 | 0.8575 | 304 |
| SIDEKICK | 1.0000 | 1.0000 | 1.0000 | 646 |
| TIPTOE | 1.0000 | 0.6495 | 0.7875 | 562 |

**Accuracy:** 0.8900
**Macro Avg:** Precision=0.8997, Recall=0.8971, F1-Score=0.8827
**Weighted Avg:** Precision=0.9128, Recall=0.8900, F1-Score=0.8863

**Sensitivity (Recall) by Class:**

- FROG: 0.7329
- HEEL: 1.0000
- KANGAROO: 1.0000
- SCISSOR: 1.0000
- SIDEKICK: 1.0000
- TIPTOE: 0.6495

---

### Naive_Bayes

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| FROG | 0.9665 | 0.7369 | 0.8363 | 745 |
| HEEL | 0.9444 | 0.9344 | 0.9394 | 655 |
| KANGAROO | 0.8003 | 0.8503 | 0.8245 | 688 |
| SCISSOR | 0.4583 | 0.9934 | 0.6272 | 304 |
| SIDEKICK | 0.9557 | 0.6347 | 0.7628 | 646 |
| TIPTOE | 0.8177 | 0.8221 | 0.8199 | 562 |

**Accuracy:** 0.8111
**Macro Avg:** Precision=0.8238, Recall=0.8286, F1-Score=0.8017
**Weighted Avg:** Precision=0.8626, Recall=0.8111, F1-Score=0.8194

**Sensitivity (Recall) by Class:**

- FROG: 0.7369
- HEEL: 0.9344
- KANGAROO: 0.8503
- SCISSOR: 0.9934
- SIDEKICK: 0.6347
- TIPTOE: 0.8221

---

## Interpretation Guide

- **Precision:** Proportion of positive predictions that are correct
- **Recall (Sensitivity):** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of samples in each class
