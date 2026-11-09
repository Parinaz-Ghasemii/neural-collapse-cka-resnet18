# English summary of the thesis

**Geometrical Analysis of Deep Neural Networks**
Parinaz Ghasemi · B.Sc. in Computer Science, University of Tehran (College of Science, School of Mathematics, Statistics, and Computer Science) · Supervisor: Dr. Arta Jamshidi · September 2026
Full thesis (Persian): [`thesis.pdf`](thesis.pdf)

## Motivation

Deep networks are usually monitored through accuracy and loss. Reaching high accuracy, however, does not imply that the internal representations have stopped changing. Papyan et al. (2020) showed that late in training, features collapse into a highly symmetric geometry (*Neural Collapse*). This thesis asks **when** the representations of a network stabilize, whether all layers do so at the same time, and how this relates to the emergence of Neural Collapse.

## Background (Chapters 1–2)

- **Linear CKA** measures similarity between two sets of representations through their Gram matrices. It is invariant to orthogonal transformations and isotropic scaling, and avoids the limitations of CCA-based methods in high dimensions. Here it compares each layer at epoch *t* with the same layer in the final model.
- **Neural Collapse** has four components: NC1 (within-class variability vanishes), NC2 (class means form a simplex equiangular tight frame), NC3 (classifier weights align with class means) and NC4 (the classifier behaves like a nearest-class-center rule).
- The **Terminal Phase of Training (TPT)** starts when training error reaches zero but the loss keeps decreasing.

## Method (Chapter 3)

- ResNet-18 (CIFAR variant) trained from scratch on CIFAR-10 for 350 epochs with SGD, momentum 0.9, weight decay 5e-4, cosine learning-rate annealing from 0.1 to 1e-5, and standard augmentation. 113 checkpoints were saved. Final test accuracy: 95.98 %.
- **CKA** was computed for conv1 and layer1–layer4 (spatially average-pooled features) and for the 512-d avgpool features, on a fixed subset of 1,000 test images.
- **NC1–NC4** were computed on the 512-d penultimate features of the training set (NC4 on the test set).
- **Stabilization epoch (strict):** the first checkpoint from which a criterion holds until the end of training. NC metrics: ≤ 1.2 × final value (NC4: ≤ 0.01). CKA: ≥ 0.95 and ≥ 0.99. **TPT start:** first epoch with 100 % training accuracy (epoch 313).

## Results (Chapter 4)

| Quantity | Stabilization epoch |
|---|---|
| NC3 | 268 |
| NC4 | 282 |
| NC2 angular / NC2 combined | 284 / 288 |
| CKA (avgpool) ≥ 0.95 / ≥ 0.99 | 290 / 306 |
| NC2 norm | 312 |
| **TPT starts** | **313** |
| NC1 | 320 |

| Layer | CKA ≥ 0.95 | CKA ≥ 0.99 |
|---|---|---|
| conv1 | 70 | 266 |
| layer1 | 274 | 300 |
| layer2 | 270 | 302 |
| layer3 | 298 | 314 |
| layer4 | 290 | 306 |

- Layers do not stabilize simultaneously: conv1 becomes similar to its final state much earlier than the others, but at the stricter 0.99 threshold all layers stabilize only in the last ~85 epochs.
- Most Neural Collapse components stabilize before TPT begins, in the order NC3 → NC4 → NC2, whereas NC1 stabilizes 7 epochs after TPT starts.
- Zero training error therefore does not mark the end of representation change: loss and within-class geometry keep evolving after accuracy saturates.
- CKA and NC measure different things (similarity to the final representation vs. class geometry); their stabilization times are close (288–306) but not equal, so they should not be treated as one event.

## Conclusions and future work (Chapter 5)

Training is better described as several coupled but non-synchronous processes (task performance, representation change, geometric organization) than as a single convergence point. "End of training" depends on what is being measured. Future work: other datasets (CIFAR-100), architectures (VGG-16, Wide ResNet), eigenspace/subspace analysis of the features, alternative CKA references and alternative stabilization definitions.

## Limitations

One architecture, one dataset, one run. Layer-wise CKA uses spatially pooled features. NC features are extracted under training augmentation. See the repository README for details.
