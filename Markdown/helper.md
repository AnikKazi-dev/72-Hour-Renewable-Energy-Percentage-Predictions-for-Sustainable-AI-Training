# scFL-Green: Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning

**Presented by:**  
Mahfuzur Rahman Chowdhury  
**Student ID:** 23305223

**Supervised by:**  
Dr. Anne Hartebrodt

**📍 Presented at:**  
Biomedical Network Science Lab  
Department of Artificial Intelligence in Biomedical Engineering (AIBE)  
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

---

## Index

- Introduction
- Frameworks
- Datasets
- Federated Learning
- Optimization
- Optimization Results
- Fine Tuning
- Fine Tuning Results
- Conclusion

---

# scFL-Green: Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning

---

## Single-Cell Data / Single-Cell Classification

**Single-Cell Classification**  
- Categorizing individual cells

**Single-Cell Data**  
- Gene expression profiling at the single-cell level

**Single-Cell Classification in Deep Learning**  
- Gene expression as features  
- Cell type as Labels

---

## Deep Learning Framework (scDLC)

**scDLC:** a deep learning framework to classify large sample single-cell RNA-seq data (Zhou et al., 2022; BMC Genomics) [1]

**Framework:**
- Fully Connected Layer
- Two LSTM Layers  
  - 64 neurons
- Fully Connected Layer
- Softmax

> [1] Y. Zhou, M. Peng, B. Yang, T. Tong, B. Zhang, and N. Tang, “scDLC: a deep learning framework to classify large sample single-cell RNA-seq data,” BMC Genomics, vol. 23, no. 1, Jul. 2022, doi: 10.1186/s12864-022-08715-1.

---

## Deep Learning Framework (scCDCG)

**scCDCG:** Deep Structural Clustering for Single-Cell RNA-seq via Deep Cut-informed Graph Embedding (Xu et al., 2024) [2]

**Framework:**
- **Autoencoder-Based Feature Learning Module**  
  - Encoder: Multi-layer feedforward neural network (256 → 64)  
  - Decoder: Mirrors encoder structure (64 → 256)
- **Graph Embedding Module (Deep Cut-Informed Graph Learning)**  
  - Probability Metrics Graph (PMG): Captures covariance relationships  
  - Spatial Metrics Graph (SMG): Captures cosine similarity between cells
- **Classification Head (Fully Connected Layer)**  
  - Dropout layer (0.3) for regularization

> [2] P. Xu et al., “scCDCG: Efficient Deep Structural Clustering for single-cell RNA-seq via Deep Cut-informed Graph Embedding,” arXiv.org, Apr. 09, 2024. https://arxiv.org/abs/2404.06167

---

## Deep Learning Framework (scSMD)

**scSMD:** A Deep Learning Framework for Single-Cell Clustering (Cui et al., 2025; BMC Bioinformatics) [3]

**Framework:**
- Fully Connected Encoder (Encoder)  
  - 1024 neurons
- Latent Space Representation (64 neurons)
- Dropout Layer (0.3 probability)
- Fully Connected Classifier

**Optional:**
- Self-Supervised Learning Module (CellNet)
- Multi-Dilated Attention Gate Module

> [3] X. Cui et al., “scSMD: a deep learning method for accurate clustering of single cells based on auto-encoder,” BMC Bioinformatics, vol. 26, no. 1, Jan. 2025, doi: 10.1186/s12859-025-06047-x.

---

## Deep Learning Framework (ACTINN)

**ACTINN:** Automated Cell Type Identification Using Neural Networks (Ma et al., 2019; Bioinformatics) [4]

**Framework:**
- Fully Connected (100 neurons with ReLU)
- Fully Connected (50 neurons with ReLU)
- Fully Connected (25 neurons with ReLU)
- Output (SoftMax)

> [4] F. Ma and M. Pellegrini, “ACTINN: automated identification of cell types in single cell RNA sequencing,” Bioinformatics, vol. 36, no. 2, pp. 533–538, Jul. 2019, doi: 10.1093/bioinformatics/btz592.

---

## Dataset Description

| No. | Tissue      | Data Source | Cell Count | Cell Classes |
|---:|-------------|-------------|-----------:|-------------:|
| A  | Blood       | cellxgene   | 108,717    | 15 |
| B  | Hippocampus | NCBI        | 1,190      | 3 |
| C  | Pancreas    | NCBI        | 1,895      | 9 |
| D  | Blood       | cellxgene   | 66,985     | 11 |

> [5] CellXGene Data Portal: https://cellxgene.cziscience.com/collections/0aab20b3-c30c-4606-bd2e-d20dae739c45  
> [6] CellXGene Data Portal: https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b

---

# scFL-Green: Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning

---

## Federated Learning (Concept)

**Server**

**Client  Client  Client  Client**

---

## Federated Learning

**Creates Model**

---

## Federated Learning

*(Process depiction)*

---

## Federated Learning

*(Process depiction)*

---

## Federated Learning

*(Process depiction)*

---

## Federated Learning Algorithm

**Local Models**

---

## Federated Learning

**Global Model**

---

## FedAvg [7]

**Local Model 1 — Local Model 2 — Local Model 3**

> [7] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Y. Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” arXiv.org, Feb. 17, 2016. https://arxiv.org/abs/1602.05629

---

## FedAvg [7]

**Local Model 1 — Local Model 2 — Local Model 3**  
**Average → Global Model**

> [7] H. B. McMahan et al., 2016. https://arxiv.org/abs/1602.05629

---

## Dataset Split (A & D)

**Train** → **Fine Tune**  
Clients: 1–11  
**Test**

---

## Dataset Split (B & C)

**Train**  
Clients: 1–5  
**Test**

---

## Deep Learning vs Federated Learning (Accuracy)

*(Chart placeholder)*

---

## Deep Learning vs Federated Learning (F1)

*(Chart placeholder)*

---

# scFL-Green: Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning

---

## Deep Learning vs Federated Learning (Carbon Emission)

*(Chart placeholder)*

---

# scFL-Green: Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning

---

## Small Batch

Using small batch size (Instead of using batch sizes of 1024, using 32 or 64)

- Faster Convergence
- Lower Energy Per Update

---

## Small Batch (Results)

- **scSMD:** Batch = 32; Batch = 16  
- **ACTINN:** Batch = 32; Batch = 16  
- **scCDCG:** Batch = 32; Batch = 16  
- **scDLC:** Batch = 32; Batch = 16

*(Chart placeholders)*

---

## Mixed Precision

Using lower precision (using 16-bit instead of 32-bit)

- Faster Computation
- Lower Power Consumption

---

## Reduced Model Complexity

Removing redundant layers; pruning unnecessary parameters

- Fewer Parameters
- Smaller Models Train Faster
- Less Transmission Cost

---

## Reduced Model Complexity (Examples)

- **scSMD:** Latent 64 → Latent 48
- **ACTINN:** Layer Size 100, 50, 25 → 50, 25, 12
- **scCDCG:** Encoder [256, 64] / Decoder [64, 256] → Encoder [128, 48] / Decoder [48, 128]
- **scDLC:** LSTM size 64, Layers 2 → LSTM size 48, Layers 1

---

## Federated Learning Accuracy for Different Methods by Frameworks

*(Chart placeholder)*

---

## Federated Learning F1 for Different Methods by Frameworks

*(Chart placeholder)*

---

## Federated Learning Emissions for Different Methods by Frameworks

*(Chart placeholder)*

---

## Energy-Adjusted Performance Score (EAPS)

**EAPS = F1-score / Carbon Emission**

---

## Highest EAPS Scored Method for Frameworks

| Frameworks | Dataset A | Dataset B | Dataset C | Dataset D |
|------------|-----------|-----------|-----------|-----------|
| scDLC  | Reduce Complexity | Small Batch | Reduce Complexity | Reduce Complexity |
| scCDCG | Reduce Complexity | Mixed Precision | Small Batch | Reduce Complexity |
| scSMD  | Reduce Complexity | Mixed Precision | Mixed Precision | Reduce Complexity |
| ACTINN | Reduce Complexity | Small Batch | Small Batch | Mixed Precision |

---

# scFL-Green: Optimizing Federated Learning for Single-Cell Classification with Carbon-Aware Fine-Tuning

---

## Fine-Tuning Setup (Datasets A & D)

**Model Initialization**  
**Train** → **Fine Tune**  
Clients: A, B, 1–11  
**Test**

---

## Fine-Tuning Process (Datasets A & D)

**Train** → **Fine Tune**  
Trained using **Federated Learning**  
**Copying Model**  
**Test**

---

## Emission Comparison (Dataset A)

*With vs Without Fine-Tuning*  
*(Chart placeholder)*

---

## Emission Comparison (Dataset D)

*With vs Without Fine-Tuning*  
*(Chart placeholder)*

---

## Accuracy per Epochs (Dataset A)

*With vs Without Fine-Tuning*  
*(Chart placeholder)*

---

## Accuracy per Epochs (Dataset D)

*With vs Without Fine-Tuning*  
*(Chart placeholder)*

---

## Conclusion — Takeaways

- Federated Learning maintains accuracy and F1-score but incurs high carbon emissions
- EAPS improves with:  
  - Small Batch Training  
  - Mixed Precision  
  - Reduced Complexity
- Reduced Complexity is the most effective method in most cases
- Fine-tuning pre-trained FL models on new clients leads to faster convergence

---

## Conclusion — Limitations & Future Work

**Limitations:**
- Limited computational power

**Future Work:**
- HPC (High-Performance Computing) for larger datasets
- Develop a model-sharing platform where users can upload/download models

---

## References

1. Y. Zhou, M. Peng, B. Yang, T. Tong, B. Zhang, and N. Tang, “scDLC: a deep learning framework to classify large sample single-cell RNA-seq data,” *BMC Genomics*, 23(1), Jul. 2022. doi: 10.1186/s12864-022-08715-1.
2. P. Xu et al., “scCDCG: Efficient Deep Structural Clustering for single-cell RNA-seq via Deep Cut-informed Graph Embedding,” *arXiv*, Apr. 09, 2024. https://arxiv.org/abs/2404.06167
3. X. Cui et al., “scSMD: a deep learning method for accurate clustering of single cells based on auto-encoder,” *BMC Bioinformatics*, 26(1), Jan. 2025. doi: 10.1186/s12859-025-06047-x.
4. F. Ma and M. Pellegrini, “ACTINN: automated identification of cell types in single cell RNA sequencing,” *Bioinformatics*, 36(2), pp. 533–538, Jul. 2019. doi: 10.1093/bioinformatics/btz592.
5. CellXGene Data Portal. https://cellxgene.cziscience.com/collections/0aab20b3-c30c-4606-bd2e-d20dae739c45
6. CellXGene Data Portal. https://cellxgene.cziscience.com/collections/e1a9ca56-f2ee-435d-980a-4f49ab7a952b
7. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Y. Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” *arXiv*, Feb. 17, 2016. https://arxiv.org/abs/1602.05629

---

## Thank You

**Any Questions?**

