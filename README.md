# Fashion Style Recommendation Using CLIP, Cosine Similarity, and HDBSCAN

## Overview

This project builds a **fashion recommendation system** based on **user-uploaded images**. It uses **CLIP embeddings**, **cosine similarity**, and **HDBSCAN clustering** to recommend items from a catalog that match the style and formal attire of the input images.

Specifically, three user images (men dressed in formal shirts and pants) were used as input to extract their style features and find visually and semantically similar products.

## Project Pipeline

1. **Input Images**: Three images of men in formal attire were provided by the user.
2. **Feature Extraction**: Images were passed through OpenAI's **CLIP ViT-B/32** model to generate 512-dimensional embeddings.
3. **Similarity Computation**: Cosine similarity was used to measure the closeness between user embeddings and catalog embeddings.
4. **Clustering**: **HDBSCAN** was applied to group catalog products into meaningful style-based clusters.
5. **Recommendation**: Top matching items were selected based on high similarity scores and cluster membership.

## Technologies Used

- **CLIP (Contrastive Language-Image Pretraining)** - for semantically rich feature extraction
- **Cosine Similarity** - for comparing feature vectors
- **HDBSCAN** - for unsupervised clustering of similar styles
- **Python** - programming language
- **PyTorch** - deep learning framework
- **Pandas, NumPy** - data processing
- **Matplotlib, Seaborn** - visualization

## Why These Techniques?

| Technique | Why it was used |
|-----------|-----------------|
| CLIP | Captures both visual and semantic meaning from fashion images |
| Cosine Similarity | Measures closeness in high-dimensional feature space |
| HDBSCAN | Identifies meaningful style clusters without needing a fixed number of clusters |

## Dataset

A catalog dataset was used, containing product metadata such as:
- ID
- Gender
- MasterCategory
- SubCategory
- ArticleType
- BaseColor
- Season
- Year
- Usage
- ProductDisplayName

However, **only the product images** were used in this project. Metadata was not needed because **visual style** was extracted directly from images using CLIP, making the system more flexible and unsupervised.

## How to Run

1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install torch torchvision hdbscan pandas matplotlib scikit-learn
   ```
3. Place your input images in the `input/` folder.
4. Run the main script:
   ```bash
   python recommend.py
   ```
5. View the top recommended products displayed as output.

## Example Output

The system recommends formal shirts and pants visually matching the style of the uploaded images, showcasing highly relevant and semantically aligned products.

## Folder Structure

```
fashion-recommendation/
│
├── input/                # User-provided images
├── catalog/              # Product images dataset
├── recommend.py          # Main code for feature extraction, clustering, and recommendation
├── README.md             # This file
└── outputs/              # Recommendation results
```

## Future Work

- Add metadata filtering (e.g., filter by gender, season, etc.)
- Add textual query support ("Show me winter formal jackets")
- Build a web UI to upload images and view recommendations live

## References

- Radford, A., et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.
- McInnes, L., Healy, J., & Astels, S. *hdbscan: Hierarchical density-based clustering*. Journal of Open Source Software, 2017.
- OpenAI's CLIP model documentation.
