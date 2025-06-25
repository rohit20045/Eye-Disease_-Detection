# Eye-Disease_-Detection
This project presents a deep learning-based solution for multi-class eye disease detection using high-resolution fundus images. Leveraging the EfficientNetB6 architecture, the model was trained to classify 8 types of eye conditions from a dataset of nearly 10,000 images.

### Model Overview
- Architecture: EfficientNetB6

- Image Size: 512x512

- Dataset Size: 9,868 labeled fundus images

- Epochs Trained: 50

- Best Validation Accuracy: 76.68%

- Best Validation Loss: 1.1706

- Final Training Accuracy: 97.44%

- Final Training Loss: 0.2792

- Learning Rate (Final Epoch): 1.25e-05

### Target Classes
The model classifies images into the following 8 categories:

- N: Normal
- D: Diabetes
- G: Glaucoma
- C: Cataract
- A: Age-related Macular Degeneration
- H: Hypertension
- M: Pathological Myopia
- O: Other diseases/abnormalities

### Highlights
- Utilizes the power of EfficientNetB6 for balancing performance and efficiency on high-resolution medical images.
- Suitable for applications in medical image classification tasks or as a baseline for related work in ophthalmology AI.

## Dataset 
- Kaggle Link: https://www.kaggle.com/datasets/rohitrawat25/combined-fundus-images.
- Dataset can be downloaded from the kaggle page.
