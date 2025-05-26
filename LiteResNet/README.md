# LiteResNet

LiteResNet is a lightweight adaptation of the ResNet architecture from [Microsoft Research](https://arxiv.org/abs/1512.03385) designed for efficient image classification. The model consists of **4,919,626 parameters**, optimizing performance while maintaining computational efficiency. This repository contains the necessary notebooks and models used for training, fine-tuning, and predicting.

## Repository Contents

### **Notebooks**
- **`training.ipynb`** - Kaggle notebook used to train the LiteResNet model.
- **`fine_tuning.ipynb`** - Kaggle notebook used to fine-tune the trained model.
- **`generate_predictions.ipynb`** - Kaggle notebook used to perform test-time augmentation, generating predictions over **7 different variations of an image** and providing the final output.
- **`graphs.ipynb`** - Notebook containing plots for training and fine-tuning data.
- **`nolabel_analysis.ipynb`** - Notebook containing analysis performed on no-label test data to determine suitable augmentations.

### **Data and Outputs**
- **`loss & accuracy/`** - Folder containing training and fine-tuning logs.
- **`models/`** - Folder containing generated models:
  - **`best_model.pth`** - Model obtained after initial training.
  - **`best_model_finetuned.pth`** - Final model obtained after fine-tuning.

### **Documentation**
- **`LiteResNet-paper.pdf`** - Technical report detailing the architecture, training methodology, evaluation metrics, and results.

## Training Phase (Hyperparameters)
| Parameter          | Value                    |
|-------------------|-------------------------|
| Epochs           | 125                      |
| Optimizer        | AdamW                    |
| Learning Rate    | 1e-3                     |
| Weight Decay     | 0.01                     |
| Scheduler        | CosineAnnealingLR        |
| Min Learning Rate | 1e-6                     |

## Fine-Tuning Phase (Hyperparameters)
| Parameter          | Value                    |
|-------------------|-------------------------|
| Epochs           | 30                       |
| Optimizer        | SGD                      |
| Learning Rate    | 1e-4                     |
| Momentum        | 0.9                       |
| Weight Decay     | 0.01                     |
| Scheduler        | CosineAnnealing          |
| Min Learning Rate | 1e-6                     |

## Usage
To utilize the models or run the notebooks, ensure that all necessary dependencies are installed and execute the notebooks in sequence.

## Installation
Clone the repository and navigate to the project directory:
```sh
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

Ensure all dependencies are installed:
```sh
pip install -r requirements.txt
```

## Training and Fine-Tuning
Run the training notebook:
```sh
jupyter notebook training.ipynb
```
Run the fine-tuning notebook:
```sh
jupyter notebook fine_tuning.ipynb
```

## Predictions 
To generate predictions using test-time augmentation:
```sh
jupyter notebook generate_predictions.ipynb
```

## Model Details
- **Architecture**: LiteResNet
- **Total Parameters**: 4,919,626

## Contributions
Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

## License
This project is open-source and available under the MIT License.

