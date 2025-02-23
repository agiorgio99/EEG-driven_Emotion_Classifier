# EEG-Based Emotion Recognition Using LLaMA

This repository contains a Python script for **EEG-based emotion recognition** leveraging a [LLaMA-based](https://arxiv.org/abs/2302.13971) language model, fine-tuned to classify **valence and arousal** into three discrete levels each. The code is adapted from a Jupyter Notebook (originally used in an interactive environment) and shows data preprocessing, dataset loading, model definition, training, and testing steps.

## Method Overview

1. **Data Preprocessing (DEAP dataset)**  
   - Loads EEG signals for each subject.  
   - Applies **z-score normalization**.  
   - Converts signals into overlapping segments.  
   - **Quantizes** the EEG amplitudes into discrete symbols (e.g., binary codes).  

2. **Modeling**  
   - Uses a **LLaMA**-based model in 4-bit precision.  
   - Adds a **LoRA** (Low-Rank Adaptation) layer for parameter-efficient finetuning.  
   - Outputs **6 logits**: (3 for valence + 3 for arousal).  

3. **Training and Testing**  
   - Splits the dataset into training, validation, and test sets.  
   - Minimizes **cross-entropy loss** on the valence/arousal classification.  
   - Tracks **accuracy** (valence, arousal, and overall).  

## Key Results

- The script logs training accuracy and validation accuracy at each epoch.  
- After training, a test evaluation reports final accuracy for valence, arousal, and overall classification.
- Typical results (for a small subset of the DEAP dataset) might range from 65–75% overall accuracy, though actual performance varies with hyperparameters, dataset splits, and model size.

## Installation

Below are the main packages you need. (The script includes commands to install them if run in a Python context that supports shell commands, but you can also install them manually.)

1. [bitsandbytes](https://pypi.org/project/bitsandbytes/)  
2. [transformers](https://pypi.org/project/transformers/)  
3. [accelerate](https://pypi.org/project/accelerate/)  
4. [peft](https://pypi.org/project/peft/)  
5. [python-dotenv](https://pypi.org/project/python-dotenv/)  
6. [einops](https://pypi.org/project/einops/)  
7. [scikit-learn](https://pypi.org/project/scikit-learn/)  
8. [scipy](https://pypi.org/project/scipy/)  
9. [matplotlib](https://pypi.org/project/matplotlib/)  
10. [tabulate](https://pypi.org/project/tabulate/)  
11. [tqdm](https://pypi.org/project/tqdm/)  
12. [huggingface_hub](https://pypi.org/project/huggingface-hub/)

You can install them all via:

```bash
pip install -U bitsandbytes transformers accelerate peft python-dotenv einops scikit-learn scipy matplotlib tabulate tqdm huggingface_hub
```
## How to Run

1. **Clone or download this repository.**  

2. **Acquire the DEAP dataset and place its preprocessed Python files (.dat files) in ./DEAP_Dataset/data_preprocessed_python/.**  

3. **Obtain LLaMA weights (you must have authorization from Meta if using official weights). By default, the script references a model named "Llama-3.1-8B-Instruct", which should be present or downloaded locally.**
   
4. **Set up any environment variables (e.g., HUGGINGFACE_TOKEN) in a .env file if needed.**

5. **Run all the cells in the jupyter notebook.**
   This will:
   - Preprocess the EEG data.  
   - Create training/validation/test splits.
   - Train the LLaMA-based classifier.
   - Evaluate performance and show final results.
   
## Usage Notes

- Hyperparameters such as num_bins, window_size, overlap, and training settings (batch_size, learning_rate, epochs) can be changed in the script.
- Logging and checkpoints are saved in a Trainings/ directory.
- System Requirements: GPU with sufficient memory is recommended (10GB+). The script uses 4-bit quantization to reduce GPU usage.


## Acknowledgments

- DEAP Dataset for emotion recognition from EEG data.
- Meta AI for the LLaMA foundation model.
- LoRA (Low-Rank Adaptation) technique for efficient finetuning.