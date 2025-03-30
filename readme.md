# EEG-Based Emotion Recognition Using LLaMA

The project aims to classify, from the EEG signals in the **DEAP** dataset, **valence and arousal** emotions into three discrete levels each, leveraging **LoRA** finetuning on **LLaMA 3.1 8B** model. The code is now divided into two files: one jupyter notebook (*EEG_Driven_Emotion_Classifier.ipynb*) handling data preprocessing, loading, training, and testing, and a separate python file (*model.py*) containing the model definition.

## Method Overview

1. **Data Preprocessing (DEAP dataset)**
   - Loads EEG signals for each subject from `.dat` files.
   - Applies **z-score normalization**.
   - Converts signals into **overlapping segments**.
   - **Quantizes** the EEG amplitudes into discrete symbols (e.g., binary codes).

2. **Model Architecture**
   1. **Llama Backbone in 4-Bit**  
      - Loads a pre-trained Llama model with **nf4** 4-bit quantization for efficient memory usage.  
      - LoRA fine-tuning targets multiple projection layers with:
      - **r = 16**  
      - **lora_alpha = 8**  
      - **lora_dropout = 0.1**  
      - For the first 10 epochs, the Llama backbone is **frozen** to train only LoRA parameters, then fully unfrozen thereafter.

   2. **Mean + Max Pooling**  
      - After the final hidden states are obtained, the model performs **mean-pooling** and **max-pooling**, then **concatenates** these two outputs to form a single vector.

   3. **Classification Head (4 Logits)**  
      - A fully connected layer projects the concatenated vector into **4 logits**:
      - **2 logits** for **valence** (binary classes: low vs. high),  
      - **2 logits** for **arousal** (binary classes: low vs. high).  

3. **Training and Testing**
   - Splits the dataset into **training**, **validation**, and **test** sets.
   - Minimizes **cross-entropy loss** on the valence/arousal classification.
   - Tracks **accuracy** for valence, arousal, and overall performance.
   - Saves experiment outputs (model weights, logs, plots) in a dedicated folder.

## Key Results

- The code logs training and validation losses/accuracies **per epoch**, and then prints final test results.
- Metrics include **valence accuracy**, **arousal accuracy**, and an **overall accuracy** (both valence and arousal correct).

## Installation

Below are the main packages required. (In the provided scripts, `%pip install` commands are used if running in an environment that supports IPython magic. Otherwise, install via CLI.)

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

1. **Clone or Download This Repository**  
   Make sure the folder structure preserves the code files (`model.py` or `model_definition.py`, plus the main script that handles preprocessing and training).

2. **Acquire the DEAP Dataset**  
   Place the `data_preprocessed_python/` folder under the `./DEAP_Dataset/` directory. Each subject’s `.dat` file should be in `./DEAP_Dataset/data_preprocessed_python/`.

3. **Obtain LLaMA Weights**  
   - You must have authorization from Meta to download official LLaMA weights.
   - By default, the script references a model called `"Llama-3.1-8B-Instruct"` and downloads it locally via:
     ```bash
     huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir Llama-3.1-8B-Instruct --exclude "original/*"
     ```
   - Update the `model_path` argument in the code if you are using a different local or remote model name.

4. **Set Up Environment Variables** (Optional)  
   - If you need a `HUGGINGFACE_TOKEN` for private model repos or for the Hugging Face Hub, place it in a `.env` file or specify it directly in the script.

5. **Run the Scripts**  
   - **Preprocessing and Training**: 
     - The main script in the jupyter notebook loads the DEAP data, preprocesses it (z-score normalization, segmentation, quantization), and then trains the model.   
     - Adjust hyperparameters like `num_bins`, `window_size`, `overlap`, `batch_size`, and `learning_rate` in the code as needed.
   - **Testing**:  
     - After training, the script loads the best saved model weights and runs evaluation on the test split, printing final accuracies and saving a `test_results.txt`.

## Usage Notes

- For Training, execute only the notebook sections: Installations (if not already installed), Imports and Directories (with directories modified as needed), Preprocessing (only the first time), Dataset, Model (which imports from model.py), Dataloader, and Training (where hyperparameters can be modified).
- For Testing, execute only the notebook sections: Installations (if not already installed), Imports and Directories (with directories modified as needed), Dataset, Dataloader, and Testing (with directories modified as needed).
- System Requirements: GPU with sufficient memory is recommended (20GB+). The script uses 4-bit quantization to reduce GPU usage.

## Acknowledgments

- DEAP Dataset for emotion recognition from EEG data.
- Meta AI for the LLaMA foundation model.
- LoRA (Low-Rank Adaptation) technique for efficient finetuning.