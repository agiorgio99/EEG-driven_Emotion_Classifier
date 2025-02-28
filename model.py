import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shutil

from datetime import datetime
from tqdm import tqdm
from IPython.display import Javascript, display

# For LLaMA and quantization
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# For LoRA-based fine-tuning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Model Definition
class LlamaEmotionClassifier(nn.Module):

    def __init__(self, model_path, device, train_folder):

        """
        Initialize the Llama-based model for valence & arousal 
        classification into 3 classes (low,mid,high) for valence, and 3 classes for arousal 
        => total 6 logits per sample.

        Args:
            model_path (str): Local path to the pre-trained Llama model.
            device (str): "cuda" or "cpu".
            train_folder (str): Where model trainings are saved.

        Returns:
            LlamaEmotionClassifier: An instance of the LlamaEmotionClassifier model with a linear regression head.
        """

        super(LlamaEmotionClassifier, self).__init__()
        self.model_path = model_path
        self.device = device
        self.train_folder = train_folder

        print(f"Loading model on device: {self.device}")

        # Load in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )

        # 1) Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            use_cache=False
        )

        # 2) Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        # 3) LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=8,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
        )
        self.model = get_peft_model(model, peft_config).to(self.device)

        # 4) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 5) Classification head => hidden_size => 6
        hidden_size = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_size, 6).to(self.device)

        print("LlamaEmotionClassifier initialized.")



    def forward(self, input_ids, attention_mask=None):

        """
        Forward pass for the Llama model.
        Uses the last hidden state for classification, 
        but excludes padding tokens from the mean-pooling.
        Returns a 2D (batch_size, 2) => valence, arousal in [0,1].

        Args:
            input_ids (torch.Tensor): Input tensor for the model.
            attention_mask (torch.Tensor): Attention mask to exclude padding tokens.

        Returns:
            torch.Tensor: Model output with reduced dimensionality for emotion classification.
        """
        # Pass through Llama model => [batch_size, seq_len, hidden_size]
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        last_hidden = outputs.hidden_states[-1]

        # Masked mean-pool across seq_len (exclude padding by using attention_mask)
        # 1) Expand attention_mask for broadcasting: [batch_size, seq_len] => [batch_size, seq_len, hidden_size]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()  # float for multiplication

        # 2) Element-wise multiply to zero out padded positions, then sum over seq_len
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)  # [batch_size, hidden_size]

        # 3) Count of valid (non-padded) tokens per sequence
        sum_mask = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)      # [batch_size, hidden_size]

        # 4) Divide by count of non-padded tokens
        pooled = sum_embeddings / sum_mask  # [batch_size, hidden_size]

        # Output => [batch_size, 6]
        logits = self.fc(pooled)

        # return raw logits (CrossEntropyLoss will handle log-softmax)
        return logits

    def compute_loss_and_accuracy(self, logits, labels):
        """
        logits: [batch_size, 6]
          first 3 => valence logits
          next 3  => arousal logits
        labels: [batch_size, 2] => [valence_class, arousal_class]

        Returns: loss, val_acc, aro_acc, overall_acc
        """
        batch_size = logits.size(0)
        val_logits = logits[:, :3]  # shape [batch_size, 3]
        aro_logits = logits[:, 3:]  # shape [batch_size, 3]

        val_labels = labels[:, 0]   # shape [batch_size]
        aro_labels = labels[:, 1]   # shape [batch_size]

        ce_loss = nn.CrossEntropyLoss()
        val_loss = ce_loss(val_logits, val_labels)
        aro_loss = ce_loss(aro_logits, aro_labels)
        loss = val_loss + aro_loss

        # Accuracy
        val_preds = torch.argmax(val_logits, dim=1)
        aro_preds = torch.argmax(aro_logits, dim=1)
        val_correct = (val_preds == val_labels).sum().item()
        aro_correct = (aro_preds == aro_labels).sum().item()
        val_acc = val_correct / batch_size
        aro_acc = aro_correct / batch_size

        # Overall => both must match
        both_correct = ((val_preds == val_labels) & (aro_preds == aro_labels)).sum().item()
        overall_acc = both_correct / batch_size

        return loss, val_acc, aro_acc, overall_acc
    


    def train_model(self, train_loader, val_loader, hparams, index_dict):
        """
        Fine-tune using cross-entropy classification on valence/arousal.
        Trains the model and saves:
        - Model weights
        - Hyperparameters
        - Training curves
        - Notebook (`.ipynb`)
        - Model instance (for reloading without redefining the class)
        """

        print("Starting training with cross-entropy classification...")

        # Create subfolder for experiment
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S") 
        self.save_folder = os.path.join(self.train_folder, experiment_name)
        os.makedirs(self.save_folder, exist_ok=True)
        print(f"Experiment folder created: {self.save_folder}")

        # Save hyperparams
        self.save_hyperparams(hparams)
        
        index_path = os.path.join(self.save_folder, "dataset_indices.json")
        with open(index_path, "w") as f:
            json.dump(index_dict, f)

        optimizer = torch.optim.AdamW(self.parameters(), lr=hparams["learning_rate"])
        
        # For plotting
        self.train_losses = []
        self.val_losses = []
        self.train_overall_accs = []
        self.val_overall_accs = []

        for epoch in range(hparams["epochs"]):
            # -------------------------
            #       TRAINING
            # -------------------------
            self.train()
            running_loss = 0.0
            running_val_acc = 0.0
            running_aro_acc = 0.0
            running_overall_acc = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hparams['epochs']} [TRAIN]", leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self(input_ids, attention_mask)
                loss, val_acc, aro_acc, overall_acc = self.compute_loss_and_accuracy(logits, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_val_acc += val_acc
                running_aro_acc += aro_acc
                running_overall_acc += overall_acc

                train_pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "val_acc": f"{val_acc:.3f}",
                    "aro_acc": f"{aro_acc:.3f}",
                    "overall": f"{overall_acc:.3f}"
                })

            n_batches = len(train_loader)
            epoch_train_loss = running_loss / n_batches
            epoch_train_val_acc = running_val_acc / n_batches
            epoch_train_aro_acc = running_aro_acc / n_batches
            epoch_train_overall_acc = running_overall_acc / n_batches

            # -------------------------
            #       VALIDATION
            # -------------------------
            self.eval()
            val_running_loss = 0.0
            val_running_val_acc = 0.0
            val_running_aro_acc = 0.0
            val_running_overall_acc = 0.0

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hparams['epochs']} [VAL]", leave=False)
                for batch in val_pbar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = self(input_ids, attention_mask)
                    loss, val_acc, aro_acc, overall_acc = self.compute_loss_and_accuracy(logits, labels)

                    val_running_loss += loss.item()
                    val_running_val_acc += val_acc
                    val_running_aro_acc += aro_acc
                    val_running_overall_acc += overall_acc

                    val_pbar.set_postfix({
                        "val_loss": f"{loss.item():.3f}",
                        "val_acc": f"{val_acc:.3f}",
                        "aro_acc": f"{aro_acc:.3f}",
                        "overall": f"{overall_acc:.3f}"
                    })

            n_val_batches = len(val_loader)
            epoch_val_loss = val_running_loss / n_val_batches
            epoch_val_val_acc = val_running_val_acc / n_val_batches
            epoch_val_aro_acc = val_running_aro_acc / n_val_batches
            epoch_val_overall_acc = val_running_overall_acc / n_val_batches

            # Store for plots
            self.train_losses.append(epoch_train_loss)
            self.val_losses.append(epoch_val_loss)
            self.train_overall_accs.append(epoch_train_overall_acc)
            self.val_overall_accs.append(epoch_val_overall_acc)

            # Print a short summary
            print(f"\n[Epoch {epoch+1}/{hparams['epochs']}]")
            print(f"  Train: loss={epoch_train_loss:.4f} | val_acc={epoch_train_val_acc:.4f} "
                  f"| aro_acc={epoch_train_aro_acc:.4f} | overall_acc={epoch_train_overall_acc:.4f}")
            print(f"  Val:   loss={epoch_val_loss:.4f}   | val_acc={epoch_val_val_acc:.4f} "
                  f"| aro_acc={epoch_val_aro_acc:.4f}   | overall_acc={epoch_val_overall_acc:.4f}\n")

        # At the end of training, save model weights, definition & code
        self.save_model_weights()
        self.save_model_definition()
        self.save_notebook()

        self.plot_training_curves()

        print("Training complete.")

    def test_model(self, test_loader):
        """
        Runs inference on the test set:
         - Prints per-batch overall accuracy
         - Prints  and returns final accuracies (valence, arousal, overall)
        """
        self.eval()

        batch_accuracies = []
        all_val_preds = []
        all_val_labels = []
        all_aro_preds = []
        all_aro_labels = []

        print("\n--- TESTING ---")
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc="Testing", leave=False)):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self(input_ids, attention_mask)
                # We won't compute full loss here, just get accuracy
                _, val_acc, aro_acc, overall_acc = self.compute_loss_and_accuracy(logits, labels)
                batch_accuracies.append(overall_acc)

                val_logits = logits[:, :3]
                aro_logits = logits[:, 3:]
                val_preds = torch.argmax(val_logits, dim=1)
                aro_preds = torch.argmax(aro_logits, dim=1)

                all_val_preds.append(val_preds.cpu().numpy())
                all_val_labels.append(labels[:, 0].cpu().numpy())
                all_aro_preds.append(aro_preds.cpu().numpy())
                all_aro_labels.append(labels[:, 1].cpu().numpy())

                print(f"  Batch {idx+1}/{len(test_loader)} => Overall Acc: {overall_acc:.3f}")

        # Compute final accuracies across the entire test set
        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        all_aro_preds = np.concatenate(all_aro_preds)
        all_aro_labels = np.concatenate(all_aro_labels)

        val_accuracy = (all_val_preds == all_val_labels).mean()
        aro_accuracy = (all_aro_preds == all_aro_labels).mean()
        both_correct = ((all_val_preds == all_val_labels) & (all_aro_preds == all_aro_labels)).sum()
        overall_accuracy = both_correct / len(all_val_preds)

        print(f"\n[Test Summary]")
        print(f"Valence Accuracy: {val_accuracy:.4f}")
        print(f"Arousal Accuracy: {aro_accuracy:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

        return {
            "valence_accuracy": val_accuracy,
            "arousal_accuracy": aro_accuracy,
            "overall_accuracy": overall_accuracy
        }

    # -------------------------------------------------------------------------
    #                    LOCAL SAVING METHODS
    # -------------------------------------------------------------------------

    def save_hyperparams(self, hparams):
        """
        Save hyperparameters as a json in self.save_folder/hyperparams.json
        """
        if not hasattr(self, "save_folder"):
            print("Error: no self.save_folder. Hyperparams not saved.")
            return

        hparam_path = os.path.join(self.save_folder, "hyperparams.json")
        with open(hparam_path, "w") as f:
            json.dump(hparams, f, indent=4)
        print(f"Hyperparameters saved to: {hparam_path}")



    def save_model_weights(self):
        """
        Save model state_dict to 'model_weights.pt' in self.save_folder.
        """
        if not hasattr(self, "save_folder"):
            print("Error: no self.save_folder. Model weights not saved.")
            return

        weights_path = os.path.join(self.save_folder, "model_weights.pt")
        torch.save(self.state_dict(), weights_path)
        print(f"Model weights saved: {weights_path}")



    def save_model_definition(self):
        """
        Save the entire model.py file (including imports and class definition)
        to model_definition.py in self.save_folder.
        """
        model_definition_path = os.path.join(self.save_folder, "model_definition.py")
        
        # Path to the original script
        original_file_path = os.path.abspath(__file__)  # Get the current file path dynamically
        
        try:
            # Copy the exact file to destination
            shutil.copy(original_file_path, model_definition_path)
            print(f"Entire Python file copied to {model_definition_path}")
        except Exception as e:
            print(f"Error copying file: {e}")




    def save_notebook(self):
        """Saves the current Jupyter notebook into the experiment folder."""
        display(Javascript('IPython.notebook.save_checkpoint()'))
        notebook_filename = "EEG_Driven_Emotion_Classifier.ipynb" 
        save_path = os.path.join(self.save_folder, notebook_filename)
        os.system(f"cp {notebook_filename} '{save_path}'")
        print(f"Notebook saved: {save_path}")
    


    def plot_training_curves(self):
        """
        Plot & save training vs. validation curves for loss & overall accuracy
        """
        if not hasattr(self, "train_losses") or not hasattr(self, "val_losses"):
            return  # no data

        # Loss curve
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))
        # Left: Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("CrossEntropy Loss")
        plt.legend()

        # Right: Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_overall_accs, label="Train Overall Acc")
        plt.plot(epochs, self.val_overall_accs, label="Val Overall Acc")
        plt.title("Overall Accuracy per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()

        if hasattr(self, "save_folder"):
            plot_path = os.path.join(self.save_folder, "training_curves.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Training curves saved to {plot_path}")
        else:
            plt.show()
