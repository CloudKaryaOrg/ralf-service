import os
import pandas as pd
import pickle
import warnings
import psutil  # Add this import
from openai import OpenAI
import humanize
import google.generativeai as genai
from transformers import AutoConfig

if os.environ.get('RALF-SERVICE') != '1':
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoConfig
    from transformers.trainer_callback import TrainerCallback
    from sklearn.model_selection import train_test_split
    from datasets import Dataset, ClassLabel, Features, Value

else:
    class TrainerCallback:          # This was created to fix the build error when LLM libraries are not included
        pass

from nltk.corpus import wordnet # Ensure you have the OpenAI Python client installed
import json
import re

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

OPEN_AI_MODEL = "gpt-4-turbo-preview"
GEMINI_MODEL = "gemini-2.5-flash"
# ---------------------- SYSTEM INFO ----------------------
def get_system_info():
    if os.environ.get('RALF-SERVICE') != '1':
        gpu_available = torch.cuda.is_available()
        gpu_info = f"{torch.cuda.get_device_name(0)}" if gpu_available else "No GPU"
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB" if gpu_available else "N/A"
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    else:
        gpu_available = False
        gpu_info = "N/A"
        gpu_memory = "N/A"
        gpu_count = 0

    ram = humanize.naturalsize(psutil.virtual_memory().total)
    return {
        "GPU Available": "✅ Yes" if gpu_available else "❌ No",
        "GPU Model": gpu_info,
        "GPU Memory": gpu_memory,
        "GPU Count": gpu_count,
        "System RAM": ram
    }

# Define the custom callback for saving the Ralf instance
class RalfSavingCallback(TrainerCallback):
    """
    A custom callback to save the Ralf instance periodically during training.
    """
    def __init__(self, ralf_instance, save_path="ralf_state.pkl"):
        self.ralf_instance = ralf_instance
        self.save_path = save_path

    def on_save(self, args, state, control, **kwargs):
        """
        Event called after a checkpoint is saved.
        """
        print(f"Saving Ralf state at step {state.global_step}...")
        self.ralf_instance.save_state(file_path=self.save_path)
        print("Ralf state saved.")

# Functions related to finetuning/reTraining the model
class RalfTraining:
    def __init__(self):
        self.golden_dataset = None
        self.platinum_dataset = None
        # Add other datasets as needed
        self.other_datasets = {}
        self.model_name = None
        self.trainer = None
        self.num_labels = None
        self.label_to_id = None
        self.id_to_label = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.model = None

    def format_param_size(self, total_params):
        """Formats parameter count with units M, B, or P."""
        if total_params >= 1e15:
            return f"{total_params / 1e15:.2f}P"
        elif total_params >= 1e12:
            return f"{total_params / 1e12:.2f}T" # Using T for trillion
        elif total_params >= 1e9:
            return f"{total_params / 1e9:.2f}B"
        elif total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        else:
            return str(total_params) # Return as string for smaller numbers


    def estimate_param_count(self, model_id="distilbert-base-uncased"):
        """Estimates parameter count for a given model ID."""
        try:
            # Pass HF_TOKEN if available when loading the config
            config = AutoConfig.from_pretrained(model_id, token=self.hf_token)

            # Get common configuration attributes, handling different names
            vocab_size = getattr(config, 'vocab_size', None)
            hidden_size = getattr(config, 'hidden_size', getattr(config, 'dim', None))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', None))
            intermediate_size = getattr(config, 'intermediate_size', getattr(config, 'hidden_dim', None))
            num_attention_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_heads', None))

            if None in [vocab_size, hidden_size, num_layers, intermediate_size, num_attention_heads]:
                 return "Error: Could not get all config attributes for parameter estimation."

            # This is a simplified estimation and may not be perfectly accurate for all models
            # It primarily covers BERT-like architectures

            # Embeddings
            embeddings = vocab_size * hidden_size

            # Transformer layers (simplified estimation covering common components)
            # Attention parameters (simplified: QKV weights + output weights + biases)
            attn_params_per_head = hidden_size * (hidden_size // num_attention_heads) + (hidden_size // num_attention_heads) # QKV per head
            attn_output_per_layer = hidden_size * hidden_size + hidden_size # Output projection + bias
            total_attn_params_per_layer = num_attention_heads * attn_params_per_head * 3 + attn_output_per_layer # 3 for QKV

            # FFN parameters (input weight + output weight + biases)
            ffn_params_per_layer = hidden_size * intermediate_size + intermediate_size * hidden_size + intermediate_size + hidden_size # weights + biases

            # Layer Norm parameters (approximate: 2*hidden_size for gamma and beta)
            layer_norm_params_per_layer = 2 * hidden_size * 2 # Two layer norms per layer typically
            transformer_total = num_layers * (total_attn_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)
            total = embeddings + transformer_total # This estimation might still miss some parameters

            return self.format_param_size(total)

        except Exception as e:
            return f"Error estimating: {e}"

    def get_llm_client(self):
        """Helper method to get the appropriate LLM client."""
        if self.open_api_key:
            return {
                'type': 'openai',
                'client': OpenAI(api_key=self.open_api_key),
                'model': OPEN_AI_MODEL
          }
        elif self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            return {
                'type': 'gemini',
                'client': genai.GenerativeModel(GEMINI_MODEL),
                'model': GEMINI_MODEL
           }
        return None

    def get_llm_response(self, client_info, prompt):
        """Helper method to get responses from either OpenAI or Gemini."""
        try:
           if client_info['type'] == 'openai':
               response = client_info['client'].chat.completions.create(
                    model=client_info['model'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.1,
        )
               return response.choices[0].message.content
           else:  # Gemini
               response = client_info['client'].generate_content(prompt)
               return response.text
        except Exception as e:
           raise Exception(f"Error calling {client_info['type']} API: {str(e)}")

    def load_and_process_data(self, df: pd.DataFrame, text_column: str, label_column: str, model_name: str):
        """
        Loads, processes, and tokenizes the data, and splits it into training and validation sets.

        Args:
            df: The input pandas DataFrame.
            text_column: The name of the column containing the text data.
            label_column: The name of the column containing the labels.
            model_name: The name of the pre-trained model to load (e.g., "bert-base-uncased").
        """
        self.model_name = model_name # Set model_name here

        # Ensure the DataFrame has 'text' and 'label' columns
        if text_column != 'text' or label_column != 'label':
            df = df.rename(columns={text_column: 'text', label_column: 'label'})

        # Determine unique labels and create mappings
        unique_conditions = df['label'].unique().tolist()
        self.num_labels = len(unique_conditions)
        self.label_to_id = {condition: i for i, condition in enumerate(unique_conditions)}
        self.id_to_label = {i: condition for i, condition in enumerate(unique_conditions)}

        # Map string labels to integer IDs
        df['label'] = df['label'].map(self.label_to_id)

        # Select only the 'text' and 'label' columns for the dataset
        dataset_df = df[['text', 'label']]

        # Convert pandas DataFrame to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(dataset_df)

        # Convert the 'label' column to ClassLabel
        features = hf_dataset.features.copy()
        features['label'] = ClassLabel(num_classes=self.num_labels, names=unique_conditions)
        hf_dataset = hf_dataset.cast(features)


        # Split the dataset into training and validation sets
        train_df, val_df = train_test_split(
            dataset_df,
            test_size=0.2,
            random_state=42, # for reproducibility
            stratify=dataset_df['label'] # Stratify to maintain class distribution
        )

        # Convert split DataFrames to Hugging Face Datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        # Initialize the tokenizer using the model name
        if self.model_name is None:
            raise ValueError("model_name must be set before calling load_and_process_data")
        # Use HF_TOKEN if available when loading the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)

        # Tokenize function
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        # Tokenize the training and validation datasets
        self.train_dataset = train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Remove unnecessary columns (original text column and any extra index columns)
        self.train_dataset = self.train_dataset.remove_columns(['text', '__index_level_0__']) # '__index_level_0__' is added by from_pandas
        self.val_dataset = self.val_dataset.remove_columns(['text', '__index_level_0__'])

        print("Data loading and processing completed.")
        print(f"Number of labels: {self.num_labels}")
        print("Label mapping:", self.label_to_id)

    def load_and_configure_model(self): # Removed model_name argument
        """
        Loads a pre-trained model and configures it for sequence classification with LoRA.

        Args:
            model_name: The name of the pre-trained model to load (e.g., "bert-base-uncased").
        """
        # Use self.model_name
        # Use HF_TOKEN if available when loading the model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels, token=self.hf_token)

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,  # LoRA attention dimension
            lora_alpha=16,  # Alpha parameter for LoRA scaling
            target_modules=["query", "value"],  # Modules to apply LoRA to
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            bias="none",  # Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
            task_type="SEQ_CLS",  # Task type, e.g. "SEQ_CLS" for sequence classification
        )

        # Apply the LoRA configuration
        self.model = get_peft_model(self.model, lora_config)

        # Print the trainable parameters
        self.model.print_trainable_parameters()

        print(f"Model loading and LoRA setup completed for '{self.model_name}'.")

    def initialize_trainer(self, output_dir: str = "./results", save_path: str = "ralf_state.pkl"):
        """
        Intializes the Hugging Face Trainer object for training.

        Args:
            output_dir: The output directory for model checkpoints and logs.
            save_path: The path to save the Ralf state using the custom callback.
        """
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,  # Output directory for model checkpoints and logs
            num_train_epochs=3,  # Number of training epochs
            per_device_train_batch_size=16,  # Batch size for training
            per_device_eval_batch_size=16,  # Batch size for evaluation
            warmup_steps=500,  # Number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # Strength of weight decay
            logging_dir="./logs",  # Directory for storing logs
            logging_steps=10, # Log every 10 steps
            eval_strategy="epoch", # Evaluate at the end of each epoch
            save_strategy="epoch", # Save checkpoint at the end of each epoch
            load_best_model_at_end=True, # Load the best model at the end of training
            metric_for_best_model="eval_loss", # Metric to use for loading the best model
            greater_is_better=False, # For eval_loss, lower is better
            report_to="none", # Disable reporting to services like W&B for simplicity
            hub_token=self.hf_token, # Pass HF token to trainer for potentially pushing to hub
            hub_model_id=f"my-awesome-model-{os.path.basename(output_dir)}" # Example hub_model_id
        )

        # Initialize the custom callback
        ralf_saving_callback = RalfSavingCallback(self, save_path=save_path)

        # Initialize the Trainer
        self.trainer = Trainer(
            model=self.model,  # Our LoRA-configured model
            args=training_args,  # Training arguments
            train_dataset=self.train_dataset,  # Training dataset
            eval_dataset=self.val_dataset,  # Validation dataset
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer), # Use DataCollatorWithPadding
            # compute_metrics=compute_metrics # Optional: define a function to compute metrics
            callbacks=[ralf_saving_callback] # Add the custom callback here
        )

        print("Trainer initialization completed with RalfSavingCallback.")

    def save_state(self, file_path: str = "ralf_state.pkl"):
        """
        Saves the current state of the Ralf instance using pickling.

        Args:
            file_path: The path to the file where the state will be saved.
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"Ralf state successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving Ralf state: {e}")

    @staticmethod
    def load_state(file_path: str = "ralf_state.pkl"):
        """
        Loads a previously saved Ralf instance from a pickle file.

        Args:
            file_path: The path to the pickle file.

        Returns:
            The loaded Ralf instance, or None if loading fails.
        """
        try:
            with open(file_path, 'rb') as f:
                ralf_instance = pickle.load(f)
            print(f"Ralf state successfully loaded from {file_path}")
            return ralf_instance
        except FileNotFoundError:
            print(f"Error loading Ralf state: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading Ralf state: {e}")
            return None
