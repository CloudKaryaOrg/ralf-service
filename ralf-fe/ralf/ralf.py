import os
import pandas as pd
import pickle
import warnings
import psutil  # Add this import
import humanize
from nltk.corpus import wordnet # Ensure you have the OpenAI Python client installed
import json
import re
import random
from ralf_train import RalfTraining, get_system_info, RalfSavingCallback

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Define the Ralf class
class Ralf(RalfTraining):
    """
    A class to encapsulate the datasets, model, and trainer for the Ralf project.
    """
    def __init__(self, HF_TOKEN, OPENAI_API_KEY=None, GEMINI_API_KEY=None): # Made HF_TOKEN required and others optional for recommendation
        """
        Initializes the Ralf class with placeholders for datasets, model name, and trainer.
        Requires HF_TOKEN and at least one of OPENAI_API_KEY or GEMINI_API_KEY for recommendations.
        """

        # Validate required keys
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is required.")
        if not OPENAI_API_KEY and not GEMINI_API_KEY:
            raise ValueError("Either OPENAI_API_KEY or GEMINI_API_KEY must be provided for recommendations.")

        # Hardware checks
        system_info = get_system_info()
        self.gpu_available = system_info["GPU Available"] == "✅ Yes"
        self.gpu_name = system_info["GPU Model"]
        self.gpu_ram_gb = system_info["GPU Memory"]
        self.ram_gb = system_info["System RAM"]
        self.gpu_count = system_info["GPU Count"]

        print(f"GPU available: {self.gpu_available}")
        if self.gpu_available:
            print(f"GPU count: {self.gpu_count}")
            print(f"GPU name: {self.gpu_name}")
            print(f"GPU RAM: {self.gpu_ram_gb} GB")
        print(f"Available system RAM: {self.ram_gb} GB")

        # API keys
        self.open_api_key = OPENAI_API_KEY
        self.gemini_key = GEMINI_API_KEY
        self.hf_token = HF_TOKEN # Stored the HF_TOKEN

    def set_keys(self, open_api_key=None, gemini_key=None, hf_token=None):
        """
        Set API keys for OpenAI, Gemini, and Hugging Face.
        Validates that at least one of open_api_key or gemini_key is provided if setting either.
        """
        if hf_token is not None:
            self.hf_token = hf_token

        if open_api_key is not None:
            self.open_api_key = open_api_key
        elif gemini_key is not None:
            self.gemini_key = gemini_key
        else:
            raise ValueError("Either open_api_key or gemini_key must be provided.")

    def recommend(self, input_csv_file,
                source_col,
                target_col,
                ):
        """Recommends top 3 LLMs for fine-tuning and a golden dataset based on problem type and resources using GPT-4o-mini or Gemini."""
        try:
            df = pd.read_csv(input_csv_file)
            if source_col not in df.columns:
                return (pd.DataFrame(), pd.DataFrame(), 
                   f"Error: Source column '{source_col}' not found in CSV file. Available columns: {', '.join(df.columns)}")
            if target_col not in df.columns:
                return (pd.DataFrame(), pd.DataFrame(), 
                   f"Error: Target column '{target_col}' not found in CSV file. Available columns: {', '.join(df.columns)}")
        except Exception as e:
            return (pd.DataFrame(), pd.DataFrame(), f"Error reading input CSV file: {e}")

        gpu_available = self.gpu_available
        gpu_ram_gb = self.gpu_ram_gb
        ram_gb = self.ram_gb
        client_info = self.get_llm_client()
        if not client_info:
            llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters","Description"])
            dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
            return llm_df, dataset_df, "Error: Neither OpenAI nor Gemini API key provided (should have been caught in initialization)."

        # Get the problem type analysis
        try:
            analysis = self.analyze_problem_type(pd.read_csv(input_csv_file), source_col, target_col)
            if not isinstance(analysis, dict):
                 llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters","Description"])
                 dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
                 return llm_df, dataset_df, analysis # Return error from analyze_problem_type
            problem_types = analysis.get('types', [])
            reasoning = analysis.get('reasoning', 'No reasoning provided.')
        except Exception as e:
             llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters","Description"])
             dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
             return llm_df, dataset_df, f"Error during problem type analysis: {e}"

        # Prompt for LLM recommendations and Hugging Face links
        llm_recommendation_prompt = (
            f"Based on the following problem types ({', '.join(problem_types)}) "
            f"and the available resources (GPU: {gpu_available}, GPU RAM: {gpu_ram_gb} GB, System RAM: {ram_gb} GB), "
            "recommend the top 5 open-source LLM models that would be suitable for fine-tuning for this task. "
            "For each, return:\n"
            "- name\n"
            "- model_id (Hugging Face identifier)\n"
            "- huggingface_url\n"
            "- size (e.g., 1.3B, 7B, 13B)\n"
            "- description (1-liner summary of the model's specialty)\n\n"
            "Return a JSON object with a single key 'llm_info' which is a list of dictionaries, where each dictionary contains 'name', 'model_id', 'huggingface_url', 'size' and 'description'. "
            "Do not include any other text, just the JSON object.\n\n"
            f"Problem types reasoning: {reasoning}"
        )

        # Prompt for Golden Dataset recommendation
        dataset_recommendation_prompt = (
            f"Based on the problem types ({', '.join(problem_types)}) and the sample data provided earlier (related to drug reviews and conditions), "
            "recommend a suitable, publicly available **golden dataset** for fine-tuning LLMs for this task. "
            "Only consider datasets available on **Hugging Face Datasets** or **Kaggle**, and ensure that the dataset:\n"
            "- is publicly accessible,\n"
            "- the page/link exists (i.e., does not return a 404 or error),\n"
            "- and contains actual data samples (i.e., not an empty or placeholder dataset).\n\n"

            "Once verified, return a JSON object with a single key 'golden_dataset_info', which is a dictionary containing:\n"
            "- 'name': the official name of the dataset,\n"
            "- 'url': a working link to the dataset (either Hugging Face or Kaggle),\n"
            "- 'source_column': name of the column used as input text,\n"
            "- 'target_column': name of the column used as label/class/output.\n\n"

            "Do not include any explanations or extra text — just return the JSON object.\n\n"
            f"Problem types reasoning: {reasoning}"
        )

        try:
            llm_content = self.get_llm_response(client_info, llm_recommendation_prompt)
            dataset_content = self.get_llm_response(client_info, dataset_recommendation_prompt)
            llm_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
            llm_recommendations = json.loads(llm_match.group(0)) if llm_match else json.loads(llm_content)
            dataset_match = re.search(r'\{.*\}', dataset_content, re.DOTALL)
            golden_dataset_info = json.loads(dataset_match.group(0)) if dataset_match else json.loads(dataset_content)

            llm_data = [
                {
                    "Name": info.get("name", "N/A"),
                    "Hugging Face URL": info.get("huggingface_url", "N/A"),
                    "Model ID": info.get("model_id", "N/A"),
                    "Parameters": self.estimate_param_count(info.get("model_id", "N/A")),
                    "Description": info.get("description", "N/A")
                } for info in llm_recommendations.get('llm_info', [])  # Use the estimate_param_count method
            ]
            dataset_data = [
                {
                    "Name": golden_dataset_info['golden_dataset_info'].get("name", "N/A"),
                    "URL": golden_dataset_info['golden_dataset_info'].get("url", "N/A"),
                    "Source Column": golden_dataset_info['golden_dataset_info'].get("source_column", "N/A"),
                    "Target Column": golden_dataset_info['golden_dataset_info'].get("target_column", "N/A")
                }
            ]

            return pd.DataFrame(llm_data), pd.DataFrame(dataset_data), analysis  # Return the DataFrames and analysis

        except Exception as e:
            # Handle cases where the API call fails or returns unexpected results
            # Return empty DataFrames and the error message
            llm_df = pd.DataFrame(columns=["Name", "Hugging Face URL", "Model ID", "Parameters","Description"])
            dataset_df = pd.DataFrame(columns=["Name", "URL", "Source Column", "Target Column"])
            analysis = f"Error calling API: {e}"

    def analyze_problem_type(self, df, source_col, target_col):
        """Analyze the problem type using either OpenAI or Gemini."""
        # Take a sample of 5 rows for context
        sample_df = df[[source_col, target_col]].dropna().sample(n=min(200, len(df)), random_state=42)
        sample_text = sample_df.to_csv(index=False)

        prompt = (
            "Given the following pairs of source and target data columns from a dataset, "
            "determine which of the following problem types best describe the task (one or more):\n"
            "- Classification\n"
            "- Summarization\n"
            "- Translation\n"
            "- Code Generation\n"
            "- Reasoning\n"
            "- Instruction Following\n"
            "- Safety & Refusal\n"
            "Only choose from this list. Return a JSON object with two keys: 'types' (a list of the chosen types) and 'reasoning' (a string explaining your reasoning for each category of the selection type, for example, why is it classification or why it is reason). "
            "Do not include any other text, just the JSON object.\n\n"
            f"Source column: {source_col}\n"
            f"Target column: {target_col}\n"
            f"Sample data:\n{sample_text}"
        )
        # Use the appropriate client and model based on available keys
        client_info = self.get_llm_client()
        if not client_info:
            return "Error: No LLM client available. Please set either OpenAI or Gemini API keys."
        try:
            content = self.get_llm_response(client_info, prompt)
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(content)  # If no match, return the content directly
        except Exception as e:
            return f"Error analyzing problem type: {str(e)}"

    def lustrate_data(self):
        """Clean the dataset: remove empty, duplicate or irrelevant rows"""
        before = len(self.df)
        self.df.dropna(subset=["source", "target"], inplace=True)
        self.df.drop_duplicates(subset=["source", "target"], inplace=True)
        after = len(self.df)
        print(f"Lustration complete: Removed {before - after} rows")
    
    def augment_data(self, max_aug_per_row=1):
        """Simple text augmentation using synonym replacement"""
        def synonym_augment(text):
            words = text.split()
            new_words = words.copy()
            random.shuffle(new_words)
            for i, word in enumerate(new_words):
                syns = wordnet.synsets(word)
                if syns:
                    synonyms = [lemma.name() for syn in syns for lemma in syn.lemmas() if lemma.name() != word]
                    if synonyms:
                        new_words[i] = random.choice(synonyms)
                        break
            return " ".join(new_words)

        aug_rows = []
        for _, row in self.df.iterrows():
            for _ in range(max_aug_per_row):
                aug_source = synonym_augment(row["source"])
                aug_target = row["target"]  # unchanged, or could be paraphrased
                aug_rows.append({"source": aug_source, "target": aug_target})

        aug_df = pd.DataFrame(aug_rows)
        self.df = pd.concat([self.df, aug_df], ignore_index=True)
        print(f"Augmented data: Added {len(aug_rows)} new rows")