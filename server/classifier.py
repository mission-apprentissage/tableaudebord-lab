from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from datasets import Dataset, load_dataset
import pickle as pickle
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from huggingface_hub import hf_hub_download, ModelCard, ModelCardData, EvalResult
from huggingface_hub import HfApi
from tqdm import tqdm
import logging

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

tqdm.pandas()
logger = logging.getLogger(__name__)

class Classifier:
    """
    A classifier class that uses a pre-trained language model for text encoding
    and a trained classifier model for prediction.

    Attributes:
        version (str): The version of the model.
        model_file (str): The filename of the model.
        repo_id (str): The repository ID on HuggingFace Hub.
        token (str): The HuggingFace token.
        classifier: The trained classifier model.
        dataset: The dataset used for training.
    """
    def __init__(self, version="2025-10-20", 
                 lang_model="almanach/camembertav2-base",
                 token=""):
        """
        Initializes the Trainer with a pre-trained language model.

        Args:
            version (str): The version of the model.
            lang_model (str): The huggingface path of the pre-trained language model.
        """
        self.version = version
        self.model_file = f"knn-clf-formation-domain-{version}.pkl"
        self.repo_id = f"tableaudebord-apprentissage/{version}"
        self.token = token
        self.classifier = None
        self.dataset = None
        self.nlp = spacy.load("fr_core_news_lg")
        self.model = SentenceTransformer(lang_model)

    # Embedder function
    def encoding(self, text):
        """
        Encodes the input text into a normalized embedding using the language model.

        Args:
            text (str or list): The input text(s) to be encoded.

        Returns:
            list: A list containing the normalized embedding(s) of the input text(s).
        """
        # Handle both single text and batch of texts
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        def preprocess(text):
            doc = self.nlp(text.lower())
            return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop and not token.is_punct and not token.like_num])

        cleaned_texts = [preprocess(text) for text in texts]

        # Embeddings texts
        embeddings = self.model.encode(cleaned_texts)

        # Normalize embeddings
        embeddings = normalize(embeddings)

        return embeddings.tolist()

    # Classifier export function
    def save_model(self):
        """
        Save a classifier model to the HuggingFace Hub.

        Returns:
            url: The URL of the saved model.
        """
        logger.info(f"Save model locally...")
        local_repo = mkdtemp(prefix="tba-")
        with open(Path(local_repo) / self.model_file, mode="bw") as f:
            pickle.dump(self.classifier, file=f)

        api = HfApi()

        # Delete previous repo with the same name
        try:
            logger.info(f"Deleting existing repo: {self.repo_id}")
            api.delete_repo(repo_id=self.repo_id, token=self.token)
        except:
            pass

        # Create repo
        logger.info(f"Creating repo: {self.repo_id}")
        api.create_repo(repo_id=self.repo_id, token=self.token, repo_type="model", private=True)

        # Upload model
        logger.info(f"Uploading model: {local_repo}")
        out = api.upload_folder(
            folder_path=local_repo,
            repo_id=self.repo_id,
            token=self.token,
            repo_type="model",
            commit_message=f"pushing model '{self.version}' SVC with camembert v2 embeddings",
        )
        url = f"https://huggingface.co/{self.repo_id}"
        logger.info(f"Model ready on: {url}")
        return url

    # Classifier loader function
    def load_model(self):
        """
        Load a classifier model from the HuggingFace Hub.

        Returns:
            model: The loaded classifier model.
        """
        # Download model
        logger.info(f"Downloading model: {self.repo_id}")
        model_dump = hf_hub_download(repo_id=self.repo_id, filename=self.model_file, token=self.token)
        # print(f"- Model downloaded to: {model_dump}")

        # Reload pickle model
        with open(model_dump, 'rb') as f:
            self.classifier = pickle.load(f)
        logger.info(f"Classifier model ready.")

    # Classifier function
    def score(self, texts):
        """
        Predicts the multi-labels for the inputs texts using the classifier model.

        Args:
            texts (list(str)): List of texts to be classified.

        Returns:
            dict: A dictionary containing the input texts, predicted multi-labels (OHE).
        """
        embeddings = self.encoding(texts)
        labels = self.classifier.predict(embeddings).tolist()
        return {'model': self.version,
                'texts': texts, 
                'labels': labels}
