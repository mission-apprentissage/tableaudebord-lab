from transformers import AutoTokenizer, AutoModel
import torch
import pickle
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
    def __init__(self, version="2025-08-06", 
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
        self.repo_id = f"tableau-de-bord-apprentissage/{version}"
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

        cleaned_texts = [preprocess(t) for t in texts]

        # Embeddings texts
        embeddings = self.model.encode(cleaned_texts)

        # Normalize embeddings
        embeddings = normalize(embeddings)

        return embeddings

    # Classifier function
    def score(self, text):
        """
        Predicts the label and scores for the input text using the classifier model.

        Args:
            text (str): The input text to be classified.

        Returns:
            dict: A dictionary containing the input text, predicted label, and scores for each class.
        """
        embeddings = self.encoding([text])
        x = pd.DataFrame([embeddings[0]])
        y_label = self.classifier.predict(x)[0]
        y_prob = self.classifier.predict_proba(x)[0].tolist()
        y_prob = [round(i, 4) for i in y_prob]
        return {'model': self.version,
                'text': text, 'label': y_label, 
                'scores': {'cfa': y_prob[0], 'entreprise': y_prob[1], 'entreprise_cfa': y_prob[2]}}
