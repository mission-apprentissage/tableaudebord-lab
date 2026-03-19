from datasets import Dataset, load_dataset
import joblib
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from huggingface_hub import hf_hub_download, ModelCard, ModelCardData, EvalResult
from huggingface_hub import HfApi
from tqdm import tqdm
import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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
    def __init__(self, version="2026-03-16", 
                 token=""):
        """
        Initializes the Trainer with a pre-trained language model.

        Args:
            version (str): The version of the model.
            lang_model (str): The huggingface path of the pre-trained language model.
        """
        self.version = version
        self.model_file = f"tba-rf-ml-{version}.joblib"
        self.repo_id = f"tableaudebord-apprentissage/{version}"
        self.token = token
        self.classifier = None
        self.dataset = None

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
            commit_message=f"pushing model '{self.version}' RF for contact prediction",
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

        # Reload joblib model
        self.classifier = joblib.load(model_dump)
        logger.info(f"Classifier model ready.")

    def extract_features(self, data):
        date_cols = ['apprenant.date_de_naissance',
             'formation.date_inscription',
             'formation.date_fin',
             'formation.date_entree',
             'contrat.date_debut',
             'contrat.date_fin',
             'contrat.date_rupture',
        ]
        features = pd.DataFrame(data)[date_cols]

        # Compute delay
        today = pd.to_datetime('today', utc=True)
        features = features.map(lambda x: today - pd.to_datetime(str(x), utc=True, errors='coerce'))
        features = features.map(lambda x: x.days if isinstance(x, pd.Timedelta) else 0)

        return features

    # Classifier function
    def score(self, data):
        """
        Compute the probability score for the inputs data using the classifier model.

        Args:
            data (list(str)): List of dates to be classified.

        Returns:
            dict: A dictionary containing the probability scores.
        """
        features = self.extract_features(data)
        y_probs = self.classifier.predict_proba(features)[:, 1]

        return {'model': self.version,
                'scores': y_probs.tolist()}