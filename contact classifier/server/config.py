import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace configuration
HF_TOKEN = os.getenv('HF_TOKEN')
ORG_NAME = "tableaudebord-apprentissage"

# Model configuration
MODEL_VERSION = "2026-03-16"

# Server configuration
SERVER_PORT = int(os.getenv('LAB_SERVER_PORT', 8000))
PUBLIC_VERSION = os.getenv('PUBLIC_VERSION', MODEL_VERSION)