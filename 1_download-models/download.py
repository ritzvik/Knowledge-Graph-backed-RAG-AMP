from langchain_community.embeddings import HuggingFaceEmbeddings

import utils.constants as const
from utils.huggingface_utils import quantise_and_save_local_model

# This just caches the embedding model for future use
HuggingFaceEmbeddings(model_name=const.embed_model_name, cache_folder=const.EMBED_PATH)

# cache the Llama 3 8b local model in a 4-bit quantised format
quantise_and_save_local_model()
