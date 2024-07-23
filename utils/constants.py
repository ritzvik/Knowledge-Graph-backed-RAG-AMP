import os

seed_arxiv_paper_ids = [
    # https://github.com/aimerou/awesome-ai-papers?tab=readme-ov-file
    "2302.04761",
    "1706.03762",
    "2312.05934",
    "2302.13971",
    "2305.14314",
    "2303.17564",
    "2310.06825",
    "2304.02643",
    "2301.11305",
    "2304.03277",
    "2304.03442",
    "2310.12931",
    "2309.00267",
    "2212.09720",
    "2404.14047",
    # https://www.zeta-alpha.com/post/trends-in-ai-may-2024
    "2404.11018",
    "2404.07143",
    "2404.10102",
    "2404.10301",
    "2404.18796",
    "2404.05961",
    "2404.18424",
    "2404.02489",
    "2404.19737",
]

EMBED_PATH = "./embed_models"
MODELS_PATH = "./models"

huggingface_token = os.getenv("HF_TOKEN")

embed_model_name="thenlper/gte-large"
colbert_model = "colbert-ir/colbertv2.0"
local_model_to_be_quantised = "meta-llama/Meta-Llama-3-8B-Instruct"
n_gpu_layers = 1
n_ctx = 2048

llama3_stop_token = "<|eot_id|>"
llama3_bos_token = "<|begin_of_text|>" # Beggining of sequence token

cai_model_url = "https://ml-2fed18e9-f4c.eng-ml-l.vnu8-sqze.cloudera.site/namespaces/serving-default/endpoints/nousresearch-llama3/v1"
cai_model_id = "htwa-7bsc-cn3k-v87h"
