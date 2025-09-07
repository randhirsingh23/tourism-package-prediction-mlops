
from huggingface_hub import upload_folder
import os

SPACE_ID = "RandhirSingh23/tourism-package-prediction-app"
DEPLOY_DIR = os.path.dirname(__file__)  # this folder (has Dockerfile, app.py, requirements.txt, README.md)

upload_folder(
    repo_id=SPACE_ID,
    repo_type="space",
    folder_path=DEPLOY_DIR,
    path_in_repo="."
)
print("Pushed deployment folder to:", f"https://huggingface.co/spaces/{SPACE_ID}")
