from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import HfApi

from os.path import basename
import pathlib

def upload_to_hf(path, path_in_repo, repo, clean=False):
    api = HfApi()
    if path_in_repo is None:
        path_in_repo = basename(path)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type="model")
    if clean:
        pathlib.Path(path).unlink()