import argparse
from huggingface_hub import HfApi

def upload_to_hub(local_data_folder, repo_id):
    """
    Uploads the contents of a local folder to a Hugging Face dataset repository.

    Args:
        local_data_folder (str): The path to the local folder to upload.
        repo_id (str): The ID of the Hugging Face repository (e.g., "username/repo-name").
    """
    # Ensure you have logged in via the command line first:
    # huggingface-cli login
    api = HfApi()

    print(f"Uploading folder '{local_data_folder}' to dataset repository '{repo_id}'...")

    # Upload all contents of the folder to Hugging Face.
    # It's multi-threaded and handles large files efficiently.
    api.upload_folder(
        folder_path=local_data_folder,
        repo_id=repo_id,
        repo_type="dataset"  # Specify that this is a dataset repository
    )

    print("Upload complete!")

if __name__ == "__main__":
    # Set up the argument parser to read command-line arguments
    parser = argparse.ArgumentParser(description="Upload a local dataset folder to the Hugging Face Hub.")
    parser.add_argument("local_dir", type=str, help="The path to the local data folder containing the shards.")
    parser.add_argument("repo_id", type=str, help="The destination repository ID on Hugging Face (e.g., 'username/repo-name').")

    # Parse the arguments provided by the user
    args = parser.parse_args()

    # Call the main function with the provided arguments
    upload_to_hub(args.local_dir, args.repo_id)