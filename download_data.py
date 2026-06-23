import os
import zipfile

def download_and_extract():
    # Target dataset slug on Kaggle
    dataset_slug = "vipoooool/new-plant-diseases-dataset"
    download_path = "./data"
    
    # Create data directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    print("Downloading dataset from Kaggle via API...")
    # Trigger Kaggle download
    os.system(f"kaggle datasets download -d {dataset_slug} -p {download_path}")
    
    zip_path = os.path.join(download_path, "new-plant-diseases-dataset.zip")
    
    if os.path.exists(zip_path):
        print("Extracting dataset zip archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Data extraction complete! Cleaning up zip file...")
        os.remove(zip_path)
    else:
        print("Error: Download failed. Double check your Kaggle API setup.")

if __name__ == "__main__":
    download_and_extract()
    