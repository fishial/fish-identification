import os
import json
import requests
import argparse
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def find_image_by_id(image_id: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Find and return the image dictionary with the specified ID from the COCO dataset.

    Args:
        image_id (int): The ID of the image to find.
        data (Dict[str, Any]): The COCO dataset as a dictionary.

    Returns:
        Optional[Dict[str, Any]]: The image dictionary if found, otherwise None.
    """
    for image in data.get('images', []):
        if image.get('id') == image_id:
            return image
    return None

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read a JSON file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data (Dict[str, Any]): The data to be saved.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def download_image(url_info: Tuple[str, str, int]) -> None:
    """
    Download an image from the given URL and save it to the specified file path.

    Args:
        url_info (Tuple[str, str, int]): A tuple containing:
            - The image URL (str)
            - The target file path (str)
            - The image index (int)
    """
    url, file_path, index = url_info
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded image index: {index}", end='\r')
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="COCO Image Downloader")
    parser.add_argument('-c', '--coco_path', type=str, required=True,
                        help="Path to the COCO JSON file")
    parser.add_argument('-i', '--image_path', type=str, required=True,
                        help="Directory where images will be stored")
    return parser.parse_args()

def main():
    """
    Main function to download images listed in a COCO JSON file if not already downloaded.
    """
    args = parse_arguments()
    coco_path = args.coco_path
    image_folder = args.image_path

    print(f"Start reading COCO file: {coco_path}")
    data = read_json_file(coco_path)
    print("Finished reading COCO file")
    
    # Define folder where images will be stored
    images_data_folder = os.path.join(image_folder, "data")
    os.makedirs(images_data_folder, exist_ok=True)
    
    # Get list of already downloaded images
    downloaded_files = [
        f for f in os.listdir(images_data_folder)
        if os.path.isfile(os.path.join(images_data_folder, f))
    ]
    
    # Calculate and print the count of files that differ from the expected images in the dataset
    all_image_filenames = {image['file_name'] for image in data.get('images', [])}
    different_files = set(downloaded_files) - all_image_filenames
    print(f"Count of different files: {len(different_files)}. Total already downloaded: {len(downloaded_files)}")
    
    # Prepare list of images to download
    urls_to_download: List[Tuple[str, str, int]] = []
    for i, image in enumerate(tqdm(data.get('images', []), desc="Processing images")):
        file_name = image.get('file_name')
        if file_name in downloaded_files:
            continue  # Skip images already downloaded
        target_file_path = os.path.join(images_data_folder, file_name)
        urls_to_download.append((image.get('coco_url'), target_file_path, i))
    
    print(f"Total images to download: {len(urls_to_download)}")
    
    # Save the download list to a JSON file for record
    save_json_file({'list_of_imgs': urls_to_download}, 'data.json')
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        list(tqdm(executor.map(download_image, urls_to_download),
                  total=len(urls_to_download),
                  desc="Downloading images"))
    
if __name__ == '__main__':
    main()