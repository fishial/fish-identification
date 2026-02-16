#!/usr/bin/env bash
set -euo pipefail
#RUN: chmod +x ./download_image.sh

ZIP_FILE_PATH="V0.4_export.zip"
DST_COCOFILES_PATH="testing.json"
DST_IMAGES_FOLDERNAME="images_folder"

# mkdir -p $DST_COCOFILES_PATH

#echo "[INFO] Unzip annotations files from $ZIP_FILE_PATH to folder $DST_COCOFILES_PATH"
#unzip $ZIP_FILE_PATH -d $DST_COCOFILES_PATH

# echo "[INFO] Making validation of export files"
#python /home/fishial/Fishial/Object-Detection-Model/helper/cleaner/verifier_coco.py -a "$DST_COCOFILES_PATH/$EXPORT_FILE_NAME" -p .

echo "[INFO] Download images to local storage"
python helper/segmentation/downloader_coco_imgs.py -c $DST_COCOFILES_PATH -i $DST_IMAGES_FOLDERNAME