#!/usr/bin/env bash
set -euo pipefail
#RUN: chmod +x ./settingUp.sh && sudo ./settingUp.sh

ZIP_FILE_PATH="V0.4_export.zip"
DST_COCOFILES_PATH="export_14_06_2023"
DST_IMAGES_FOLDERNAME="DATA"
CLASSIFICATION_FOLDER_NAME="CLASSIFICATION"
VOXEL_DATASET_CLASSIFICATION_NAME="voxel-classification-14-06-2023"
VOXEL_DATASET_SEGMENTATION_NAME='export-fishial-14-06-2023'
EXPORT_FILE_NAME=".04_export_Verified_ALL.json"

mkdir -p $DST_COCOFILES_PATH

#echo "[INFO] Unzip annotations files from $ZIP_FILE_PATH to folder $DST_COCOFILES_PATH"
#unzip $ZIP_FILE_PATH -d $DST_COCOFILES_PATH

echo "[INFO] Making validation of export files"
#python /home/fishial/Fishial/Object-Detection-Model/helper/cleaner/verifier_coco.py -a "$DST_COCOFILES_PATH/$EXPORT_FILE_NAME" -p .

echo "[INFO] Download images to local storage"
#python /home/fishial/Fishial/Object-Detection-Model/helper/segmentation/downloader_coco_imgs.py -c "$DST_COCOFILES_PATH/$EXPORT_FILE_NAME" -i "$DST_COCOFILES_PATH/$DST_IMAGES_FOLDERNAME"

echo "[INFO] Convert dataset from COCO to Voxel51"
#python /home/fishial/Fishial/Object-Detection-Model/helper/converterCocoToVoxel.py -c "$DST_COCOFILES_PATH/$EXPORT_FILE_NAME" -i "$DST_COCOFILES_PATH/$DST_IMAGES_FOLDERNAME/data" -ds 'export-fishial-14-06-2023'

echo "[INFO] Creating splited classification dataset"
#python /home/fishial/Fishial/Object-Detection-Model/helper/classification/classification_dataset_creator.py -dp "$DST_COCOFILES_PATH/$CLASSIFICATION_FOLDER_NAME" -i "$DST_COCOFILES_PATH/$DST_IMAGES_FOLDERNAME/data" -a "$DST_COCOFILES_PATH/$EXPORT_FILE_NAME" -dsn $VOXEL_DATASET_CLASSIFICATION_NAME  -mei 10 -mpei 0.2 -macipc 350 -micipc 50

echo "[INFO] Copying tags from the previous dataset of validation images"
python /home/fishial/Fishial/Object-Detection-Model/helper/copying_tags_fiftyone.py -src "fishial-dataset-november-2022" -dst "export-fishial-14-06-2023" -t "val"