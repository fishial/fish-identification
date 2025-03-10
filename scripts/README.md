# Fishial Training Pipeline

This repository contains a multi-step training pipeline for processing fish images. The pipeline is broken into four main stages:

1. **Download Images** (`download_image.sh`)
2. **Segmentation** (`segmentation.sh`)
3. **Object Detection** (`object_detection.sh`)
4. **Classification** (`classification.sh`)

Each stage is implemented as a Bash script that calls underlying Python scripts. Follow the instructions below to set up and run the complete pipeline.

---

## Prerequisites

- **Bash Shell**: Ensure you have a Bash-compatible environment (Linux, macOS, or Windows with WSL).
- **Python 3.x**: Installed and available in your environment.
- **Python Dependencies**: Install required packages via:
  ```bash
  pip install -r requirements.txt

	•	Executable Permissions: Grant execute permissions to the shell scripts:

chmod +x download_image.sh segmentation.sh object_detection.sh classification.sh


	•	Directory Paths: Verify that all file paths and directories specified in the scripts (e.g., dataset paths, images directories, annotation files) exist and are correctly set. Modify the scripts if necessary.

⸻

Pipeline Stages

1. Download Images (download_image.sh)

This script downloads images to local storage using the COCO annotation file.
	•	Usage:

./download_image.sh


	•	Details:
	•	(Optional) Unzips and validates annotation files (the commands are commented out; uncomment if needed).
	•	Calls the Python script downloader_coco_imgs.py with:
	•	-c: Path to the COCO annotations.
	•	-i: Destination folder for the images.

⸻

2. Segmentation (segmentation.sh)

This script prepares data for segmentation and trains a segmentation model.
	•	Usage:

./segmentation.sh [options]


	•	Options:
	•	-h : Display help.
	•	-c <file>: COCO annotation file (default provided in the script).
	•	-i <dir>: Images directory.
	•	-d <name>: Segmentation dataset name.
	•	-s <dir>: Directory where segmentation model files are saved.
	•	Example:

./segmentation.sh -c /path/to/annotations.json -i /path/to/images -d my_seg_dataset -s /path/to/save/segmentation


	•	Script Flow:
	•	Convert COCO annotations to voxel format.
	•	Split the dataset using FiftyOne.
	•	Train the segmentation model.

⸻

3. Object Detection (object_detection.sh)

This script converts the dataset to YOLO format and trains an object detection model.
	•	Usage:

./object_detection.sh [options]


	•	Options:
	•	-h : Display help.
	•	-x <prefix>: Base prefix for file paths.
	•	-d <name>: Dataset name for fish detection.
	•	-o <dir>: Output directory for YOLO dataset.
	•	-n <num>: Number of classes (default: 1).
	•	-y <yaml>: Data YAML file for object detection.
	•	-p <proj>: Project directory.
	•	-r <name>: Run name for training.
	•	Example:

./object_detection.sh -x /path/to/prefix -d my_detection_dataset -o /path/to/yolo_dataset -n 1 -y /path/to/data.yaml -p /path/to/project -r training_run


	•	Script Flow:
	•	Convert FiftyOne dataset to YOLO format.
	•	Train the object detection model.

⸻

4. Classification (classification.sh)

This script creates a classification dataset, splits it, and trains models using both triplet loss and cross entropy loss.
	•	Usage:

./classification.sh [options]


	•	Options:
	•	-h : Display help.
	•	-p <dir>: Directory of classification images.
	•	-i <dir>: Classification input directory.
	•	-a <file>: Classification annotation file.
	•	-n <name>: Classification dataset name.
	•	Example:

./classification.sh -p /path/to/classification/images -i /path/to/classification/input -a /path/to/annotations.json -n my_classification_dataset


	•	Script Flow:
	•	Create the classification dataset.
	•	Split the dataset into training and validation sets.
	•	Train a model using triplet loss.
	•	Train a model using cross entropy loss.

⸻

Running the Pipeline

It is recommended to run the scripts in the following order:
	1.	Download Images:

./download_image.sh


	2.	Segmentation:

./segmentation.sh [your options]


	3.	Object Detection:

./object_detection.sh [your options]


	4.	Classification:

./classification.sh [your options]



Ensure that you have correctly configured all paths and options before running the pipeline. Check the console outputs for status messages and troubleshooting.

⸻

Troubleshooting & Tips
	•	Permissions: If you encounter permission errors, double-check the executable permissions using chmod +x.
	•	Python Errors: Verify that all dependencies are installed and that the Python environment is correctly configured.
	•	Path Verification: Confirm that all file and directory paths in the scripts are accurate for your system.
	•	Help Option: Use the -h flag with each script to display detailed usage instructions.

⸻

Conclusion

This pipeline automates the steps required to prepare data and train segmentation, object detection, and classification models. Adjust parameters as needed for your project, and refer to the respective Python script documentation for further details.

Happy Training!

---

You can modify the file as necessary to match your project’s structure or update any paths and parameters.