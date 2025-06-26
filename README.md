# Fishial.ai - Fish Recognition Platform

[![Project Website](https://img.shields.io/badge/Website-Fishial.ai-blue.svg)](https://www.fishial.ai)
[![Demo App](https://img.shields.io/badge/Demo-Web%20Application-green.svg)](https://portal.fishial.ai/search/by-fishial-recognition)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

<a target="_blank" href="https://colab.research.google.com/drive/1nKJ0V1sBLgfNJaCTQmuqUV1ybrx1m7qI?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

This repository contains the official training and validation scripts for the **Fishial.ai** fish segmentation, detection, and classification models.

## 🚀 Getting Started

The easiest way to get started is by using our Google Colab Notebook. It allows you to run segmentation and classification models directly in your browser, using Google's cloud infrastructure or your own local machine.

* **[🚀 Open in Google Colab](https://colab.research.google.com/drive/1nKJ0V1sBLgfNJaCTQmuqUV1ybrx1m7qI?usp=sharing)**

## 🛠️ Local Installation

To run the scripts on your own machine, follow these steps.

**1. Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>
````

**2. Install dependencies:**
Make sure you have Python 3.x installed.

```bash
pip3 install -r requirements.txt
```

## ⚙️ How to Run Scripts

First, make the shell scripts executable:

```bash
chmod +x segmentation.sh classification.sh object_detection.sh
```

### Segmentation

```bash
./segmentation.sh -c <your_coco_file> -i <your_images_dir> -d <your_segmentation_ds_name> -s <your_save_dir>
```

  - `-c`: Path to your COCO annotations file.
  - `-i`: Path to the directory containing your images.
  - `-d`: Name for your segmentation dataset.
  - `-s`: Directory where the trained model will be saved.

### Classification

```bash
./classification.sh -p <your_classif_images_dir> -i <your_classif_input_dir> -a <your_annotation_file> -n <your_classification_ds_name>
```

  - `-p`: Path to the classification images directory.
  - `-i`: Path to the classification input directory.
  - `-a`: Path to your annotation file.
  - `-n`: Name for your classification dataset.

### Object Detection (YOLO)

```bash
./object_detection.sh -d <your_detection_ds> -o <your_yolo_output_dir> -n <num_classes> -y <data_yaml> -p <project_dir> -r <run_name>
```

  - `-d`: Your detection dataset.
  - `-o`: Output directory for YOLO results.
  - `-n`: Number of classes.
  - `-y`: Path to the `data.yaml` file.
  - `-p`: Project directory.
  - `-r`: Name for the specific run.

## 📂 Key Project Files

Here's a breakdown of the most important scripts and modules in this project.

### Training Scripts

  * [`train_scripts/classification/auto_train_cross.py`](https://www.google.com/search?q=train_scripts/classification/auto_train_cross.py): Automatically trains a classification model using the cross-entropy loss function. It saves the best-performing checkpoint based on validation accuracy.
  * [`train_scripts/classification/auto_train_triplet.py`](https://www.google.com/search?q=train_scripts/classification/auto_train_triplet.py): Trains a classification model using Triplet or Quadruplet loss. It saves the best checkpoint based on the k-metric on the validation set.
  * [`train_scripts/segmentation/train.py`](https://www.google.com/search?q=train_scripts/segmentation/train.py): A basic script to train a segmentation model using the Detectron2 API.
  * [`train_scripts/segmentation/train_copy_paste.py`](https://www.google.com/search?q=train_scripts/segmentation/train_copy_paste.py): Trains a segmentation model using Detectron2 with the "Copy-Paste" data augmentation technique.

### Helper Notebooks & Scripts

  * [`helper/ExportModelToTorchscript.ipynb`](https://www.google.com/search?q=helper/ExportModelToTorchscript.ipynb): A Jupyter Notebook to convert PyTorch classification models and Detectron2 segmentation models to the TorchScript format for optimized deployment.
  * [`helper/classification/CreateDataBaseTensor.py`](https://www.google.com/search?q=helper/classification/CreateDataBaseTensor.py): Generates an embedding tensor from a trained classification network. This tensor is used for efficient inference.
  * [`helper/classification/CreateDatasetAndTrain.ipynb`](https://www.google.com/search?q=helper/classification/CreateDatasetAndTrain.ipynb): A script to create training/testing datasets from a Fishial COCO export and subsequently train a network.

### Core Modules

  * [`module/classification_package/src/model.py`](https://www.google.com/search?q=module/classification_package/src/model.py) & [`.../utils.py`](https://www.google.com/search?q=module/classification_package/src/utils.py): These files contain the core implementation of the classification pipeline.

-----

## 📦 Pre-trained Models

We provide several pre-trained models for immediate use. The latest models are marked with ⭐.

| Model Description                                                                | Download Link                                                                                                 |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **⭐ Fish Detector BoundingBox** - YOLOv12 Medium (img size 640, torchscript)      | [link](https://storage.googleapis.com/fishial-ml-resources/detector_v10_m5.zip)                               |
| **⭐ Fish Classification** - beitv2_base_patch16_224 (640 classes, embed 512, torchscript)  | [link](https://storage.googleapis.com/fishial-ml-resources/classification_rectangle_v9-3.zip)                 |
| **⭐ Fish Classification** - ConvNeXt Tiny (640 classes, embed 256, torchscript)  | [link](https://storage.googleapis.com/fishial-ml-resources/classification_rectangle_v9-2.zip)                 |
| **⭐ Fish Segmentation** - FPN w/ ResNet18 (img size 416, torchscript)            | [link](https://storage.googleapis.com/fishial-ml-resources/segmentator_fpn_res18_416_1.zip)                     |
| Fish Detector BoundingBox - YOLOv10 Medium (img size 640, torchscript)      | [link](https://storage.googleapis.com/fishial-ml-resources/detector_v10_m3.zip)                               |
| Fish Classification - ConvNeXt Tiny (426 classes, embed 128, torchscript)  | [link](https://storage.googleapis.com/fishial-ml-resources/classification_rectangle_v7-1.zip)                 |
| Fish Segmentation - FPN w/ ResNet18 (img size 416, torchscript)            | [link](https://storage.googleapis.com/fishial-ml-resources/segmentator_fpn_res18_416_1.zip)                     |
| MaskRCNN Fish Segmentation (Updated 21.08.2023)                                  | [link](https://storage.googleapis.com/fishial-ml-resources/model_21_08_2023.pth)                                |
| MaskRCNN Fish Segmentation (Updated 21.08.2023, torchscript)                     | [link](https://storage.googleapis.com/fishial-ml-resources/segmentation_21_08_2023.ts)                          |
| ResNet18 v6 model pack (289 classes)                                             | [link](https://storage.googleapis.com/fishial-ml-resources/classification_fishial_30_06_2023.zip)             |
| ResNet18 v5 model pack (184 classes)                                             | [link](https://storage.googleapis.com/fishial-ml-resources/classification_22_12.zip)                            |
| MaskRCNN Fish Segmentation (Updated 15.11.2022)                                  | [link](https://storage.googleapis.com/fishial-ml-resources/model_15_11_2022.pth)                                |
| ResNet18 v4 model pack (184 classes)                                             | [link](https://storage.googleapis.com/fishial-ml-resources/classification_v5.zip)                             |
| ResNet18 DataBase Tensor                                                         | [link](https://storage.googleapis.com/fishial-ml-resources/models_29.06.2022/train%2Btest_embedding.pt)        |
| ResNet18 Fish Classification Embedding 256 V2.0                                  | [link](https://storage.googleapis.com/fishial-ml-resources/models_29.06.2022/full_256.ckpt)                    |
| ResNet18 Binary Classification                                                   | [link](https://storage.cloud.google.com/fishial-ml-resources/binary_class.ckpt)                               |
| ResNet18 Fish Classification Cross Entropy V1.0                                  | [link](https://storage.googleapis.com/fishial-ml-resources/final_cross_cross_entropy_0.9923599320882852_258571.0.ckpt) |
| MaskRCNN Fish Segmentation (Updated 29.06.2022)                                  | [link](https://storage.googleapis.com/fishial-ml-resources/models_29.06.2022/model_0259999.pth)                |

-----

## 📊 Training Results

For more detailed results, see the [Training README](https://www.google.com/search?q=train_scripts/README.md).

### Classification Model

A JSON file containing the names of all fish classes recognized by the latest model can be found here: **[labels.json](https://www.google.com/search?q=labels.json)**.

-----

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.

```