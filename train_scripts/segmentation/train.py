import sys
#Change path specificly to your directories
sys.path.insert(1, '/home/codahead/Fishial/FishialReaserch')

import os

from utils import run_eval_checkpoints
from utils import get_current_date_in_format
from utils import get_dataset_dicts
from utils import remove_tmp_files
from utils import save_json
from module.segmentation_package.src.trainer import Trainer

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
setup_logger()


def main():
    train_folder = "Train"
    for d in ["Train", "Test"]:
        DatasetCatalog.register("fishial_" + d, lambda d=d: get_dataset_dicts("fishial_collection", d, json_file='fishial_collection_correct.json'))
        MetadataCatalog.get("fishial_" + d).set(thing_classes=["fish"], evaluator_type="coco")

    config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    experiment_name = "amp_on"
    main_folder = "output"

    learning_rates = [0.00025]  # .01 .001 .0001 .00025 .000025
    roi_batch_sizes = [128]  # 32, 64, 128, 256
    freezing_layers = [2]  # 0 1 2 3 4 5
    amp = [True]
    crop_enables = [True]
    max_iters = 100000
    check_point_step = 2500
    experiment_num = 0

    # Pick and set up a GPU. This serve got 8 [0, ... , 7]
    # torch.cuda.set_device(7)

    # Tune hyperparams with val set.
    for freezing_layer in freezing_layers:
        for learning_rate in learning_rates:
            for crop_enable in crop_enables:
                for amp_state in amp:
                    for roi_batch_size in roi_batch_sizes:
                        torch.cuda.empty_cache()
                        experiment_folder = os.path.join(main_folder,
                                                         os.path.join(experiment_name, get_current_date_in_format()))
                        # Create experiment folder
                        os.makedirs(experiment_folder, exist_ok=True)
                        experiment_settings = {
                            "roi_batch_size": roi_batch_size,
                            "amp_state": amp_state,
                            "crop_enable": crop_enable,
                            "learning_rate": learning_rate,
                            "freezing_layer": freezing_layer
                        }

                        # Save config of experiment to experiment folder.
                        save_json(experiment_settings, os.path.join(experiment_folder, "ckpt_eval.json"))
                        # Get default config node.
                        cfg = get_cfg()

                        cfg.merge_from_file(model_zoo.get_config_file(config_path))

                        cfg.DATASETS.TRAIN = ("fishial_" + train_folder, )
                        # If it is set, after the trainer() will start automatically trainer.test()
                        cfg.DATASETS.TEST = ()

                        # Number of data loading threads, dft=4
                        cfg.DATALOADER.NUM_WORKERS = 2

                        # Path to a checkpoint file to be loaded to the model.
                        # You can find available models in the model zoo.
                        # Here I'm loading the weights obtained through the training on the COCO dataset
                        cfg.MODEL.WEIGHTS = "best_scores/model_0067499_amp_on.pth"

                        # Number of images per batch across all machines.
                        cfg.SOLVER.IMS_PER_BATCH = 2

                        # cfg.SEED = 3
                        cfg.SOLVER.MAX_ITER = max_iters

                        # Save checkpoits each specific step
                        cfg.SOLVER.CHECKPOINT_PERIOD = check_point_step

                        #automatic mix precision
                        cfg.SOLVER.AMP.ENABLED = amp_state

                        # `True` if cropping is used for data augmentation during training
                        cfg.INPUT.CROP.ENABLED = crop_enable
                        cfg.TEST.AUG.ENABLED = crop_enable

                        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (fish)

                        # Not filter out images with no annotations. keep them
                        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

                        # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0, 3.0, 4.0, 5.0]]

                        # cfg.MODEL.ANCHOR_GENERATOR.OFFSET = 0.5

                        # cfg.MODEL.PIXEL_MEAN = [44.2694227416269, 44.2694227416269, 44.2694227416269]
                        # cfg.MODEL.PIXEL_STD = [57.375, 57.375, 57.375]

                        cfg.OUTPUT_DIR = experiment_folder

                        cfg.MODEL.BACKBONE.FREEZE_AT = freezing_layer

                        cfg.SOLVER.BASE_LR = learning_rate

                        # RoI minibatch size *per image* (number of regions of interest [ROIs])
                        # Total number of RoIs per training minibatch =
                        # ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
                        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_size  # (default: 512)

                        # Build a trainer from the specified config
                        trainer = DefaultTrainer(cfg) #Trainer

                        # If resume==True, and last checkpoint exists, resume from it.
                        # Otherwise, load a model specified by the config.
                        trainer.resume_or_load()
                        # Start training
                        trainer.train()
                        
                        torch.cuda.empty_cache()
                        #Remove tmp files
                        remove_tmp_files(experiment_folder)
                        run_eval_checkpoints(cfg, experiment_folder, "fishial_Test")
                        experiment_num += 1
                        print("Currently was finished: |{}| ".format(experiment_num))

    print(70 * "*")
    print("Finish")
    print(70 * "*")


if __name__ == "__main__":
    main()
