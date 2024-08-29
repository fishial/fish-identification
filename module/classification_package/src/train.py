import os
import numpy as np
import torch

from apex import amp
from tqdm import tqdm

from torch import nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from module.classification_package.src.eval import evaluate
from module.classification_package.src.utils import AverageMeter
from module.classification_package.src.utils import save_checkpoint
from module.classification_package.src.dataset import FishialDataset

from module.classification_package.src.eval import pairwise_distance, get_acc, evaluate_acc

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def remove_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} has been removed.")
    else:
        print(f"{file_path} does not exist.")
        

def train(scheduler, epoch: int, opt: Optimizer, model: nn.Module, data_loader: DataLoader, ds_val: FishialDataset,
          device: torch.device, metrics: list, loss_fn: nn.Module, logger,
          output_folder="output", eval_every=None, file_name="best_score", extra_val = None):
    losses = AverageMeter()
    model = model.to(device)
    top_acc_1 = 0.0
    top_epoch, global_step = 0.0, 0.0
    max_grad_norm = 1.0
    
    t_total = len(data_loader) * epoch
    eval_every = eval_every * len(data_loader) if eval_every else t_total
    at_k_train = [1, 3, 5]
    best_model_filepath = None

    # Train!
#     logger.info("***** Running training *****")
#     logger.info("  Total optimization steps = %d", t_total)
#     logger.info("  Instantaneous batch size per GPU = %d", 80)

    while True:

        model.train()
        if metrics[0] == 'accuracy':
            model.fc_parallel.requires_grad_(True)
            model.backbone.requires_grad_(False)
            model.embeddings.requires_grad_(False)

        epoch_iterator = tqdm(data_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              disable=False)

        
        all_preds, all_label = [], []
        mean_acc = {}
        for i in at_k_train:
            mean_acc[f"at_{i}"] = []
            mean_acc[f"mf1_{i}"] = []
        for batch in epoch_iterator:
            batch = tuple(t.to(device) for t in batch)
            images, labels = batch
            opt.zero_grad()

            if metrics[0] == 'at_k':
                output = model(images)[0]
            else:
                output = model(images)[1]
 
            loss = loss_fn(output, labels)
            
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward(retain_graph=True)

            if metrics[0] == 'accuracy':
                preds = torch.argmax(output, dim=-1)
                
                if len(all_preds) == 0:
                    all_preds.append(preds.detach().cpu().numpy())
                    all_label.append(labels.detach().cpu().numpy())
                else:
                    all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
                    all_label[0] = np.append(all_label[0], labels.detach().cpu().numpy(), axis=0)
                accuracy = simple_accuracy(all_preds[0], all_label[0])
                
            else:
                
                output.to('cpu')
                distances = pairwise_distance(output, output)
                accuracy = get_acc(distances, labels, labels, at_k = at_k_train, start_since = 1)[0]
                
                for asd in accuracy:
                    mean_acc[asd].append(accuracy[asd])

                accuracy = " ".join([cvf+": "+str(round(sum(mean_acc[cvf])/(len(mean_acc[cvf]) + 1), 3)) for cvf in mean_acc])
                
            losses.update(loss.item())
            torch.nn.utils.clip_grad_norm_(amp.master_params(opt), max_grad_norm)
            
            scheduler.step()
            opt.step()
            
            global_step += 1

            description = f"Training:  EPOCH: ({round(global_step/len(data_loader), 2)}/{epoch}) (loss={round(losses.val,5)}) accuracy: [{accuracy}] Validation: {round(top_epoch, 4)}"
            epoch_iterator.set_description(description)

            if global_step % eval_every == 0:
                model.eval()
#                 logger.info("***** Running Validation *****")
                # logger.info("  Num steps = %d", len(ds_val))
                
                if metrics[0] != 'accuracy':
                    with torch.no_grad():
                        scores = evaluate(model=model, data_base = data_loader.dataset, validation = ds_val, extra_val = extra_val, save_as = [output_folder, global_step])
                        
                    logger.info(f"{global_step} :: {scores}")
                    print(f"{global_step} :: |{scores}|")
                    
                    val_acc = scores['validation_on_database']['mf1_1']      
                else:
                    val_acc = evaluate_acc(model, ds_val)
                    
                if top_epoch < val_acc:
                    top_epoch = val_acc

                    if best_model_filepath is None:
                        best_model_filepath = os.path.join(output_folder, f"best_ckpt_{val_acc}.ckpt")
                    else:
                        remove_file_if_exists(best_model_filepath)
                        best_model_filepath = os.path.join(output_folder, f"best_ckpt_{val_acc}.ckpt")
                        logger.info(f"best_model_filepath EXIST CREATE NEW: {best_model_filepath}")

                save_checkpoint(model, best_model_filepath)
                
                model.train()
                if metrics[0] == 'accuracy':
                    model.fc_parallel.requires_grad_(True)
                    model.backbone.requires_grad_(False)
                    model.embeddings.requires_grad_(False)
        losses.reset()

        mean_acc = {}
        for i in at_k_train:
            mean_acc[f"at_{i}"] = []
            mean_acc[f"mf1_{i}"] = []
            
        if global_step % t_total == 0:
            break