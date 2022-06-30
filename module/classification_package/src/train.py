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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def train(scheduler, t_total: int, opt: Optimizer, model: nn.Module, data_loader: DataLoader, ds_val: FishialDataset,
          device: torch.device, metrics: list, loss_fn: nn.Module, logger,
          output_folder="output", eval_every=None, file_name="best_score"):
    losses = AverageMeter()
    model = model.to(device)
    top_acc_1 = 0.2
    top_epoch, global_step = 0.0, 0.0
    step = 0
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    eval_every = eval_every if eval_every else t_total

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", 80)

    while True:

        model.train()
        epoch_iterator = tqdm(data_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              disable=False)

        all_preds, all_label = [], []
        for batch_idx, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            images, labels = batch

            output = model(images)
            loss = loss_fn(output, labels)

            preds = torch.argmax(output, dim=-1)

            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(labels.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], labels.detach().cpu().numpy(), axis=0
                )

            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                losses.update(loss.item() * gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(amp.master_params(opt), max_grad_norm)

                scheduler.step()
                opt.step()
                opt.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )

                if global_step % eval_every == 0:
                    logger.info("***** Running Validation *****")
                    #                     logger.info("  Num steps = %d", len(ds_val))

                    with torch.no_grad():
                        scores = evaluate(model=model, datasets=[data_loader.dataset], metrics=metrics,
                                          device=device)
                    logger.info("\n")
                    logger.info("Validation Results")
                    logger.info("Global Steps: %d" % global_step)
                    logger.info("Valid Accuracy: {}".format(scores))

                    val_acc = 1.0
                    for score in scores:
                        for a_idx, a in enumerate(scores[score]):
                            #                             writer.add_scalar(score + str(a_idx), a, global_step)
                            if val_acc > a: val_acc = a
                    if top_acc_1 <= val_acc:
                        top_acc_1 = val_acc
                        save_checkpoint(model, os.path.join(output_folder,
                                                            f'{file_name}.ckpt'))
                    if top_epoch < val_acc: top_epoch = val_acc

                    logger.info("best accuracy so far: %f" % top_epoch)

                    model.train()

                if global_step % t_total == 0:
                    break
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(device)
        train_accuracy = accuracy.detach().cpu().numpy()
        logger.info("train accuracy so far: %f" % train_accuracy)
        losses.reset()

        if global_step % t_total == 0:
            break