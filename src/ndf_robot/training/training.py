'''Implements a generic training loop.
'''

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from collections import defaultdict
import torch.distributed as dist

import ndf_robot.training.util as util


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def multiscale_training(model, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
                        dataloader_callback, dataloader_iters, dataloader_params,
                        val_loss_fn=None, summary_fn=None, iters_til_checkpoint=None, clip_grad=False,
                        overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0):

    for params, max_steps in zip(dataloader_params, dataloader_iters):
        train_dataloader, val_dataloader = dataloader_callback(*params)
        model_dir = os.path.join(model_dir, '_'.join(map(str, params)))

        model, optimizers = train(model, train_dataloader, epochs=10000, lr=lr, steps_til_summary=steps_til_summary,
                                  val_dataloader=val_dataloader, epochs_til_checkpoint=epochs_til_checkpoint, model_dir=model_dir, loss_fn=loss_fn,
                                  val_loss_fn=val_loss_fn, summary_fn=summary_fn, iters_til_checkpoint=iters_til_checkpoint,
                                  clip_grad=clip_grad, overwrite=overwrite, optimizers=optimizers, batches_per_validation=batches_per_validation,
                                  gpus=gpus, rank=rank, max_steps=max_steps)


def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                            writer.add_scalar('val_' + loss_name, single_loss, total_steps)

                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        return model, optimizers


def train_feature(model, train_dataloader, corr_model, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None):

    model.eval()
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=corr_model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)

        summaries_dir = os.path.join(model_dir, 'summaries')
        util.cond_mkdir(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt)

                start_time = time.time()

                with torch.no_grad():
                    model_output = model(model_input)

                model_output = corr_model(model_output['features'])

                losses = loss_fn(model_output, gt)
                # losses = loss_fn(model_output, gt, model_input, model)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, model_input, gt, model_output, writer, total_steps)

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)

                                with torch.no_grad():
                                    model_output = model(model_input)

                                model_output = corr_model(model_output['features'])
                                val_loss = val_loss_fn(model_output, gt, val=True)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss)
                                summary_fn(model, model_input, gt, model_output, writer, total_steps, 'val_')
                                writer.add_scalar('val_' + loss_name, single_loss, total_steps)


                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        return model, optimizers
