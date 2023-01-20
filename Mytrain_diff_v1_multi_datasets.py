import argparse
import random
import shutil
import sys
import os
import torch
from torch.utils.data import DataLoader
from models.diff_v1 import NP
from dataset.mri_dataset_aparc_aseg import Generate_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
import glob
from pytorch_lightning.strategies import DDPStrategy

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=False, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=2e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    train_dataset,val_dataset = Generate_dataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=5,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=( "cuda"),persistent_workers=True,drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=5,
        num_workers=4,
        shuffle=False,
        pin_memory=( "cuda"),persistent_workers=True,drop_last=True
    )

    root_dir_path = os.path.join('checkpoint', "PVM_diff_v1")
    single_gpu = 1 if torch.cuda.device_count() ==1 else 2
    if torch.cuda.device_count() == 1 or single_gpu==True:
        trainer = pl.Trainer(
            default_root_dir=root_dir_path,
            devices=1,
            max_epochs=args.epochs,
            precision=32, accelerator="gpu",
            callbacks=[
                ModelCheckpoint(mode="max",every_n_epochs = 1),
                LearningRateMonitor("epoch"),RichProgressBar(),
            ],
        )
    else:
        trainer = pl.Trainer(
            strategy=DDPStrategy(find_unused_parameters=False),
            default_root_dir=root_dir_path,
            devices=torch.cuda.device_count(),precision=16,
            max_epochs=args.epochs, accelerator="gpu",
            callbacks=[
                ModelCheckpoint(mode="max",every_n_epochs = 2),
                LearningRateMonitor("epoch"),RichProgressBar(),
            ]
        )
    #trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    #trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    files = glob.glob('/home/xh278/PVM/checkpoint/PVM_diff_v1/lightning_logs/*/checkpoints/*')
    # V6 achieve 0.62 dice which is very good and the corresponding model is in v9
    #PVM_res_v5_fix1_high_lr the first one we successed
    #v9 is the best 0.82
    sorted_by_mtime_descending = sorted(files, key=lambda t: -os.stat(t).st_mtime)
    pretrained_filename = sorted_by_mtime_descending[0]
    resume_path = pretrained_filename
    pl.seed_everything(7)  # To be reproducable
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        state_dict = torch.load(pretrained_filename)
        model = NP(args.learning_rate)
        model.load_state_dict(state_dict['state_dict'],strict=False)
        #model = NP(args.learning_rate,args.affine_type).load_from_checkpoint(pretrained_filename)
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        model = NP(args.learning_rate)
        trainer.fit(model, train_dataloader, val_dataloader,ckpt_path= resume_path)
    return model


if __name__ == "__main__":
    #[1,1e-1,1e-2,1e-3]
    main(sys.argv[1:])

'''
python Mytrain_UNet_reg_tv_multi_datasets -l 0.5
python Mytrain_UNet_reg_tv_multi_datasets.py -l 0.1
python Mytrain_UNet_reg_tv_multi_datasets -l 0.01
python Mytrain_UNet_reg_tv_multi_datasets -l 2


'''
