from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import datetime
import torch
import os
import shutil
import inspect
from einops import rearrange
import numpy as np

def get_clf_report(model, dataloader, save_dir, class_names):
    # GET THE PERFORMANCE ON THE BEST EPOCH ...
    model.eval()
    model = model.cuda()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            try:
                data = batch['samples'].type(torch.float).cuda()
                labels = batch['labels'].to(torch.int64).cuda()
            except:
                data, labels = batch
                data = data.type(torch.float).cuda() # rearrange(data, 'B L C -> B C L').type(torch.float).cuda()
                labels = labels.squeeze().to(torch.int64).cuda()

            # to the model ...
            logits = model(data)

            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    clf_report = classification_report(targets, predictions, target_names=class_names,
                                       digits=4, output_dict=True)
    df = pd.DataFrame(clf_report)
    accuracy = accuracy_score(targets, predictions)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = f"Best_classification_report_{datetime.datetime.now().strftime('%H_%M')}.xlsx"
    report_Save_path = os.path.join(save_dir, file_name)
    df.to_excel(report_Save_path)

def save_copy_of_files(checkpoint_callback):
    # Get the frame of the caller of this function
    caller_frame = inspect.currentframe().f_back

    # Get the filename of the caller
    caller_filename = caller_frame.f_globals["__file__"]

    # Get the absolute path of the caller script
    caller_script_path = os.path.abspath(caller_filename)

    # Destination directory (PyTorch Lightning saving directory)
    destination_directory = checkpoint_callback.dirpath

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Copy the caller script to the destination directory
    shutil.copy(caller_script_path, destination_directory)


import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

