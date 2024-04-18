from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import datetime
import torch
import os
import shutil
import inspect
import argparse


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

    # Path to the configs.py in the configs/ directory relative to the caller
    # data_configs_path = os.path.join(os.path.dirname(caller_script_path), "configs.py")
    #
    # # Check if the configs.py file exists
    # if os.path.exists(data_configs_path):
    #     # Copy the configs.py to the destination directory
    #     shutil.copy(data_configs_path, destination_directory)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x dim]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore


def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                        D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                            device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                              D))  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore