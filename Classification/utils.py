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

    # Path to the configs.py in the configs/ directory relative to the caller
    # data_configs_path = os.path.join(os.path.dirname(caller_script_path), "configs.py")
    #
    # # Check if the configs.py file exists
    # if os.path.exists(data_configs_path):
    #     # Copy the configs.py to the destination directory
    #     shutil.copy(data_configs_path, destination_directory)


# def visualize():
#     visulaize_imgs = False
#     if visulaize_imgs:
#         # Visualize some examples
#         NUM_IMAGES = 4
#         CIFAR_images = torch.stack([val_set[idx][0] for idx in range(NUM_IMAGES)], dim=0)
#         img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
#         img_grid = img_grid.permute(1, 2, 0)
#
#         plt.figure(figsize=(8, 8))
#         plt.title("Image examples of the CIFAR10 dataset")
#         plt.imshow(img_grid)
#         plt.axis("off")
#         plt.show()
#         plt.close()
#     if visulaize_imgs:
#         img_patches = img_to_patch(CIFAR_images, patch_size=4, flatten_channels=False)
#         fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(14, 3))
#         fig.suptitle("Images as input sequences of patches")
#         for i in range(CIFAR_images.shape[0]):
#             img_grid = torchvision.utils.make_grid(img_patches[i], nrow=64, normalize=True, pad_value=0.9)
#             img_grid = img_grid.permute(1, 2, 0)
#             ax[i].imshow(img_grid)
#             ax[i].axis("off")
#         plt.show()
#         plt.close()

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


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

import matplotlib.pyplot as plt
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

import matplotlib.colors as mcolors
def _plot_umap(model, data_loader, save_dir='.'):
    # import umap
    import umap.plot
    from matplotlib.colors import ListedColormap
    num_classes = len(np.unique(data_loader.dataset.y_data))
    classes_names = [f'Class {i}' for i in range(num_classes)]

    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 17}
    plt.rc('font', **font)

    with torch.no_grad():
        # Source flow
        data = data_loader.dataset.x_data.float().cuda()
        labels = data_loader.dataset.y_data.view((-1)).long()
        features = model(data)

    if not os.path.exists(os.path.join(save_dir, "umap_plots")):
        os.mkdir(os.path.join(save_dir, "umap_plots"))

    # cmaps = plt.get_cmap('jet')
    model_reducer = umap.UMAP( n_neighbors=5, min_dist=1, metric='euclidean', random_state=42)
    embedding = model_reducer.fit_transform(features.detach().cpu().numpy())

    # Normalize the labels to [0, 1] for colormap
    # norm_labels = labels / 4.0
    #
    # # Create a new colormap by extracting the first 5 colors from "Paired"
    # paired = plt.cm.get_cmap('Paired', 12)  # 12 distinct colors
    # new_colors = [paired(0), paired(1), paired(2), paired(4),
    #               paired(6)]  # Skip every second color, but take both from the first pair
    # new_cmap = ListedColormap(new_colors)
    # cmap = plt.cm.get_cmap('viridis', num_classes)
    # Set the colormap
    if num_classes <= 5:
        cmap = plt.cm.get_cmap('viridis', num_classes)
    elif num_classes <= 20:
        cmap = plt.cm.tab20
    elif num_classes <= 40:
        cmap = plt.cm.tab20b  # Additional set of 20 colors
    elif num_classes <= 60:
        cmap = plt.cm.tab20c  # Another additional set of 20 colors
    else:
        # For more than 60 classes, create a new colormap from tab20* colormaps
        colors = list(mcolors.TABLEAU_COLORS.keys())  # Get a set of colors
        while len(colors) < num_classes:
            colors.extend(colors)  # Repeat the color list to ensure enough colors
        cmap = mcolors.ListedColormap(colors[:num_classes])

    print("Plotting UMAP ...")
    plt.figure(figsize=(16, 10))
    # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=norm_labels, cmap=new_cmap, s=15)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=15)

    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, classes_names, title="Classes")
    file_name = "umap_.pdf"
    fig_save_name = os.path.join(save_dir, "umap_plots", file_name)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(fig_save_name, bbox_inches='tight')
    plt.close()