import cv2
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm as plt_cmap
from sklearn.metrics import confusion_matrix as sk_cm


def display_random_image_in_actual_size(image_path):
    dpi = 80
    image = np.asarray(Image.open(image_path))
    height = image.shape[0]
    width = image.shape[1]
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    # Display the image.
    ax.imshow(image)
    plt.show()


def plot_random_images_from_dataframe(dataframe,
                                      column_path_header="abs_path",
                                      column_label_header="label",
                                      n=25,
                                      specific_classes=None,
                                      figsize=(12, 12),
                                      color="white",
                                      title=None,
                                      output_file_path=None):
    if specific_classes is not None:
        df_list = list()
        for specific_class in specific_classes:
            df_list.append(dataframe[dataframe[column_label_header] == specific_class])
        dataframe = pd.concat(df_list)
        if len(df_list) == 0:
            print("{} not found in dataframe".format(specific_classes))

    random_df = dataframe.sample(n=n, replace=False)
    random_labels = list(random_df[column_label_header])

    if np.modf(np.sqrt(n))[0] == 0.0:
        fig, axs = plt.subplots(int(np.sqrt(n)), int(np.sqrt(n)), figsize=figsize)
        axs = axs.ravel()
        for _j, ax in enumerate(axs):
            ax.imshow(np.asarray(Image.open(list(random_df[column_path_header])[_j])))
            ax.axis("off")
            ax.set_title(random_labels[_j], color=color)
        if output_file_path:
            fig.savefig(output_file_path)
        if title:
            fig.suptitle(title)
    else:
        print("Radice di n non intera!")


def extract_random_image_and_labels_from_dataframe(dataframe,
                                                   column_path_header="abs_path",
                                                   column_label_header="label",
                                                   resize=None,
                                                   n=25):
    images = list()
    random_df = dataframe.sample(n=n, replace=False)
    random_labels = list(random_df[column_label_header])

    for _j in range(n):
        image_numpy = np.asarray(Image.open(list(random_df[column_path_header])[_j]))
        if resize:
            image_numpy = cv2.resize(np.asarray(image_numpy), (resize, resize))
        images.append(image_numpy)
    return np.array(images), np.array(random_labels)


def normalize_dataset(images, mean_image=None):
    if mean_image is None:
        mean_image = np.mean(images, axis=(0, 1, 2))

    output = np.asarray((images - mean_image) / 255.0, dtype=np.float32)
    return output, mean_image


def denormalize_dataset(images, mean_image):
    denorm = images * 255
    print("Denorm shape: {}".format(denorm.shape))
    print("Mean image: {}".format(mean_image.shape))
    output = np.zeros(denorm.shape, dtype=np.uint8)
    if len(images.shape) > 3:
        for i in range(output.shape[-1]):
            print(output[:, :, :, i].shape)
            output[:, :, :, i] = denorm[:, :, :, i] + mean_image[i]
    else:
        output = denorm + mean_image
    return output


def shuffle_images_and_labels(images, labels, auxiliary_labels=None):
    rand_indexes = np.random.permutation(images.shape[0])
    shuffled_images = images[rand_indexes]
    shuffled_labels = labels[rand_indexes]
    if auxiliary_labels is not None:
        shuffled_auxiliary_labels = auxiliary_labels[rand_indexes]
    else:
        shuffled_auxiliary_labels = None
    return shuffled_images, shuffled_labels, shuffled_auxiliary_labels


def seaborn_cm(cm, ax, tick_labels, fontsize=14):

    group_counts = ["{:0.0f}".format(value) for value in cm.flatten()]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)
    group_percentages = ["{:0.0f}".format(value*100) for value in cm.flatten()]
    cm_labels = [f"{c}\n{p}%" for c, p in zip(group_counts, group_percentages)]
    cm_labels = np.asarray(cm_labels).reshape(len(tick_labels), len(tick_labels))
    sns.heatmap(cm,
                ax=ax,
                annot=cm_labels,
                fmt='',
                cbar=False,
                cmap=plt_cmap.Greys,
                linewidths=1, linecolor='black',
                annot_kws={"fontsize": fontsize},
                xticklabels=tick_labels,
                yticklabels=tick_labels)
    ax.set_yticklabels(ax.get_yticklabels(), size=fontsize, rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), size=fontsize, rotation=45)


def plot_confusion_matrix(y_test,
                          y_preds,
                          axes,
                          labels_categorical,
                          set_ticks=True,
                          title=None,
                          filepath_plot=None,
                          cm_computed_yet=None,
                          font_size=None,
                          tight_layout=True):

    if cm_computed_yet is None:
        cm = sk_cm(y_test, y_preds)
    else:
        cm = cm_computed_yet
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    axes.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap("Greys"))

    fmt = '.2f'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if np.isnan(cm[i, j]):
            cm[i, j] = 0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black",
                  fontsize=font_size)

    if not font_size:
        font_size = 18
    if set_ticks:
        tick_marks = np.arange(len(labels_categorical))
        axes.set_xticks(tick_marks)
        axes.set_xticklabels(labels_categorical, rotation=35, fontsize=font_size)
        axes.set_yticks(tick_marks)
        axes.set_yticklabels(labels_categorical, rotation=35, fontsize=font_size)

        axes.set_ylabel('True label', fontsize=font_size)
        axes.set_xlabel('Predicted label', fontsize=font_size)
    else:
        axes.set_xticks([])
        axes.set_yticks([])

    mean_acc = np.mean(np.diag(cm))
    if title is None:
        axes.set_title("Mean accuracy = {:.3f}".format(mean_acc))
    else:
        axes.set_title(title)
    axes.grid(False)
    if tight_layout:
        plt.tight_layout()

    if filepath_plot:
        plt.savefig(filepath_plot)
    return cm


def output_message(message):
    print('#'*len(message))
    print(message)
    print('#' * len(message))
