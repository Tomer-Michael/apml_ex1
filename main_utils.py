
import matplotlib
import matplotlib.pyplot as plt

from dataset import *
from general_utils import relative_path_to_absolute_path


def imshow_torch(image):
    np_image = un_normalize_image(image)
    imshow_numpy(np_image)


def imshow_numpy(image):
    plt.imshow(image)
    plt.show()


def get_path_to_image_folder_root(root_name):
    return 'res/data/' + root_name


def load_dataset(dataset_name, should_log_metadata=False):
    relative_path_to_dataset = 'res/data/' + dataset_name
    path = relative_path_to_absolute_path(relative_path_to_dataset)
    print('Path is: ', path)

    dataset = get_dataset_as_torch_dataset(path)
    print('Dataset loaded.')

    if should_log_metadata:
        log_dataset_metadata(dataset)

    return dataset


def log_dataset_metadata(dataset):
    print('Dataset total size is: ', str(len(dataset.the_list)))

    label_names_dict = label_names()
    print('It has ' + str(len(label_names_dict)) + ' classes.')
    print('Classes labels are ' + str(label_names_dict))
    amount_per_class = {key: 0 for key in label_names_dict.keys()}

    for image in dataset:
        amount_per_class[image[1]] += 1
    print('Amount of each class is: ', amount_per_class)


def dump_organized(dataset):
    print('Dumping dataset organized.')

    base = 'original_unfixed_data/'

    label_names_dict = label_names()
    paths = {key: relative_path_to_absolute_path(base + value) for key, value in label_names_dict.items()}
    amount_per_class = {key: 0 for key in label_names_dict.keys()}

    for image in dataset:
        cur_count = amount_per_class[image[1]]
        cur_path = str(paths[image[1]]) + '_' + str(cur_count) + '.png'
        if cur_count % 50 == 0:
            print('Current path is: ' + cur_path)

        save_image_to_path(cur_path, image[0])

        amount_per_class[image[1]] += 1


def save_image_to_path(path_to_image, torch_image):
    np_image = un_normalize_image(torch_image)
    matplotlib.pyplot.imsave(path_to_image, np_image)
