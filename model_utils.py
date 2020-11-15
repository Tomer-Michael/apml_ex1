
from general_utils import relative_path_to_absolute_path
from models import SimpleModel


NAME_OF_LAST_CHECKPOINT = 'current_weights'
NAME_OF_PRE_TRAINED_CHECKPOINT = 'pre_trained'
CHECKPOINTS_FOLDER_RELATIVE_PATH = './res/checkpoints/'
CHECKPOINTS_FILES_EXTENSION = '.ckpt'


def get_path_to_checkpoint(checkpoint_name):
    relative_path = CHECKPOINTS_FOLDER_RELATIVE_PATH + checkpoint_name + CHECKPOINTS_FILES_EXTENSION
    return relative_path_to_absolute_path(relative_path)


def load_checkpoint(model, checkpoint_name=NAME_OF_LAST_CHECKPOINT):
    model.load(get_path_to_checkpoint(checkpoint_name))


def save_checkpoint(model, name, set_as_last_checkpoint=True):
    model.save(get_path_to_checkpoint(name))
    if set_as_last_checkpoint:
        model.save(get_path_to_checkpoint(NAME_OF_LAST_CHECKPOINT))


def create_last_model(is_in_eval_mode=False):
    create_model(checkpoint_name=NAME_OF_LAST_CHECKPOINT, is_in_eval_mode=is_in_eval_mode)


def create_pre_trained_model(is_in_eval_mode=False):
    create_model(checkpoint_name=NAME_OF_PRE_TRAINED_CHECKPOINT, is_in_eval_mode=is_in_eval_mode)


def create_model(checkpoint_name=NAME_OF_PRE_TRAINED_CHECKPOINT, is_in_eval_mode=False):
    model = SimpleModel()

    if checkpoint_name is not None:
        load_checkpoint(model, checkpoint_name)

    if is_in_eval_mode:
        model.eval()

    return model
