
import pathlib

_PROJECT_ABSOLUTE_PATH = pathlib.Path(__file__).parent
GRAPHS_FOLDER_RELATIVE_PATH = 'res/graphs/'
GRAPH_FILES_EXTENSION = '.png'


def relative_path_to_absolute_path(relative_path):
    return _PROJECT_ABSOLUTE_PATH / relative_path


def get_path_for_graph(graph_name):
    return relative_path_to_absolute_path(GRAPHS_FOLDER_RELATIVE_PATH + graph_name + GRAPH_FILES_EXTENSION)
