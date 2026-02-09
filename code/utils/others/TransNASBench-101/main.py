import warnings
from TransNASBench101.api import TransNASBenchAPI as API
from experiments import main_exp


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    path2nas_bench_file = "TransNASBench101/transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)
    task_list = (
        api.task_list
    )  # ['class_scene', 'class_object', 'room_layout', 'jigsaw', 'segmentsemantic', 'normal', 'autoencoder']
    search_space_list = api.search_spaces  # ['macro', 'micro']
    # for task in task_list:
    # for search_space in search_space_list:
    task = "autoencoder"
    search_space = "micro"
    main_exp(
        task=task,
        search_space=search_space,
    )
