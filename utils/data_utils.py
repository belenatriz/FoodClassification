from datasets import load_dataset


def load_food_dataset(cache_path: str = None):
    """
    Read from cache path or download the food dataset from huggingface that is
    going to be used in this project

    :param cache_path: place to save/write the data. default one is
        `"~/.cache/huggingface/datasets"`

    :return: dataset object
    """

    dataset = load_dataset(
        "Kaludi/food-category-classification-v2.0", cache_dir=cache_path
    )

    return dataset
