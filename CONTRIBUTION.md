# Contribution Guide

Please read through the **Basic Conventions** section first, then depending on your purposes, choose one "If you ..." section to follow.

## Basic Conventions:

- Please reference **`What are the pros?`** section in `README.md`.

- Please branch out from the `develop` branch and **never** merge or modifiy the `master`.

## If you want to add new models...

Please reference existing models under `comvex` folder when you're implementing (recommand `convit`). Below are some main points:

1. Wrap all your implementation into a folder named with the model's name (all lowercases) under `comvex` folder.

2. At least creat: `__init__.py`, `config.py`, `model.py`, and `README.md`. Feel free to add extra files as needed.

3. `__init__.py`: Expose all objects you implement here.

4. `config.py`: Please inherit `ConfigBase` on your `XXXConfig` object by `from comvex.utils import ConfigBase`. Some naming convention to follow: `dim`, `heads`, `head_dim`, `num_classes`, `pred_act_fnc_name`, `ff_dim`, `ff_expand_scale`, `ff_dropout`, `attention_dropout`, `path_dropout`, `depth`, `image_channel`, `image_size`, `patch_size`, and `num_xxx`. (Pull a request here if you find more!)

5. `README.md`: Must have title, objects, usage, demo, and citation sections (please reference existing `README.md`)

6. `demo.py` and testing file: Please reference existing ones under `examples` and `tests` (or `tests/template.py`).

## If you find a bug or unclear comments:

Thanks and happy to accept any!
