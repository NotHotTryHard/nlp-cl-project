import importlib
import json
import logging
import os
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path

from src.logger import setup_logging
from src.utils import read_json, write_json, ROOT_PATH

class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training.
                       contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict {keychain: value}, specifying position values to be replaced
                             from config dict.
        :param run_id: Unique Identifier for training processes.
                       Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = str(save_dir / "models" / exper_name / run_id)
        self._log_dir = str(save_dir / "log" / exper_name / run_id)

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args, options="", hardcoded_val_names={}):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.json"
        else:
            msg_no_cfg = "Configuration file need to be specified. " \
                         "Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))
        
        # if a separate data config was passed, load it and override/insert under "data"
        if getattr(args, "data_config", None):
            config["data"] = read_json(Path(args.data_config))
        
        # if a modification config was passed, overwrite some fields
        if getattr(args, "mod_config", None):
            mod_config = read_json(args.mod_config)
            flattened_mod_config = flatten_mods(mod_config, sep=";")
            config = _update_config(config, flattened_mod_config, verbose=True)
        
        # report train batch_size
        if args.bs:
            print(f'Changed train batch_size to {args.bs}')

        # change all val splits batch_sizes:
        if args.task_type and args.val_batch_size:
            for val_name in hardcoded_val_names[args.task_type]:
                print(f'Changed {val_name} validation batch_size to {args.val_batch_size}') 
                config["data"][val_name]["batch_size"] = args.val_batch_size

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags))
            for opt in options if opt.target is not None
        }
        return cls(config, resume, modification)

    @staticmethod
    def init_obj(obj_dict, default_module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj(config['param'], module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if "module" in obj_dict:
            default_module = importlib.import_module(obj_dict["module"])

        module_name = obj_dict["type"]
        module_args = dict(obj_dict["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(default_module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return Path(self._save_dir)

    @property
    def log_dir(self):
        return Path(self._log_dir)

    @classmethod
    def get_default_configs(cls):
        config_path = ROOT_PATH / "src" / "config.json"
        with config_path.open() as f:
            return cls(json.load(f))

    @classmethod
    def get_test_configs(cls):
        config_path = ROOT_PATH / "src" / "tests" / "config.json"
        with config_path.open() as f:
            return cls(json.load(f))


# helper functions to update config dict with custom cli options
def _update_config(config, modification, verbose=False):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
            if verbose:
                print(f"Overriding / adding keys={k} to the main config!")
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    try:
        _get_by_path(tree, keys[:-1])[keys[-1]] = value
    except KeyError:
        _set_new_field_by_path(tree, keys, value)

def _set_new_field_by_path(tree, keys, value):
    for k in keys[:-1]:
        if k not in tree or not isinstance(tree[k], dict):
            tree[k] = {}
        tree = tree[k]
    tree[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)

def flatten_mods(nested, parent_key="", sep=";"):
    """Turn a nested dict like into depth-one dict with fields merged via sep"""
    flat = {}
    for k, v in nested.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_mods(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat
