import inspect
import re
from typing import Dict

import classes.classes as cls
import torch.optim as opt
import training.featurizations as fts
from classes.classes import ModelConfig, Name
from training.featurizations import Featurization


def parameterized_config(base_config_name: str, **args):
    assert "-" not in base_config_name, (
        "Model type should not contain underscores. This is not supported on the receiver side!"
    )

    # encode all input arguments as list
    arg_string_list = []
    for key, value in args.items():
        if key == "wandb_group":
            continue

        assert "-" not in key, (
            "Arguments should not contain underscores. This is not supported on the receiver side!"
        )
        if value is True:
            arg_string_list.append(key)
        elif key == "featurization":
            arg_string_list.append(value.__name__)
        elif key == "optimizer_class_name":
            arg_string_list.append(value)
        elif key in ["pretrained_model_dir", "pretrained_model_name"]:
            # skip these
            # make sure they are passed via call argument - otherwise it will not be used on the cluster!
            continue
        else:
            arg_string_list.append(f"{key}{value}")
    model_name = f"pm-{base_config_name}-{'-'.join(arg_string_list)}"
    if model_name.endswith("-"):
        model_name = model_name[:-1]

    name = Name(model_name, model_name)

    # add a custom name in the args
    args["name"] = name

    # get the basic model configuration
    base_configuration = get_model_config(base_config_name, args)

    return base_configuration


def get_model_config(model_type: str, m_args: dict) -> ModelConfig:
    if model_type.startswith("pm-"):
        # this is a model string which is parameterized (i.e. in the string the arguments are encoded)
        return get_parameterized_model_config(model_type, m_args)

    classes_dict = dict()
    for name, obj in cls.__dict__.items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, ModelConfig)
            and "name" in obj.__annotations__
        ):
            instantiated_obj = obj()
            if instantiated_obj.name is None:
                continue
            assert instantiated_obj.name.NAME not in classes_dict, (
                f"Model type {instantiated_obj.name.NAME} already exists in classes.py. "
                "This is not allowed. Please change the name."
            )
            classes_dict[instantiated_obj.name.NAME] = obj

    if model_type not in classes_dict.keys():
        raise Exception(
            f"Model type {model_type} not found in classes.py \n{classes_dict.keys()}"
        )

    print(f"Got the following args: {m_args} for model type {model_type}")

    # filter out empty strings
    m_args = {k: v for k, v in m_args.items() if v != "" and k != ""}

    if len(m_args) == 0:
        return classes_dict[model_type]()

    return classes_dict[model_type](**m_args)


def split_if_ends_with_digits(input_string):
    # Match groups of non-digits and digits at the end of the string
    match = re.match(r"^(.*?)(\d+)$", input_string)
    if match:
        non_digit_part = match.group(1)
        digit_part = match.group(2)
        return non_digit_part, digit_part
    return (
        input_string,
        None,
    )  # Return the original string and None if no digits at the end


def get_parameterized_model_config(model_name: str, m_args: Dict) -> ModelConfig:
    assert model_name.startswith("pm-")

    args = model_name[3:].split("-")
    model_type = args[0]

    args = args[1:]
    args_dict = m_args.copy()

    # featurziation dict
    featurization_dict = get_featurization_dict()
    print(featurization_dict, flush=True)

    for arg in args:
        # check if arg is a featurization
        if arg in featurization_dict.keys():
            args_dict["featurization"] = featurization_dict[arg]
            continue
        if arg in opt.__dict__.keys() or arg in ["AdamWScheduleFree"]:
            args_dict["optimizer_class_name"] = arg
            continue

        # detect str key/value arguments
        str_keywords = [
            "ohem_slice_by",
            "subplan_monotonicity_loss_fn",
            "loss_class_name",
        ]
        found = False
        for str_keyword in str_keywords:
            if arg.startswith(str_keyword):
                args_dict[str_keyword] = arg[len(str_keyword) :]
                found = True
                break

        # we already handled this case - continue
        if found:
            continue

        # check if arg ends with number and if so, split it
        param_name, number = split_if_ends_with_digits(arg)

        if number is not None:
            args_dict[param_name] = int(number)
        elif param_name.endswith("False"):
            args_dict[param_name.replace("False", "")] = False
        else:
            # it is a flag - make it a boolean
            args_dict[param_name] = True

    args_dict["name"] = Name(model_name, model_name)

    return get_model_config(model_type, args_dict)


def get_featurization_dict():
    featurization_dict = dict()
    for name, obj in fts.__dict__.items():
        if inspect.isclass(obj) and issubclass(obj, Featurization):
            featurization_dict[name] = obj

    return featurization_dict
