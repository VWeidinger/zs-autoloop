import math

import torch
from ldb_models.models.zeroshot.utils import activations
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class DynamicLayer(nn.Module):
    """
    A layer of an MLP with a specifiable dropout, norm and activation class
    """

    def __init__(
        self,
        p_dropout=None,
        activation_class_name=None,
        activation_class_kwargs=None,
        norm_class_name=None,
        norm_class_kwargs=None,
        inplace=False,
        **kwargs,
    ):
        super().__init__()
        # initialize base NN
        self.inplace = inplace
        self.p_dropout = p_dropout
        self.act_class = activations.__dict__[activation_class_name]
        self.act_class_kwargs = activation_class_kwargs
        self.norm_class = nn.__dict__[norm_class_name]
        self.norm_class_kwargs = norm_class_kwargs

    def get_act(self):
        if "inplace" in self.act_class.__init__.__code__.co_varnames:
            return self.act_class(inplace=self.inplace, **self.act_class_kwargs)
        else:
            return self.act_class(**self.act_class_kwargs)

    def get_norm(self, num_feats):
        # Some norms (e.g., BatchNorm variants) expose `inplace`, while others
        # (e.g., LayerNorm/Identity) do not; handle both signatures safely.
        if "inplace" in self.norm_class.__init__.__code__.co_varnames:
            return self.norm_class(
                num_feats, inplace=self.inplace, **self.norm_class_kwargs
            )
        else:
            return self.norm_class(num_feats, **self.norm_class_kwargs)


# Residual block
class ResidualBlock(DynamicLayer):
    """
    One residual layer in an MLP.
    """

    def __init__(
        self,
        layer_in,
        layer_out,
        norm: bool = None,
        activation=False,
        dropout=False,
        final_out_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if norm is None:
            # Interpret omitted `norm` as a boolean derived from configured norm class.
            norm = kwargs.get("norm_class_name") is not None and not final_out_layer
        if final_out_layer:
            activation = False
            dropout = False

        assert layer_out == layer_in
        hidden_dim = layer_in

        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        if norm:
            self.layers.append(self.get_norm(hidden_dim))
        if activation:
            self.layers.append(self.get_act())

        # if dropout:
        #     self.layers.append(nn.Dropout(self.p_dropout))
        last_linear = nn.Linear(hidden_dim, hidden_dim)
        # Start residual branch near-identity; this can affect optimization speed,
        # so keep an eye on convergence metrics when reusing this block in new setups.
        torch.nn.init.zeros_(last_linear.weight)
        if last_linear.bias is not None:
            torch.nn.init.zeros_(last_linear.bias)
        self.layers.append(last_linear)
        if norm:
            self.layers.append(self.get_norm(hidden_dim))
        if dropout and not final_out_layer:
            # this can never be run inplace since it raises an error
            self.layers.append(nn.Dropout(self.p_dropout, inplace=False))

        if activation:
            self.final_act_layer = self.get_act()

    def forward(self, x):
        residual = x
        out = x
        for layer in self.layers:
            out = layer(out)

        out += residual
        if self.activation:
            out = self.final_act_layer(out)

        return out


class FcLayer(DynamicLayer):
    """
    One layer in an MLP.
    """

    def __init__(
        self,
        layer_in,
        layer_out,
        norm: bool = None,
        activation=False,
        dropout=False,
        final_out_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if norm is None:
            # Interpret omitted `norm` as a boolean derived from configured norm class.
            norm = kwargs.get("norm_class_name") is not None and not final_out_layer
        if final_out_layer:
            activation = False
            dropout = False

        assert layer_out > 0, (
            "Output dimension must be greater than 0, but got 0. Check the model configuration."
        )
        layers = [nn.Linear(layer_in, layer_out)]
        if activation:
            layers.append(self.get_act())
        if norm:
            layers.append(self.get_norm(layer_out))
        if dropout and not final_out_layer:
            # this can never be run inplace since it raises an error
            layers.append(nn.Dropout(self.p_dropout, inplace=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FcOutModel(DynamicLayer):
    """
    Model with an MLP (Multi-Layer-Perceptron) on top.
    """

    def __init__(
        self,
        output_dim=None,
        input_dim=None,
        n_layers=None,
        width_factor=None,
        residual=True,
        task=None,
        **kwargs,
    ):
        is_final_output_model = kwargs.pop("final_out_layer", False)
        super().__init__(**kwargs)

        assert output_dim > 0, (
            "Output dimension must be greater than 0, but got 0. Check the model configuration."
        )

        self.task = task
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.no_input_required = input_dim == 0

        if self.input_dim == 0:
            self.replacement_param = Parameter(torch.Tensor(self.output_dim))
            # init.kaiming_uniform_(self.replacement_param, a=math.sqrt(5))
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.replacement_param)
            bound = 1 / math.sqrt(self.output_dim)
            init.uniform_(self.replacement_param, -bound, bound)
            return

        assert width_factor is not None, (
            "Width factor must be specified for the output model."
        )
        assert self.input_dim is not None, (
            "Input dimension must be specified for the output model."
        )

        layer_dims = (
            [self.input_dim]
            + [int(width_factor * self.input_dim)] * n_layers
            + [self.output_dim]
        )

        layers = []
        for idx, (layer_in, layer_out) in enumerate(zip(layer_dims, layer_dims[1:])):
            # `len(layer_dims) - 2` is the final edge in the dims chain, i.e.,
            # the last Linear mapping before output.
            is_last_layer = idx == len(layer_dims) - 2
            layer_kwargs = {
                **kwargs,
                "final_out_layer": is_final_output_model and is_last_layer,
            }
            if not residual or layer_in != layer_out:
                layers.append(FcLayer(layer_in, layer_out, **layer_kwargs))
            else:
                layers.append(ResidualBlock(layer_in, layer_out, **layer_kwargs))

        self.fcout = nn.Sequential(*layers)
