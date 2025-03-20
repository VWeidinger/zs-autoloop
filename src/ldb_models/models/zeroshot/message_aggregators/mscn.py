import dgl.function as fn
from torch import nn

from models.zeroshot.message_aggregators.aggregator import MessageAggregator


class MscnConv(MessageAggregator):
    """
    A message aggregator that sums up child messages and afterward combines them with the current hidden state of the
    parent node using an MLP
    """

    def __init__(self, hidden_dim=0, layer_norm: bool = False, **kwargs):
        super().__init__(input_dim=2 * hidden_dim, output_dim=hidden_dim, **kwargs)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def forward(self, graph=None, etypes=None, in_node_types=None, out_node_types=None, feat_dict=None):
        if len(etypes) == 0:
            return feat_dict

        with graph.local_scope():
            graph.ndata['h'] = feat_dict

            # message passing
            graph.multi_update_all({etype: (fn.copy_u('h', 'm'), fn.sum('m', 'ft')) for etype in etypes},
                                   cross_reducer='sum')

            feat = graph.ndata['h']
            rst = graph.ndata['ft']

            out_dict = self.combine(feat, out_node_types, rst)

            if self.layer_norm is not None:
                out_dict = {k: self.layer_norm(v) for k, v in out_dict.items()}

            return out_dict
