from fairseq import models
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from torch import nn


class KWS(nn.Module):

    def __init__(self, n_class=30, encoder_hidden_dim=768, w2v_sd=None):
        super(KWS, self).__init__()

        self.n_class = n_class
        
        cfg = convert_namespace_to_omegaconf(w2v_sd['args'])
        task = tasks.setup_task(cfg.task)
        state_dict = w2v_sd['model']

        assert not cfg is None
        assert not state_dict is None
        
        self.w2v_encoder = task.build_model(cfg.model)
        self.w2v_encoder.load_state_dict(state_dict)
        
        out_channels = 112
        self.decoder = nn.Sequential(
            nn.Conv1d(encoder_hidden_dim, out_channels, 25, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, self.n_class, 1)
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        output = self.w2v_encoder(**x, features_only=True)
        output= output['x']
        b,t,c = output.shape
        output = output.reshape(b,c,t)
        output = self.decoder(output).squeeze()
        if self.training:
            return self.softmax(output)
        else:
            return output

