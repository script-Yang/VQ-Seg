import torch
import torch.nn as nn
from model.semseg.dpt import DPT
class MyModel(nn.Module):
    def __init__(self, model_type='small', nclass=2, weights_path=None):
        super(MyModel, self).__init__()
        self.model_configs = {
            'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        model_config = self.model_configs.get(model_type, self.model_configs['small'])
        self.model = DPT(**{**model_config, 'nclass': nclass})
        
        if weights_path:
            self.load_weights(weights_path)

    @torch.no_grad()
    def reset_codebook_stats(self):
        # print(self.model)
        if hasattr(self.model, "vq") and hasattr(self.model.vq, "reset_stats"):
            self.model.vq.reset_stats()

    @torch.no_grad()
    def get_codebook_usage(self):
        assert hasattr(self.model, "vq")
        vq = self.model.vq
        code_use = vq.code_use_count      # (num_codes,)
        total_codes = vq.num_codes

        used_codes = int((code_use > 0).sum().item())
        usage_ratio = used_codes / float(total_codes)

        total_tokens = int(vq.total_tokens.item())
        avg_tokens_per_used_code = (
            total_tokens / max(used_codes, 1)
            if total_tokens > 0 else 0.0
        )
        return {
            "used_codes": used_codes,
            "total_codes": total_codes,
            "usage_ratio": usage_ratio,
            "avg_tokens_per_used_code": avg_tokens_per_used_code,
        }

    def forward(self, x, use_qpm=False, use_pfa=False, use_usage=False, epo = 0.1):
        # return self.model(x)
        return self.model(x, use_qpm=use_qpm, use_pfa=use_pfa, use_usage=use_usage, epo = epo)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        self.model.backbone.load_state_dict(state_dict)
        print(f"Model weights loaded from {weights_path}")

    def save_weights(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights saved to {save_path}")
