import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch

def get_entropy_loss(latent_embed, codebook_embed, inv_entropy_tau: float = 1.0):
    # N = latent_embed.shape[0]
    # K = codebook_embed.shape[0]
    x2 = latent_embed.pow(2).sum(dim=1, keepdim=True)          # (N, 1)
    e2 = codebook_embed.pow(2).sum(dim=1, keepdim=True).T      # (1, K)
    E_dist = x2 + e2 - 2 * (latent_embed @ codebook_embed.T)   # (N, K)

    logits = -E_dist * inv_entropy_tau                         # (N, K)

    # q(k|x)
    prob = logits.softmax(dim=-1)          # (N, K)
    log_prob = logits.log_softmax(dim=-1)  # (N, K)
    per_sample_entropy = (-prob * log_prob).sum(dim=-1).mean()   # scalar
    avg_prob = prob.mean(dim=0)                                  # (K,)
    log_avg_prob = torch.log(avg_prob + 1e-7)
    codebook_entropy = (-avg_prob * log_avg_prob).sum()          # scalar
    entropy_loss = per_sample_entropy - codebook_entropy
    return entropy_loss

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.requires_grad = False
        self.embedding_proj = nn.Linear(self.code_dim, self.code_dim)
        nn.init.normal_(self.codebook.weight, mean=0, std=self.code_dim**-0.5)

        self.register_buffer(
            "code_use_count",
            torch.zeros(num_codes, dtype=torch.long)
        )

        self.register_buffer(
            "total_tokens",
            torch.zeros(1, dtype=torch.long)
        )

    @torch.no_grad()
    def reset_stats(self):
        self.code_use_count.zero_()
        self.total_tokens.zero_()

    def forward(self, z):
        z_flat = z.reshape(-1, self.code_dim)        # (B*N, C)
        # (B*N, num_codes)
        quant_codebook = self.embedding_proj(self.codebook.weight)
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flat, rearrange(quant_codebook, 'n d -> d n'))
        indices = torch.argmin(d, dim=1)             # (B*N,)
        # .z_q = self.codebook(indices)                 # (B*N, C)
        # z_q = z_q.view_as(z)                         # reshape
        z_q = F.embedding(indices, quant_codebook)          # (B*N, C)
        z_q = z_q.view_as(z)                                # (B, N, C)
        with torch.no_grad():
            flat_idx = indices.view(-1) 
            self.total_tokens += flat_idx.numel()
            self.code_use_count += torch.bincount(
                flat_idx,
                minlength=self.num_codes
            )
        return z_q, indices.view(z.shape[:-1])
    # def forward(self, z):
    #     z_flat = z.reshape(-1, self.code_dim)        # (B*N, C)
    #     quant_codebook = self.embedding_proj(self.codebook.weight)
    #     z_norm = z_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    #     z_flat_norm = z_flat / z_norm
    #     code_norm = quant_codebook.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    #     quant_codebook_norm = quant_codebook / code_norm

    #     d = torch.sum(z_flat_norm ** 2, dim=1, keepdim=True) + \
    #         torch.sum(quant_codebook_norm ** 2, dim=1) - 2 * \
    #         torch.einsum('bd,dn->bn',
    #                     z_flat_norm,
    #                     rearrange(quant_codebook_norm, 'n d -> d n'))

    #     indices = torch.argmin(d, dim=1)
    #     z_q = self.codebook(indices)                            
    #     z_q = z_q / (z_q.norm(dim=-1, keepdim=True).clamp(min=1e-6))
    #     z_q = z_q * z_norm                                      
    #     z_q = z_q.view_as(z)
    #     with torch.no_grad():
    #         flat_idx = indices.view(-1) 
    #         self.total_tokens += flat_idx.numel()
    #         self.code_use_count += torch.bincount(
    #             flat_idx,
    #             minlength=self.num_codes
    #         )
    #     return z_q, indices.view(z.shape[:-1])

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv(path_1)
        
        return out
    
    def forward_features(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        return path_4, path_3, path_2, path_1        
        # return out


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        self.code_dim = self.backbone.embed_dim
        self.num_codes = 16384
        self.vq = VectorQuantizer(self.num_codes, self.code_dim)
        self.qpm_eps = 0.1
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        
    # def qpm(self, indices):
    #     if self.qpm_eps <= 0:
    #         return indices
    #     B, N = indices.shape
    #     device = indices.device
    #     keep_mask = torch.rand(B, N, device=device) > self.qpm_eps
    #     random_indices = torch.randint(
    #         low=0, high=self.vq.num_codes, size=(B, N), device=device
    #     )
    #     new_indices = torch.where(keep_mask, indices, random_indices)
    #     return new_indices
    def qpm(self, indices, epo=0.1, temperature: float = 1.0):
        self.qpm_eps = epo
        if self.qpm_eps <= 0:
            return indices
        B, N = indices.shape
        device = indices.device
        idx_flat = indices.view(-1)
        perturb_mask = torch.rand_like(idx_flat.float(), device=device) < self.qpm_eps
        if not perturb_mask.any():
            return indices
        codebook_q = self.vq.embedding_proj(self.vq.codebook.weight).to(device)
        unique_ids = idx_flat[perturb_mask].unique()
        new_idx_flat = idx_flat.clone()
        for cid in unique_ids:
            cid_int = int(cid.item())
            ci = codebook_q[cid_int]
            d = (codebook_q - ci).pow(2).sum(dim=1)
            d[cid_int] = float('inf')
            logits = -d / temperature
            probs = torch.softmax(logits, dim=0)
            pos = (idx_flat == cid) & perturb_mask
            num_pos = pos.sum().item()
            if num_pos == 0:
                continue
            sampled_ids = torch.multinomial(probs, num_samples=num_pos, replacement=True)
            new_idx_flat[pos] = sampled_ids
        return new_idx_flat.view_as(indices)

    # def loss_code_usage(self, eps: float = 1e-8):
    #     if getattr(self.vq, "last_indices", None) is None:
    #         return next(self.parameters()).new_zeros(())
    #     flat_idx = self.vq.last_indices.view(-1)
    #     if flat_idx.numel() == 0:
    #         return next(self.parameters()).new_zeros(())
    #     hist = torch.bincount(flat_idx, minlength=self.num_codes).float()

    #     total = hist.sum()
    #     if total.item() == 0:
    #         return next(self.parameters()).new_zeros(())

    #     probs = hist / (total + eps)         # (num_codes,)
    #     entropy = -(probs * (probs + eps).log()).sum() 
    #     loss_usage = -entropy
    #     return loss_usage
    def loss_code_usage(self, z_cont, inv_entropy_tau: float = 1.0):
        B, N, C = z_cont.shape
        z_flat = z_cont.reshape(-1, C)   # (B*N, C)
        codebook_q = self.vq.embedding_proj(self.vq.codebook.weight)  # (K, C)
        return get_entropy_loss(z_flat, codebook_q, inv_entropy_tau)


    # def apply_vq_qpm(self, features, use_qpm=False, ids=None):
    def apply_vq_qpm(self, features, use_qpm=False, ids=[3], use_pfa=False, epo=0.1):
        if ids is None:
            ids = []
        vq_features = []
        align_cont_features = []

        for i, feat in enumerate(features):
            if i not in ids:
                vq_features.append(feat)
                if use_pfa:
                    align_cont_features.append(None)
                continue

            z_cont = feat
            z_q_raw, indices = self.vq(feat)

            if use_pfa:
                # align_cont_features.append(z_cont)
                align_cont_features.append((z_q_raw, z_cont))

            if use_qpm:
                perturbed_idx = self.qpm(indices, epo=epo)
                # z_q = self.vq.codebook(perturbed_idx)
                codebook_q = self.vq.embedding_proj(self.vq.codebook.weight)
                z_q_raw = F.embedding(perturbed_idx, codebook_q)

            # z_q = z_cont + (z_q - z_cont).detach()
            z_q_ste = z_cont + (z_q_raw - z_cont).detach()
            vq_features.append(z_q_ste)

        if use_pfa:
            return vq_features, align_cont_features
        return vq_features

    # def loss_pfa(self, f_pfa, f_fm, tau: float = 0.07):
    #     # f_pfa, f_fm: (B, C, H, W)
    #     B, C, H, W = f_pfa.shape
    #     L = H * W
    #     q = f_pfa.view(B, C, L).permute(0, 2, 1)  # (B, L, C)
    #     k = f_fm.view(B, C, L).permute(0, 2, 1)   # (B, L, C)
    #     q = F.normalize(q, dim=-1)
    #     k = F.normalize(k, dim=-1)
    #     logits = torch.bmm(q, k.transpose(1, 2)) / tau  # (B, L, L)
    #     targets = torch.arange(L, device=logits.device).unsqueeze(0).expand(B, -1)
    #     loss = F.cross_entropy(
    #         logits.reshape(B * L, L),
    #         targets.reshape(B * L)
    #     )
    #     return loss

    # def loss_pfa(self, f_pfa, f_fm, tau: float = 1):
    #     # f_pfa, f_fm: (B, L, C)
    #     B, L, C = f_pfa.shape
    #     # print(tau)
    #     q = F.normalize(f_pfa, dim=-1)   # (B, L, C)
    #     k = F.normalize(f_fm.detach(), dim=-1)

    #     logits = torch.bmm(q, k.transpose(1, 2)) / tau   # (B, L, L)

    #     targets = torch.arange(L, device=logits.device).unsqueeze(0).expand(B, -1)  # (B, L)

    #     loss = F.cross_entropy(
    #         logits.reshape(B * L, L),
    #         targets.reshape(B * L)
    #     )
    #     return loss

    def loss_pfa(self, f_vq, f_fm, beta: float = 0.25):
        # f_pfa = F.normalize(f_pfa, dim=-1)
        # f_fm  = F.normalize(f_fm, dim=-1)
        # f_pfa = F.normalize(f_pfa, dim=-1)
        # f_fm  = F.normalize(f_fm,  dim=-1)
        loss_embed = (f_vq - f_fm.detach())**2
        loss_commit = (f_fm - f_vq.detach())**2
        # return 0.1*(loss_embed.mean() + beta * loss_commit.mean())
        return loss_embed.mean() + beta * loss_commit.mean()
        # return loss_embed.mean()

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    # def forward_features(self, x, use_qpm=False, use_pfa=False):
    #     patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

    #     features = self.backbone.get_intermediate_layers(
    #         x, self.intermediate_layer_idx[self.encoder_size]
    #     )   # list of 4 tensors, each (B, N, C)

    #     # === QPM ===
    #     features, align_cont_features = self.apply_vq_qpm(features, use_qpm=use_qpm, use_pfa=use_pfa)
    #     out_features = self.head.forward_features(features, patch_h, patch_w)
    #     if use_pfa:
    #         loss_pfa_val = self.loss_pfa(features[-1], align_cont_features[-1], tau=0.07)
    #         return out_features, loss_pfa_val
    #     return out_features

    # def forward(self, x, comp_drop=False):
    #     patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
    #     features = self.backbone.get_intermediate_layers(
    #         x, self.intermediate_layer_idx[self.encoder_size]
    #     )
        
    #     if comp_drop:
    #         bs, dim = features[0].shape[0], features[0].shape[-1]
            
    #         dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
    #         dropout_mask2 = 2.0 - dropout_mask1
    #         dropout_prob = 0.5
    #         num_kept = int(bs // 2 * (1 - dropout_prob))
    #         kept_indexes = torch.randperm(bs // 2)[:num_kept]
    #         dropout_mask1[kept_indexes, :] = 1.0
    #         dropout_mask2[kept_indexes, :] = 1.0
            
    #         dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            
    #         features = (feature * dropout_mask.unsqueeze(1) for feature in features)
            
    #         out = self.head(features, patch_h, patch_w)
            
    #         out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
            
    #         return out
        
    #     out = self.head(features, patch_h, patch_w)
    #     out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        
    #     return out
    def forward(self, x, use_qpm=False, use_pfa=False, use_usage=False, epo=0.1):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )   # list of (B, N, C)
        # features = self.apply_vq_qpm(features, use_qpm=use_qpm, use_pfa=use_pfa)
        # out = self.head(features, patch_h, patch_w)
        # out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        # return out
        if use_pfa:
            # features, align_cont_features = self.apply_vq_qpm(features, use_qpm=use_qpm, use_pfa=use_pfa)
            features, align_cont_pairs = self.apply_vq_qpm(features, use_qpm=use_qpm, use_pfa=use_pfa, epo=epo)
        else:
            features = self.apply_vq_qpm(features, use_qpm=use_qpm, use_pfa=use_pfa, epo=epo)
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        # if use_pfa:
        #     # loss_pfa_val = self.loss_pfa(features[-1], align_cont_features[-1])
        #     z_q_raw, z_cont = align_cont_pairs[-1]
        #     loss_pfa_val = self.loss_pfa(z_q_raw, z_cont) 
        #     return out, loss_pfa_val
        # return out
        if not use_pfa and not use_usage:
            return out

        z_q_raw, z_cont = align_cont_pairs[-1]

        loss_pfa_val = None
        loss_usage_val = None

        if use_pfa:
            loss_pfa_val = self.loss_pfa(z_q_raw, z_cont)

        if use_usage:
            loss_usage_val = self.loss_code_usage(z_cont, inv_entropy_tau=1.0)

        return out, loss_pfa_val, loss_usage_val

