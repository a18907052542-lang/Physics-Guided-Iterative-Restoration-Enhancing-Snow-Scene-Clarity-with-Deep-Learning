"""
PAID-SnowNet: Physics-Aware Iterative Denoising Network.

Architecture (Fig. 1 of paper):
    Input snowy image
        -> SPM (Snow Physics Modeling)  : scattering / transmission / occlusion / blur params
        -> SIRM (Snow Iterative Restoration): AdaGrad + U-Net completion, T iterations
        -> Restored image

Ablation flags propagate into both SPM and SIRM so that every cell of
Tables 3, 5, 6, 12, 13, 14, 15 in the paper can be run from a single class.
"""

import torch
import torch.nn as nn

from .spm_module import SPMModule
from .sirm_module import SIRMModule


class PAIDSnowNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 # ---- SPM hyperparameters ----
                 spm_branch_channels=64,
                 spm_kernel_sizes=(3, 5, 7),
                 spm_attention_dim=32,
                 spm_disable_mie=False,
                 spm_disable_rayleigh=False,
                 spm_disable_occlusion=False,
                 spm_fusion_mode='attention',
                 use_spm=True,
                 # ---- SIRM hyperparameters ----
                 sirm_iterations=3,
                 sirm_eta0=0.1,
                 sirm_epsilon=1e-8,
                 sirm_alpha_w=0.5, sirm_beta_w=0.3, sirm_gamma_w=0.2,
                 sirm_lambda1=0.01, sirm_lambda2=0.001,
                 unet_layers=4,
                 unet_base_channels=64,
                 convergence_threshold=1e-4,
                 sirm_inner_optimizer='adagrad',
                 sirm_completion_type='unet',
                 sirm_use_physics_weight=True):
        super().__init__()

        self.use_spm = use_spm

        self.spm = SPMModule(
            in_channels=in_channels,
            branch_channels=spm_branch_channels,
            kernel_sizes=spm_kernel_sizes,
            attention_hidden_dim=spm_attention_dim,
            disable_mie=spm_disable_mie,
            disable_rayleigh=spm_disable_rayleigh,
            disable_occlusion=spm_disable_occlusion,
            fusion_mode=spm_fusion_mode,
        )

        self.sirm = SIRMModule(
            num_iterations=sirm_iterations,
            eta0=sirm_eta0,
            epsilon=sirm_epsilon,
            alpha_w=sirm_alpha_w, beta_w=sirm_beta_w, gamma_w=sirm_gamma_w,
            lambda1=sirm_lambda1, lambda2=sirm_lambda2,
            unet_layers=unet_layers,
            unet_base_channels=unet_base_channels,
            convergence_threshold=convergence_threshold,
            inner_optimizer=sirm_inner_optimizer,
            completion_type=sirm_completion_type,
            use_physics_weight=sirm_use_physics_weight,
        )

    def _neutral_physics_params(self, y):
        """Return neutral physics params (used when SPM is disabled).
        Transmission=1, occlusion=1, scattering=0, depth=0, sigma=1."""
        B, _, H, W = y.shape
        device = y.device
        return {
            'scattering_coeff': torch.zeros(B, 1, H, W, device=device),
            'transmission': torch.ones(B, 1, H, W, device=device),
            'occlusion_mask': torch.ones(B, 1, H, W, device=device),
            'psf_sigma': torch.ones(B, 1, device=device),
            'depth_map': torch.zeros(B, 1, H, W, device=device),
            'fused_features': torch.zeros_like(y[:, :1]),
        }

    def forward(self, x):
        if self.use_spm:
            physics_params = self.spm(x)
        else:
            physics_params = self._neutral_physics_params(x)
        restored, intermediates = self.sirm(x, physics_params)
        return restored, physics_params, intermediates

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config):
        return cls(
            in_channels=config.spm_in_channels,
            spm_branch_channels=config.spm_branch_channels[0],
            spm_kernel_sizes=config.spm_branch_kernel_sizes,
            spm_attention_dim=config.spm_attention_hidden_dim,
            sirm_iterations=config.sirm_iteration_count,
            sirm_eta0=config.sirm_initial_step_size,
            sirm_epsilon=config.sirm_adagrad_epsilon,
            sirm_alpha_w=config.alpha_w_occlusion,
            sirm_beta_w=config.beta_w_transmittance,
            sirm_gamma_w=config.gamma_w_constant,
            sirm_lambda1=config.lambda1_sparsity,
            sirm_lambda2=config.lambda2_smoothness,
            unet_layers=config.sirm_unet_layers,
            unet_base_channels=config.sirm_unet_base_channels,
            convergence_threshold=config.convergence_threshold,
        )
