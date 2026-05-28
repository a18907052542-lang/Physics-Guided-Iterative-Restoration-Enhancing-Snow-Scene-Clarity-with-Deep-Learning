"""
Complete Hyperparameter Configuration for PAID-SnowNet
Matches Table 8 in the paper for full reproducibility.
"""

import os

class Config:
    # ======================== Paths ========================
    data_root = "./data/Snow100K"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    result_dir = "./results"

    # ======================== Training ========================
    learning_rate = 1e-4                # Initial learning rate for Adam
    lr_schedule = "CosineAnnealing"     # Decay to 1e-6 over 200 epochs
    lr_min = 1e-6
    batch_size = 32                     # Per-GPU batch size
    num_epochs = 200                    # Total training epochs
    random_seed = 42                    # For reproducibility
    num_workers = 8
    pin_memory = True

    # Data augmentation
    augment_hflip = True                # Horizontal flip
    augment_rotation = True             # 90-degree rotations
    train_patch_size = 256              # Input image resolution (training)

    # ======================== SPM Module ========================
    spm_branch_kernel_sizes = [3, 5, 7]     # Multi-scale feature extraction
    spm_branch_channels = [64, 64, 64]      # Per-branch feature channels
    spm_attention_hidden_dim = 32           # Fusion attention dimension
    spm_in_channels = 3                     # RGB input

    # ======================== SIRM Module ========================
    sirm_iteration_count = 3            # Default iteration number T
    sirm_initial_step_size = 0.1        # AdaGrad initial learning rate eta_0
    sirm_adagrad_epsilon = 1e-8         # Numerical stability constant
    sirm_unet_layers = 4               # Encoder-decoder depth
    sirm_unet_base_channels = 64       # Initial channel count

    # ======================== Loss Function ========================
    lambda1_sparsity = 0.01             # Sparsity regularization weight
    lambda2_smoothness = 0.001          # Smoothness regularization weight
    alpha_perceptual = 0.5              # Perceptual loss weight
    beta_edge = 0.1                     # Edge loss weight
    charbonnier_epsilon = 1e-3          # Charbonnier penalty epsilon

    # ======================== Physics Parameters ========================
    alpha_w_occlusion = 0.5             # Physics-aware step size weight
    beta_w_transmittance = 0.3          # Physics-aware step size weight
    gamma_w_constant = 0.2              # Physics-aware step size constant
    snow_albedo = 0.9                   # rho_s, albedo of snow

    # ======================== Convergence ========================
    convergence_threshold = 1e-4        # delta for early stopping
    max_iterations = 5                  # T_max

    # ======================== Evaluation ========================
    metrics = ["PSNR", "SSIM", "LPIPS", "FID"]

    # ======================== Device ========================
    device = "cuda"
    gpu_id = 0
