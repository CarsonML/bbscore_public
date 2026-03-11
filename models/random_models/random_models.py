import torch
import torch.nn as nn
import math

# Relative imports from sibling directories based on your provided ls output
# and file contents.
from ..scaling_models.scaling_model import ConvnextLargeImagenetFullSeed0Loader
from ..resnext101wsl.resnext_wsl import ResNeXtWSL
from ..s3d_text_video.s3d import S3DHowTo100M
from ..videomae.videomae import VideoMAE
# --- Setup ---

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


# --- Randomization Helper ---

def reinitialize_model(model):
    """
    Applies Kaiming initialization to all applicable layers in a PyTorch module,
    resetting weights to random values.
    """
    def kaiming_init(module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            # For Conv2d and Conv3d layers
            nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            # For Linear (fully connected) layers
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

        elif isinstance(module, (nn.LayerNorm, nn.Embedding, nn.BatchNorm2d, nn.BatchNorm3d)):
            # reset LayerNorm and Embedding layers
            module.reset_parameters()

    # Check if model is compiled (OptimizedModule) and access original module if so
    if hasattr(model, "_orig_mod"):
        model._orig_mod.apply(kaiming_init)
    else:
        model.apply(kaiming_init)


# --- Random Model Loaders ---

class RandomConvnextLargeImagenetFullSeed0Loader(ConvnextLargeImagenetFullSeed0Loader):
    """
    Randomized version of ConvNeXt Large (ImageNet Full).
    """

    def get_model(self, identifier=None):
        # Load the standard architecture.
        # Note: BaseModelLoader finds configs.json relative to the base file,
        # so this works even from a subclass in a different folder.
        model = super().get_model(identifier)

        print(f"Randomizing weights for {self.MODEL_ID}...")
        reinitialize_model(model)

        return model


class RandomResnext101_32x8d_wsl(ResNeXtWSL):
    """
    Randomized version of ResNeXt-101 32x8d WSL.
    """

    def get_model(self, identifier=None):
        # We must pass the original identifier expected by ResNeXtWSL
        original_id = "resnext101_32x8d_wsl"

        # Load model architecture
        model = super().get_model(original_id)

        print(f"Randomizing weights for resnext101_32x8d_wsl...")
        reinitialize_model(model)

        return model


class RandomS3DHowTo100M(S3DHowTo100M):
    """
    Randomized version of S3D (HowTo100M architecture).
    """

    def get_model(self, identifier=None):
        original_id = "s3d-HowTo100M"

        model = super().get_model(original_id)

        print(f"Randomizing weights for s3d-HowTo100M...")
        reinitialize_model(model)

        return model


class RandomVideoMAEV1L(VideoMAE):
    """
    Randomized version of VideoMAE V1 Large.
    """

    def get_model(self, identifier=None):
        # Map to the specific key expected by VideoMAE class ("VIDEOMAE-LARGE")
        original_id = "VIDEOMAE-LARGE"

        model = super().get_model(original_id)

        print(f"Randomizing weights for VIDEOMAE-LARGE...")
        reinitialize_model(model)

        return model


