"""Configuration for lepinet models loaded through transformers (``trust_remote_code=True``)."""
from transformers import PretrainedConfig


class LepinetConfig(PretrainedConfig):
    """A lepinet classifier: an open_clip-style ViT image tower and a cosine classification head.

    ``id2label`` holds the species names (so ``pipeline("image-classification")`` works as is). The
    taxonomy needed for genus / family is stored alongside it: ``species_to_genus[i]`` is the genus
    index of species ``i``, and ``genus_to_family`` likewise one rank up.
    """

    model_type = "lepinet"

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        head_hidden: int = 1024,
        n_classes: tuple = (12041, 4333, 102),
        temperature: float = 1.0,
        species_to_genus: list | None = None,
        genus_to_family: list | None = None,
        species_keys: list | None = None,
        genus_labels: list | None = None,
        genus_keys: list | None = None,
        family_labels: list | None = None,
        family_keys: list | None = None,
        **kwargs,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.head_hidden = head_hidden
        self.n_classes = list(n_classes)
        self.temperature = temperature
        self.species_to_genus = species_to_genus or []
        self.genus_to_family = genus_to_family or []
        self.species_keys = species_keys or []
        self.genus_labels = genus_labels or []
        self.genus_keys = genus_keys or []
        self.family_labels = family_labels or []
        self.family_keys = family_keys or []
        super().__init__(**kwargs)
