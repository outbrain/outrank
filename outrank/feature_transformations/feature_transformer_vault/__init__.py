from __future__ import annotations

from outrank.feature_transformations.feature_transformer_vault.default_transformers import DEFAULT_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.default_transformers import MINIMAL_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.fw_transformers import (
    FW_TRANSFORMERS,
)

_tr_global_namespace = {
    'default': DEFAULT_TRANSFORMERS,
    'minimal': MINIMAL_TRANSFORMERS,
    'fw-transformers': FW_TRANSFORMERS,
}
