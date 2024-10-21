from __future__ import annotations

from outrank.feature_transformations.feature_transformer_vault.default_transformers import DEFAULT_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.default_transformers import EXTENDED_ROUNDED_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.default_transformers import EXTENDED_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.default_transformers import MINIMAL_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.default_transformers import VERBOSE_TRANSFORMERS
from outrank.feature_transformations.feature_transformer_vault.fw_transformers import \
    FW_TRANSFORMERS

_tr_global_namespace = {
    'default': DEFAULT_TRANSFORMERS,
    'minimal': MINIMAL_TRANSFORMERS,
    'fw-transformers': FW_TRANSFORMERS,
    'extended': EXTENDED_TRANSFORMERS,
    'verbose': VERBOSE_TRANSFORMERS,
    'extended_rounded': EXTENDED_ROUNDED_TRANSFORMERS,
}
