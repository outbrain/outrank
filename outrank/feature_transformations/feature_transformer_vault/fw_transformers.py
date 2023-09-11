from __future__ import annotations

import numpy as np

from outrank.feature_transformations.feature_transformer_vault.default_transformers import (
    DEFAULT_TRANSFORMERS,
)

FW_TRANSFORMERS = DEFAULT_TRANSFORMERS.copy()
resolution_range = [1, 10, 50, 100]
greater_than_range = [1, 2, 4, 8, 16, 32, 64, 96]

for resolution in resolution_range:
    for greater_than in greater_than_range:
        FW_TRANSFORMERS[f'_tr_fw_sqrt_res_{resolution}_gt_{greater_than}'] = (
            f'np.where(X < {greater_than}, '
            f'X, '
            f'np.where(X>{greater_than} ,'
            f'np.round(np.sqrt(X-{greater_than})*{resolution},0), 0))'
        )

        FW_TRANSFORMERS[
            f'_tr_fw_log_res_{resolution}_gt_{greater_than}'
        ] = f'np.where(X <{greater_than}, X, np.where(X >{greater_than}, np.round(np.log(X-{greater_than})*{resolution},0), 0))'

for resolution in resolution_range:
    for greater_than in [np.divide(x, 100) for x in greater_than_range]:
        FW_TRANSFORMERS[
            f'_tr_fw_prob_sqrt_res_{resolution}_gt_{greater_than}'
        ] = f'np.where(X < {greater_than}, X, np.where(X>{greater_than}, np.round(np.sqrt(X-{greater_than})*{resolution},0), 0))'

        FW_TRANSFORMERS[
            f'_tr_fw_prob_log_res_{resolution}_gt_{greater_than}'
        ] = f'np.where(X <{greater_than},X, np.where(X>{greater_than}, np.round(np.log(X-{greater_than})*{resolution},0), 0))'

if __name__ == '__main__':
    print(len(FW_TRANSFORMERS))
