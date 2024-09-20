from __future__ import annotations

import numpy as np
from outrank.feature_transformations.feature_transformer_vault.default_transformers import EXTENDED_TRANSFORMERS

COMPREHENSIVE_TRANSFORMERS = EXTENDED_TRANSFORMERS.copy()

powers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
shifts = [-1, 0, 1, 2, 5, 10]
scales = [0.5, 1, 2, 5, 10, 50]
Min_values = ["-np.inf", -10, -1, 0, 1, 10, 100, 1000]
Max_values = [1, 10, 100, 1000, 10000, "np.inf"]
param_a = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 500]

for power in powers:
    for shift in shifts:
        transformer_name = f'_tr_logistic_shift_{shift}_power_{power}'
        transformer_expression = f'1 / (1 + np.exp(-{power}*X - {shift}))'
        COMPREHENSIVE_TRANSFORMERS[transformer_name] = transformer_expression

for scale in scales:
    for shift in shifts:
        transformer_name = f'_tr_tanh_scale_{scale}_shift_{shift}'
        transformer_expression = f'{scale} * np.tanh(X + {shift})'
        COMPREHENSIVE_TRANSFORMERS[transformer_name] = transformer_expression

for shift in shifts:
    for scale in scales:
        for Min in Min_values:
            for Max in Max_values:
                transformer_name = f'_tr_log_clip_C_{shift}_N_{scale}_Min_{Min}_Max_{Max}'
                transformer_expression = f'np.clip(np.log(X + {shift}) * {scale}, {Min}, {Max})'
                COMPREHENSIVE_TRANSFORMERS[transformer_name] = transformer_expression

for a in scales:
    for power in powers:
        for Min in Min_values:
            for Max in Max_values:
                transformer_name = f'_tr_alpha_pow_clip_alpha_{a}_P_{power}_Min_{Min}_Max_{Max}'
                transformer_expression = f'np.clip({a} * np.power(X, {power}), {Min}, {Max})'
                COMPREHENSIVE_TRANSFORMERS[transformer_name] = transformer_expression

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_sqrt_plus_{a}': f'np.sqrt(X + {a})',
        f'_tr_sqrt_power_{a}': f'np.sqrt(np.power(X, {a}))',
        f'_tr_sqrt_log1p_{a}': f'np.sqrt(np.log1p(X * {a}))',
        f'_tr_sqrt_div_{a}': f'np.sqrt(X) / {a}'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_log_abs_plus_{a}': f'np.log(np.abs(X) + {a})',
        f'_tr_log_abs_times_{a}': f'np.log(np.abs(X)) * {a}',
        f'_tr_log_abs_squared_{a}': f'np.log(np.abs(X) ** 2 + {a})',
        f'_tr_log_abs_sqrt_{a}': f'np.log(np.abs(X)) + np.sqrt(np.abs(X) + {a})',
        f'_tr_log_abs_cbrt_{a}': f'np.log(np.abs(X) + np.cbrt(X * {a}))'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_div_x_absx_plus1_log_absx_{a}': f'np.divide(X, np.abs(X) + {a}) * np.log(np.abs(X))',
        f'_tr_div_x_absx_times_log_absx_{a}': f'np.divide(X, np.abs(X)) * np.log(np.abs(X)) * {a}',
        f'_tr_div_x_absx_log_absx_plus_sin_{a}': f'np.divide(X, np.abs(X)) * np.log(np.abs(X)) + np.sin(X * {a})',
        f'_tr_div_x_absx_log_absx_squared_{a}': f'np.divide(X, np.abs(X)) * np.log(np.abs(X)) ** {a}',
        f'_tr_div_x_absx_log_absx_cbrt_{a}': f'np.divide(X, np.abs(X)) * np.log(np.abs(X)) + np.cbrt(X * {a})'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_log_x_plus_sqrt_x2_plus{a}': f'np.log(X + np.sqrt(np.power(X, 2) + {a}))',
        f'_tr_log_x_plus_sqrt_x2_times{a}': f'np.log(X + np.sqrt(np.power(X, 2))) * {a}',
        f'_tr_log_x_plus_sqrt_x2_plus_logx_{a}': f'np.log(X + np.sqrt(np.power(X, 2) + {a})) + np.log(X + {a})',
        f'_tr_log_x_plus_sqrt_x2_plus_absx_{a}': f'np.log(X + np.sqrt(np.power(X, 2) + {a})) + np.abs(X)',
        f'_tr_log_x_plus_sqrt_x2_times{a}_plus2': f'np.log(X + np.sqrt(np.power(X, 2) + {a})) * {a}'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_log_plus{a}_times_sqrt_x': f'np.log(X + {a}) * np.sqrt(X)',
        f'_tr_log_times_sqrt_plus{a}': f'np.log(X) * (np.sqrt(X) + {a})',
        f'_tr_log_squared_times_sqrt_{a}': f'np.log(X + {a}) ** 2 * np.sqrt(X)',
        f'_tr_log_times_sqrt_x_plus{a}': f'np.log(X + {a}) * (np.sqrt(X) + {a})',
        f'_tr_log_times_sqrt_x_div{a}': f'np.log(X + {a}) * np.sqrt(X) / {a}'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_nonzero_plus{a}': f'np.where(X != 0, 1, 0) + {a}',
        f'_tr_nonzero_times{a}': f'np.where(X != 0, 1, 0) * {a}',
    })

stat_funcs = ['min', 'mean', 'median', 'std', 'sum']
for func in stat_funcs:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_round_div_{func}': f'np.round(np.divide(X, np.{func}(X)), 0)'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_squared_plus{a}': f'np.square(X) + {a}',
        f'_tr_squared_times{a}': f'np.square(X) * {a}',
        f'_tr_squared_log': f'np.square(X) * np.log(X + {a})',
        f'_tr_squared_cbrt': f'np.square(X) + np.cbrt(X) * {a}'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_cubed_plus{a}': f'np.power(X, 3) + {a}',
        f'_tr_cubed_times{a}': f'np.power(X, 3) * {a}',
        f'_tr_cubed_log': f'np.power(X, 3) * np.log(X + {a})',
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_inverse_plus{a}': f'np.divide(1, X + {a}, out=np.zeros_like(X), where=(X+{a})!=0)',
        f'_tr_inverse_times{a}': f'{a} * np.divide(1, X, out=np.zeros_like(X), where=X!=0)',
        f'_tr_inverse_log': f'np.divide(1, X, out=np.zeros_like(X), where=X!=0) * np.log(X + {a})'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_exp_plus{a}': f'np.exp(X) + {a}',
        f'_tr_exp_minus{a}': f'np.exp(X) - {a}',
        f'_tr_exp_times{a}': f'np.exp(X) * {a}',
        f'_tr_exp_log_plus{a}': f'np.exp(X) * np.log(X + {a})',
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_expm1_times{a}': f'np.expm1(X) * {a}',
        f'_tr_expm1_squared': 'np.expm1(X) ** 2',
        f'_tr_expm1_log_plus{a}': f'np.expm1(X) * np.log(X + {a})',
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_sin_plus{a}': f'np.sin(X) + {a}',
        f'_tr_sin_times{a}': f'np.sin(X) * {a}',
        f'_tr_sin_squared': 'np.sin(X) ** 2',
        f'_tr_sin_log_plus{a}': f'np.sin(X) * np.log(X + {a})',
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_cos_plus{a}': f'np.cos(X) + {a}',
        f'_tr_cos_times{a}': f'np.cos(X) * {a}',
        f'_tr_cos_squared': 'np.cos(X) ** 2',
        f'_tr_cos_log_plus{a}': f'np.cos(X) * np.log(X + {a})',
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_tan_plus{a}': f'np.tan(X) + {a}',
        f'_tr_tan_times{a}': f'np.tan(X) * {a}',
        f'_tr_tan_squared': 'np.tan(X) ** 2',
        f'_tr_tan_log_plus{a}': f'np.tan(X) * np.log(X + {a})',
        f'_tr_tan_sin': 'np.tan(X) + np.sin(X)'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_arcsin_safe_plus{a}': f'np.arcsin(np.clip(X, -1, 1)) + {a}',
        f'_tr_arcsin_safe_times{a}': f'np.arcsin(np.clip(X, -1, 1)) * {a}',
        f'_tr_arcsin_safe_squared': 'np.arcsin(np.clip(X, -1, 1)) ** 2',
        f'_tr_arcsin_safe_log_plus{a}': f'np.arcsin(np.clip(X, -1, 1)) * np.log(X + {a})',
        f'_tr_arcsin_safe_sqrt': 'np.sqrt(np.arcsin(np.clip(X, -1, 1)))'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_arccos_safe_plus{a}': f'np.arccos(np.clip(X, -1, 1)) + {a}',
        f'_tr_arccos_safe_times{a}': f'np.arccos(np.clip(X, -1, 1)) * {a}',
        f'_tr_arccos_safe_squared': 'np.arccos(np.clip(X, -1, 1)) ** 2',
        f'_tr_arccos_safe_log_plus{a}': f'np.arccos(np.clip(X, -1, 1)) * np.log(X + {a})',
        f'_tr_arccos_safe_sqrt': 'np.sqrt(np.arccos(np.clip(X, -1, 1)))'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_arctan_plus{a}': f'np.arctan(X) + {a}',
        f'_tr_arctan_times{a}': f'np.arctan(X) * {a}',
        f'_tr_arctan_squared': 'np.arctan(X) ** 2',
        f'_tr_arctan_log_plus{a}': f'np.arctan(X) * np.log(X + {a})',
        f'_tr_arctan_sqrt': 'np.sqrt(np.arctan(X))'
    })

z_score_funcs = ['min', 'median', 'mean']
for func in z_score_funcs:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_z_score_{func}_minmax': f'(X - np.{func}(X)) / np.std(X)',
        f'_tr_z_score_{func}_scaled': f'(X - np.{func}(X)) / (np.std(X) + 1)',
        f'_tr_z_score_{func}_log': f'(X - np.{func}(X)) / np.std(X) * np.log(X + 1)',
        f'_tr_z_score_{func}_squared': f'((X - np.{func}(X)) / np.std(X)) ** 2'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_min_max_scale_plus{a}': f'(X - np.min(X)) / (np.max(X) - np.min(X)) + {a}',
        f'_tr_min_max_scale_times{a}': f'(X - np.min(X)) / (np.max(X) - np.min(X)) * {a}',
        f'_tr_min_max_scale_log_plus{a}': f'(X - np.min(X)) / (np.max(X) - np.min(X)) * np.log(X + {a})',
        f'_tr_min_max_scale_squared': '((X - np.min(X)) / (np.max(X) - np.min(X))) ** 2',
        f'_tr_min_max_scale_sqrt': 'np.sqrt((X - np.min(X)) / (np.max(X) - np.min(X)))'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_quadratic_root_plus{a}': f'np.cbrt(X) + {a}',
        f'_tr_quadratic_root_times{a}': f'np.cbrt(X) * {a}',
        f'_tr_quadratic_root_squared': 'np.cbrt(X) ** 2',
        f'_tr_quadratic_root_log_plus{a}': f'np.cbrt(X) * np.log(X + {a})',
        f'_tr_quadratic_root_sqrt': 'np.sqrt(np.cbrt(X))'
    })

for a in param_a:
    COMPREHENSIVE_TRANSFORMERS.update({
        f'_tr_shifted_sigmoid_plus{a}': f'1 / (1 + np.exp(-X - 0.5)) + {a}',
        f'_tr_shifted_sigmoid_times{a}': f'{a} / (1 + np.exp(-X - 0.5))',
        f'_tr_shifted_sigmoid_scaled': '1 / (1 + np.exp(-X - 0.5)) * 0.5',
        f'_tr_shifted_sigmoid_shift': '1 / (1 + np.exp(-(X - 1) - 0.5))',
        f'_tr_shifted_sigmoid_squared': '(1 / (1 + np.exp(-X - 0.5))) ** 2'
    })

if __name__ == '__main__':
    print(len(COMPREHENSIVE_TRANSFORMERS))
