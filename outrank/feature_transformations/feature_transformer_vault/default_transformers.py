# Some boilerplate transformations people tend to use
from __future__ import annotations
MINIMAL_TRANSFORMERS = {
    '_tr_sqrt': 'np.sqrt(X)',
    '_tr_log(x+1)': 'np.log(X + 1)',
    '_tr_sqrt(abs(x))': 'np.sqrt(np.abs(X))',
    '_tr_log(abs(x)+1)': 'np.log(np.abs(X) + 1)',
}

DEFAULT_TRANSFORMERS = {
    '_tr_sqrt': 'np.sqrt(X)',
    '_tr_log(x+1)': 'np.log(X + 1)',
    '_tr_sqrt(abs(x))': 'np.sqrt(np.abs(X))',
    '_tr_log(abs(x)+1)': 'np.log(np.abs(X) + 1)',
    '_tr_div(x,abs(x))*log(abs(x))': 'np.divide(X, np.abs(X)) * np.log(np.abs(X))',
    '_tr_log(x + sqrt(pow(x,2), 1)': 'np.log(X + np.sqrt(np.power(X, 2) + 1))',
    '_tr_log*sqrt': 'np.log(X + 1) * np.sqrt(X)',
    '_tr_log*100': 'np.round(np.log(X + 1) * 100, 0)',
    '_tr_nonzero': 'np.where(X != 0, 1, 0)',
    '_tr_round(div(x,max))': 'np.round(np.divide(X, np.max(X)), 0)',
}

if __name__ == '__main__':
    import numpy as np

    # generate some input (call it X)
    X = np.random.random(100)

    # get some transformer
    some_transformer = DEFAULT_TRANSFORMERS.get('_tr_nonzero')

    if some_transformer is None:
        some_transformer = ''

    # evaluate to get output
    output = eval(some_transformer)

    # check output somehow
    print(output)
