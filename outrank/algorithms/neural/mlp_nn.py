from __future__ import annotations

import logging

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax import random

logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)

key = random.PRNGKey(7235123)


class GenericJaxNN(nn.Module):
    num_features: int
    architecture: jnp.array

    @nn.compact
    def __call__(self, x):
        for num_units in self.architecture:
            x = nn.Dense(features=num_units)(x)
            x = nn.relu(x)
            x = nn.Dense(features=2)(x)
        return x


class NNClassifier:

    def __init__(self, learning_rate=0.001, architecture=[48, 48], epochs=100):
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.num_epochs = epochs
        self.ncl = None

        batch_size = 10
        features = 5  # Number of input features to the model

        # Pass the features and architecture parameters to the model constructor
        self.mlp_model = GenericJaxNN(
            num_features=features,
            architecture=jnp.array(architecture),
        )

    def fit(self, X, Y, print_loss=True):

        loss_grad_fn = jax.value_and_grad(self.forward_loss)
        self.ncl = len(jnp.unique(X))
        X = jax.nn.one_hot(X, num_classes=self.ncl).reshape(X.shape[0], -1)
        sample_batch = jnp.ones((1, X.shape[1]))
        self.init_internal_mlp(sample_batch)

        for i in range(self.num_epochs):
            loss_val, grads = loss_grad_fn(self.parameters, X, Y)
            updates, self.opt_state = self.tx.update(grads, self.opt_state)
            self.parameters = optax.apply_updates(self.parameters, updates)
            if print_loss:
                print(f'Loss step {i + 1}: ', loss_val.item())

    def forward_loss(self, parameters, X, Y):

        def get_logits(x):
            pred_logits = self.mlp_model.apply(parameters, x)
            return pred_logits

        batch_logits = jax.vmap(get_logits)(X)
        one_hot_labels = jax.nn.one_hot(Y, num_classes=2)
        loss = optax.softmax_cross_entropy(
            logits=batch_logits,
            labels=one_hot_labels,
        ).mean()

        return loss

    def forward_pass(self, X):

        def get_logits(x):
            pred_logits = self.mlp_model.apply(self.parameters, x)
            return jax.nn.softmax(pred_logits)

        batch_logits = jax.vmap(get_logits)(X)
        return batch_logits

    def init_internal_mlp(self, sample_batch):

        self.parameters = self.mlp_model.init(key, sample_batch)
        self.tx = optax.adam(learning_rate=self.learning_rate)
        self.opt_state = self.tx.init(self.parameters)

    def selftest(self):
        random_data = random.randint(
            minval=0,
            maxval=200,
            key=key,
            shape=(10, 1),
        )
        ncl = len(jnp.unique(random_data))
        random_data = jax.nn.one_hot(random_data, num_classes=ncl).reshape(
            random_data.shape[0], -1,
        )
        sample_batch = jnp.ones((1, random_data.shape[1]))
        self.init_internal_mlp(sample_batch)

        is_class1 = jnp.sum(random_data, axis=1) < 500
        random_labels = (is_class1 + 0).astype(jnp.int32)
        real_output = self.forward_pass(random_data)
        assert jnp.sum(jnp.sum(real_output, axis=1)) == real_output.shape[0]

        forward_loss = self.forward_loss(
            self.parameters, random_data,
            random_labels,
        )
        print('Self-test loss:', forward_loss.item())
        assert forward_loss.item() > 0.6
        self.fit(random_data, random_labels)

        preds = self.predict(random_data)
        assert len(preds) == 10
        assert preds[0, 0] < 0.1

    def predict(self, X):

        if self.ncl is not None:
            X_ohe = jax.nn.one_hot(X, num_classes=self.ncl).reshape(
                X.shape[0], -1,
            )
            return self.forward_pass(X_ohe)
        else:
            logger.error('number of classes unknown (NNClassifier)!')


if __name__ == '__main__':
    clf = NNClassifier(learning_rate=0.01, architecture=[48, 48], epochs=10)
    clf.selftest()
