## MLX vainilla FFN VAE implementation

import math
import time
from functools import partial
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

import mlx.optimizers as optim
from mlx.data.datasets import load_mnist
from mlx.utils import tree_flatten


class Encoder(nn.Module):
    def __init__(self, num_latent_dims, hidden_dim, input_shape=784):
        super().__init__()

        layers_dims = [ input_shape, hidden_dim,  hidden_dim//3,  hidden_dim//9, hidden_dim]
        layers = []
        for i in range(len(layers_dims)-1):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i+1]))
            layers.append(nn.LeakyReLU())        

        self.ffn = nn.Sequential(*layers)        
        self.proj_mu = nn.Linear(hidden_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(hidden_dim, num_latent_dims)

    def __call__(self, x):
        x, mu = self.ffn(x), self.proj_mu(x)
        logvar = self.proj_log_var(x)
        sigma = mx.exp(logvar * 0.5)
        eps = mx.random.normal(sigma.shape)
        z = eps * sigma + mu
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()

        layers_dims = [latent_dim, hidden_dim, hidden_dim//3,  hidden_dim//9,  output_dim]
        layers = []
        for i in range(len(layers_dims)-1):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i+1]))
            layers.append(nn.LeakyReLU())
        
        self.ffn = nn.Sequential(*layers)

    def __call__(self, z):
        return mx.sigmoid(self.ffn(z))


class CVAE(nn.Module):
    def __init__(self, num_latent_dims, input_shape):
        super().__init__()
        
        self.num_latent_dims = num_latent_dims
        self.encoder = Encoder(num_latent_dims, input_shape)
        self.decoder = Decoder(num_latent_dims, input_shape, input_shape)

    def __call__(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.decode(z)
        return x, mu, logvar

    def encode(self, x): return self.encoder(x)[0]

    def decode(self, z): return self.decoder(z)

# borrowed from https://github.com/ml-explore/mlx-examples/blob/main/cvae/main.py
def mnist(batch_size, img_size, root=None):
    load_fn = load_mnist
    tr = load_fn(root=root, train=True)
    test = load_fn(root=root, train=False)
    num_img_channels = 1

    def normalize(x): return x.astype("float32") / 255.0

    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(batch_size)
        .prefetch(4, 4)
    )

    test_iter = (
        test.to_stream()
        .image_resize("image", h=img_size[0], w=img_size[1])
        .key_transform("image", normalize)
        .batch(batch_size)
    )
    return tr_iter, test_iter


def loss_fn(model: CVAE, X):
    X_recon, mu, logvar = model(X)
    recon_loss = nn.losses.mse_loss(X_recon, X, reduction="sum")
    kl_div = -0.5 * mx.sum(1 + logvar - mu.square() - logvar.exp())
    return recon_loss + kl_div



if __name__ == "__main__":
    train_iter, test_iter = mnist(batch_size=64, img_size=(28,28)) 
    img_size = (28, 28)

    ZDIM = 20

    model = CVAE(32, math.prod(img_size))
    mx.eval(model.parameters())
    model.train()

    num_params = sum(x.size for _, x in tree_flatten(model.trainable_parameters()))
    print("Number of trainable params: {:0.04f} M".format(num_params / 1e6))

    optimizer = optim.AdamW(learning_rate=0.001)
    train_iter.reset()
    train_batch = next(train_iter)
    test_batch = next(test_iter)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(X):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        return loss

    for e in range(1, 500 + 1):
        train_iter.reset()
        model.train()

        tic = time.perf_counter()
        loss_acc = 0.0
        throughput_acc = 0.0

        for batch_count, batch in enumerate(train_iter):
            X = mx.array(batch["image"]).reshape(-1, math.prod(img_size))
            throughput_tic = time.perf_counter()
            loss = step(X)
            mx.eval(state)

            throughput_toc = time.perf_counter()
            throughput_acc += X.shape[0] / (throughput_toc - throughput_tic)
            loss_acc += loss.item()

            # borrowed from https://github.com/ml-explore/mlx-examples/blob/main/cvae/main.py
            if batch_count > 0 and (batch_count % 10 == 0):
                print(
                    " | ".join(
                        [
                            f"Epoch {e:4d}",
                            f"Loss {(loss_acc / batch_count):10.2f}",
                            f"Throughput {(throughput_acc / batch_count):8.2f} im/s",
                            f"Batch {batch_count:5d}",
                        ]
                    ),
                    end="\r",
                )
