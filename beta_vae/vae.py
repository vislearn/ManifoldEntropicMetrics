import torch
import torch.nn as nn

from typing import Union, Callable
from typing import Iterable, Tuple, List

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

# Adapted from https://github.com/ludovicobuizza/gan_vae

def make_module(conv_layer,
                 hyper_params,
                 activation=nn.ReLU):
    modules = []
    in_channels = hyper_params["in_channels"]
    conds = hyper_params['conds']
    for i in range(len(hyper_params["hidden_channels"])):
        module = nn.Sequential(
            conv_layer(
                in_channels=in_channels+conds,
                out_channels=hyper_params["hidden_channels"][i],
                kernel_size=hyper_params["kernels"][i],
                stride=hyper_params["strides"][i],
                padding=hyper_params["paddings"][i]
            ),
            nn.BatchNorm2d(num_features=hyper_params["hidden_channels"][i]),
            activation(),
        )
        in_channels = hyper_params["hidden_channels"][i]
        modules.append(module)
    return nn.Sequential(*modules)


def make_final_decoder_layer(decoder_hyper_params):
    conds = decoder_hyper_params['conds']
    final_layer = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=decoder_hyper_params["hidden_channels"][-1]+conds,
            out_channels=decoder_hyper_params["out_channels"],
            kernel_size=decoder_hyper_params["final_kernel"],
            stride=decoder_hyper_params["final_stride"],
            padding=decoder_hyper_params["final_padding"],
            output_padding=decoder_hyper_params["final_output_padding"],
        ),
        nn.BatchNorm2d(num_features=decoder_hyper_params["out_channels"]),
        nn.LeakyReLU(),
        nn.Conv2d(
            in_channels=decoder_hyper_params["out_channels"],
            out_channels=decoder_hyper_params["out_channels"],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Sigmoid(),
    )
    return final_layer

class VAEFlow(nn.Module): #rewrite forward pass
    def __init__(self, vae):
        super(VAEFlow, self).__init__()
        self.vae = vae

    def forward(self, x_or_z, c=None, rev=False):
        if rev:
            return self.vae.decode(x_or_z, c=c), torch.zeros(x_or_z.shape[0]).to(x_or_z.device)
        else:
            mu, _ = self.vae.encode(x_or_z, c=c)
            return mu, torch.zeros(x_or_z.shape[0]).to(x_or_z.device)

class VAE(nn.Module):
    def __init__(self, encoder_hyper_params=None, decoder_hyper_params=None, encoder=None, decoder=None, latent_dim=None, conditional=False, conds=0):
        """Variational Autoencoder (VAE) model

        Args:
            encoder_hyper_params (dict): Dictionary of hyperparams for encoder.
                It should contain the following keys:
                - latent_dim (int): Dimensionality of the latent space.
                - hidden_channels (list): List of hidden dimensions.
                - kernels (list): List of kernel sizes.
                - strides (list): List of strides.
                - paddings (list): List of paddings.
                - in_channels (int): Number of input channels.
                - fc_neurons (int): Number of neurons in the fully connected
                  layer.
            decoder_hyper_params (dict): Dictionary of hyperparams for decoder.
                It should contain the following keys:
                - in_channels (int): Number of input channels.
                - hidden_channels (list): List of hidden channels.
                - kernels (list): List of kernel sizes.
                - strides (list): List of strides.
                - paddings (list): List of paddings.
                - out_channels (int): Number of output channels.
                - final_kernel (int): Kernel size of the final layer.
        """
        super(VAE, self).__init__()
        if encoder_hyper_params is not None and decoder_hyper_params is not None:
            self.mode = "default"
            self.latent_dim = encoder_hyper_params["latent_dim"]
            self.encoder_hyper_params = encoder_hyper_params
            self.decoder_hyper_params = decoder_hyper_params

            if not conditional:
                conds = 0
            self.encoder_hyper_params['conds'] = conds
            self.decoder_hyper_params['conds'] = conds

            self.conditional = conditional
            self.encoder = make_module(
                conv_layer=nn.Conv2d,
                hyper_params=encoder_hyper_params,
                activation=nn.LeakyReLU,
            )
            self.fc_mu = nn.Linear(encoder_hyper_params["fc_neurons"], self.latent_dim)
            self.fc_var = nn.Linear(encoder_hyper_params["fc_neurons"], self.latent_dim)

            last_dim = encoder_hyper_params["hidden_channels"][-1]
            self.decoder_input = nn.Linear(self.latent_dim+conds, last_dim * 4)
            self.decoder = make_module(
                conv_layer=nn.ConvTranspose2d,
                hyper_params=decoder_hyper_params,
                activation=nn.LeakyReLU,
            )
            self.final_layer = make_final_decoder_layer(decoder_hyper_params)
        else:
            self.mode = "custom"
            self.latent_dim = latent_dim

            if not conditional:
                conds = 0
            self.conditional = conditional

            self.encoder = encoder
            self.decoder = decoder
            

    def encode(self, x, c=None):
        if c is not None and type(c) == list:
            c = c[0]
        if self.mode == "normal":
            #result = self.encoder(x)
            for layer in self.encoder:
                if self.conditional:
                    #concatenate c to z along the channel dimension
                    c_temp = c[..., None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
                    x = torch.cat((x, c_temp), dim=1)
                x = layer(x)
            result = x
            result = torch.flatten(result, start_dim=1)
        else:
            mu = self.encoder(x, c=c).embedding
            log_var = self.encoder(x, c=c).log_var
        return mu, log_var

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, c=None):
        if c is not None and type(c) == list:
            c = c[0]
        if self.mode == "normal":
            #check if c is a list
            if self.conditional:
                z = torch.cat((z, c), dim=1)
            z = self.decoder_input(z)
            z = z.view(-1, int(z.shape[1] / 4), 2, 2)
            for layer in self.decoder:
                if self.conditional:
                    #concatenate c to z along the channel dimension
                    if z.dim() == 4:
                        c_temp = c[..., None, None].expand(-1, -1, z.shape[-2], z.shape[-1])
                    else:
                        c_temp = c
                    z = torch.cat((z, c_temp), dim=1)
                z = layer(z)
            #result = self.decoder(result)
            if self.conditional:
                c_temp = c[..., None, None].expand(-1, -1, z.shape[-2], z.shape[-1])
                z = torch.cat((z, c_temp), dim=1)
            result = self.final_layer(z)
        else:
            result = self.decoder(z, c=c).reconstruction
        return result

    def forward(self, x, c=None):
        mu, log_var = self.encode(x, c=c)
        z = self.reparametrize(mu, log_var)
        return self.decode(z, c=c), x, mu, log_var
    
# Adapted from pythae models
class Encoder_AE_MNIST(BaseEncoder):
    def __init__(self, input_dim, latent_dim, conds=0, ch_mult=1, use_resblocks=True):
        BaseEncoder.__init__(self)

        #conds gives number of classes

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_channels = input_dim[0]
        self.conds = conds
        self.ch_mult = ch_mult
        self.use_resblocks = use_resblocks

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels + self.conds, 32*ch_mult, 4, 2, padding=1),
                nn.BatchNorm2d(32*ch_mult),
                nn.ReLU(),
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.Conv2d(32*ch_mult + self.conds, 64*ch_mult, 4, 2, padding=1), nn.BatchNorm2d(64*ch_mult), nn.ReLU()
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.Conv2d(64*ch_mult + self.conds, 128*ch_mult, 4, 2, padding=1), nn.BatchNorm2d(128*ch_mult), nn.ReLU()
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.Conv2d(128*ch_mult + self.conds, 256*ch_mult, 4, 2, padding=1), nn.BatchNorm2d(256*ch_mult), nn.ReLU()
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Linear(256*ch_mult + self.conds, self.latent_dim)
        )
        #nn.init.zeros_(layers[-1].weight)
        #nn.init.ones_(layers[-1].bias)
        if use_resblocks:
            for _ in range(4):
                temp = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, self.latent_dim)
                )
                nn.init.zeros_(temp[-1].weight)
                nn.init.ones_(temp[-1].bias)
                # ResBlock(args.latent_dim, 512)
                layers.append(SkipConnection(temp))
        #else:
        #    layers.append(nn.BatchNorm1d(self.latent_dim))

        self.layers = layers
        self.depth = len(layers)

        #identity
        self.embedding = nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        #x_step = 1/255
        #x = torch.special.logit(x*(1-2*x_step) + x_step)  # Avoid nans

        out = x

        for i in range(max_depth):
            #if linear_layer we need to reshape the input
            if i == 4:
                out = out.reshape(x.shape[0], -1)
            if self.conds > 0:
                assert c is not None, "Conditional input is required"
                if out.dim() == 4:
                    cond = c.view(c.shape[0], c.shape[1], 1, 1)
                    cond = cond.repeat(1, 1, out.shape[2], out.shape[3])
                else:
                    cond = c
                out = torch.cat([out, cond], dim=1)
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output
    
class Decoder_AE_MNIST(BaseDecoder):
    def __init__(self, input_dim, latent_dim, conds=0, ch_mult=1, use_resblocks=True):
        BaseDecoder.__init__(self)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_channels = input_dim[0]
        self.conds = conds
        self.ch_mult = ch_mult
        self.use_resblocks = use_resblocks

        layers = nn.ModuleList()
        if use_resblocks:
            for _ in range(4):
                temp = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim + self.conds, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, self.latent_dim + conds)
                )
                # ResBlock(args.latent_dim, 512)
                layers.append(SkipConnection(temp))

        layers.append(nn.Linear(self.latent_dim + self.conds, 4096*ch_mult))
        #nn.init.zeros_(layers[-1].weight)
        #nn.init.ones_(layers[-1].bias)

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256*ch_mult + self.conds, 128*ch_mult, 3, 2, padding=1),
                nn.BatchNorm2d(128*ch_mult),
                nn.ReLU(),
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128*ch_mult + self.conds, 64*ch_mult, 3, 2, padding=1, output_padding=1),
                nn.BatchNorm2d(64*ch_mult),
                nn.ReLU(),
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    64*ch_mult + self.conds, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='sigmoid')

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor = None, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            if i == 1+4*self.use_resblocks:
                out = out.reshape(z.shape[0], 256*self.ch_mult, 4, 4)
            if self.conds > 0:
                assert c is not None, "Conditional input is required"
                if out.dim() == 4:
                    cond = c.view(c.shape[0], c.shape[1], 1, 1)
                    cond = cond.repeat(1, 1, out.shape[2], out.shape[3])
                else:
                    cond = c
                out = torch.cat([out, cond], dim=1)
            #print(out.shape, i, self.layers[i])
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output

class Encoder_VAE_MNIST(BaseEncoder):
    def __init__(self, input_dim, latent_dim, conds=0, ch_mult=1, use_resblocks=True):
        BaseEncoder.__init__(self)

        #conds gives number of classes

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_channels = input_dim[0]
        self.conds = conds
        self.ch_mult = ch_mult
        self.use_resblocks = use_resblocks

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels + self.conds, 32*ch_mult, 4, 2, padding=1),
                nn.BatchNorm2d(32*ch_mult),
                nn.ReLU(),
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.Conv2d(32*ch_mult + self.conds, 64*ch_mult, 4, 2, padding=1), nn.BatchNorm2d(64*ch_mult), nn.ReLU()
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.Conv2d(64*ch_mult + self.conds, 128*ch_mult, 4, 2, padding=1), nn.BatchNorm2d(128*ch_mult), nn.ReLU()
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        layers.append(
            nn.Sequential(
                nn.Conv2d(128*ch_mult + self.conds, 256*ch_mult, 4, 2, padding=1), nn.BatchNorm2d(256*ch_mult), nn.ReLU()
            )
        )
        #torch.nn.init.kaiming_uniform_(layers[-1][0].weight, nonlinearity='relu')

        #nn.init.zeros_(layers[-1].weight)
        #nn.init.ones_(layers[-1].bias)
        if use_resblocks:
            for _ in range(4):
                temp = torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, 512), torch.nn.SiLU(),
                    torch.nn.Linear(512, self.latent_dim)
                )
                nn.init.zeros_(temp[-1].weight)
                nn.init.ones_(temp[-1].bias)
                # ResBlock(args.latent_dim, 512)
                layers.append(SkipConnection(temp))
        #else:
        #    layers.append(nn.BatchNorm1d(self.latent_dim))

        self.layers = layers
        self.depth = len(layers) + 1

        self.embedding = nn.Linear(256*ch_mult + self.conds, self.latent_dim)
        self.log_var = nn.Linear(256*ch_mult + self.conds, self.latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        #x_step = 1/255
        #x = torch.special.logit(x*(1-2*x_step) + x_step)  # Avoid nans

        out = x

        for i in range(max_depth):
            #if linear_layer we need to reshape the input
            if i == 4:
                out = out.reshape(x.shape[0], -1)
            if self.conds > 0:
                assert c is not None, "Conditional input is required"
                if out.dim() == 4:
                    cond = c.view(c.shape[0], c.shape[1], 1, 1)
                    cond = cond.repeat(1, 1, out.shape[2], out.shape[3])
                else:
                    cond = c
                out = torch.cat([out, cond], dim=1)
            if i < self.depth - 1:
                out = self.layers[i](out)
            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_var"] = self.log_var(out.reshape(x.shape[0], -1))

        return output