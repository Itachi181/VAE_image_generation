from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from six.moves import xrange
import torchvision
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_csv(csv_filename):
    if not os.path.exists(csv_filename):
        print(f"Creating CSV file: {csv_filename}")
        with open(csv_filename, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "recon_error", "perplexity", "total_loss"])
            writer.writeheader()
        print(f"Headers written to {csv_filename}")
    else:
        print(f"CSV file {csv_filename} already exists.")

def prepare_data_loaders(image_size, batch_size, train_data_path, seed=42):
    """
    Prepares data loaders for training and validation with specified image transformations.

    Parameters:
        image_size (int): Size to which images will be resized (image_size x image_size).
        batch_size (int): Batch size for the training data loader.
        train_data_path (str): Path to the folder containing training images.
        seed (int): Seed for reproducibility.

    Returns:
        tuple: A tuple containing (training_loader, validation_loader).
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Define image transformations: resize, tensor conversion, and normalization
    print(">> Setting up transformations >>")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    ])

    # Load training and validation datasets
    print(">> Loading training data >>")
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    
    print(">> Loading validation data >>")
    validation_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)

    # Create data loaders
    print(">> Setting up training and validation loaders >>")
    training_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

    print(">> Data loaders are ready >>")
    return training_loader, validation_loader


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        ### Create an embedding matrix with size number of embedding X embedding dimension
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances between flattened input and embedding vector
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
            
        # Choose indices that are min in each row
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        ## Create a matrix of dimensions B*H*W into number of embeddings
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        ### Convert index to on hot encoding 
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


### Create Residual connections
class Residual(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_hiddens):
        super(Residual,self).__init__()
        self._block=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                     out_channels=num_residual_hiddens,
                     kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                     out_channels=num_hiddens,
                     kernel_size=1,stride=1,bias=False)
        )
        
    def forward(self,x):
        return x + self._block(x)
        
class ResidualStack(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(ResidualStack,self).__init__()
        self._num_residual_layers=num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels,num_hiddens,num_residual_hiddens) for _ in range(self._num_residual_layers)])
    def forward(self,x):
        for i in range(self._num_residual_layers):
            x=self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Encoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2,padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels = num_hiddens,
                                 kernel_size=4,
                                 stride=2,padding=1
                                )
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                out_channels=num_hiddens,
                                kernel_size=3,
                                stride=1,padding=1)
        self._residual_stack = ResidualStack(in_channels = num_hiddens,
                                             num_hiddens = num_hiddens,
                                             num_residual_layers = num_residual_layers,
                                             num_residual_hiddens = num_residual_hiddens
                                            )
    def forward(self,inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(Decoder,self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels= num_hiddens,
                                kernel_size=3,
                                stride=1,padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens= num_residual_hiddens
                                            )
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                               out_channels=num_hiddens//2,
                                               kernel_size=4,
                                               stride=2,padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                               out_channels=3,
                                               kernel_size=4,
                                               stride=2,padding=1)
    def forward(self,inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)


class Model(nn.Module):
    def __init__(self,num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0):
        super(Model,self).__init__()
        self._encoder_= Encoder(3,num_hiddens,num_residual_layers,num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings,embedding_dim,commitment_cost)
        self._decoder = Decoder(embedding_dim,
                              num_hiddens,
                              num_residual_layers,
                              num_residual_hiddens)
    def forward(self,x):
        z = self._encoder_(x)
        z = self._pre_vq_conv(z)
        loss,quantized,perplexity,_ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss,x_recon,perplexity

def train_model(training_loader, num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0):

    print(">"*90)
    model = Model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    print("-"*90)
    model.train()
    print("+"*90)
    
    train_res_recon_error = []
    train_res_perplexity = []
    
    losses = []
    epoch_data = []
    
    for i in xrange(num_training_updates):
        # print("*"*90)
        print("Epoch >>> ", i)
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()
    
        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()
    
        optimizer.step()

        # Collect epoch data
        epoch_loss_data = {
            "epoch": i + 1,
            "recon_error": recon_error.item(),
            "perplexity": perplexity.item(),
            "total_loss": loss.item(),
        }
        epoch_data.append(epoch_loss_data)
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
    
        if (i+1) % 100 == 0:
            print(f"Saving data for epoch {i + 1} to CSV...")
            with open(csv_filename, "a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["epoch", "recon_error", "perplexity", "total_loss"])
                writer.writerows([
                    {
                        "epoch": entry["epoch"],
                        "recon_error": entry["recon_error"],
                        "perplexity": entry["perplexity"],
                        "total_loss": entry["total_loss"]
                    }
                    for entry in epoch_data[-100:]  # Write the last 100 epochs
                ])

            print(f"Data saved up to epoch {i + 1}")
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
    
        if (i+1) % 5000 == 0 :
            torch.save(model.state_dict(), f'vae_cat_trained_model_50k/Advance_model_{i+1}.pth')
            
    torch.save(model.state_dict(), 'vae_cat_trained_model_50k/Advance_model.pth')

if __name__ == "__main__":

    num_training_updates = 50000
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim= 512
    num_embeddings = 512
    commitment_cost = 0.25
    learning_rate = 1e-3

    csv_filename = "epoch_losses_3.csv"  # CSV file to store summary
    initialize_csv(csv_filename)

    # Parameters
    image_size = 128
    batch_size = 1080
    train_data_path = 'cats'
    
    # Call the function
    training_loader, validation_loader = prepare_data_loaders(image_size, batch_size, train_data_path)

    train_model(training_loader, num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost,decay=0)
