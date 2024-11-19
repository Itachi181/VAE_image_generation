import torch
from VAE_training import Model
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate new images
def generate_images(model, num_samples=16, latent_size=(32, 32)):
    """
    Generate images using the decoder of the trained VAE model.

    Args:
        model (torch.nn.Module): The trained VAE model.
        num_samples (int): Number of images to generate.
        latent_size (tuple): Spatial size of the latent embeddings (H, W).

    Returns:
        torch.Tensor: Generated images with shape [num_samples, 3, H, W].
    """
    # Randomly sample latent indices
    latent_indices = torch.randint(0, model._vq_vae._num_embeddings, (num_samples, *latent_size)).to(device)

    # Get embeddings from the codebook
    embeddings = model._vq_vae._embedding(latent_indices)
    embeddings = embeddings.permute(0, 3, 1, 2)  # Rearrange to [batch, embedding_dim, H, W]

    with torch.no_grad():
        generated_images = model._decoder(embeddings)
    
    return generated_images

if __name__ == "__main__":

    # Load the trained model
    model_path = "vae_cat_trained_model_50k/Advance_model_5000.pth"
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 512
    num_embeddings = 512
    commitment_cost = 0.25
    
    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate and visualize images
    latent_size = (32, 32)  # Adjust based on your encoder's output size
    generated_images = generate_images(model, num_samples=16, latent_size=latent_size)
    generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (batch, H, W, C)
    
    # Display or save images
    for i, img in enumerate(generated_images):
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255
        Image.fromarray(img).save(f"generated_image_{i}.png")
