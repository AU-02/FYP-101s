import torch
import model as Model
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Path to the checkpoint and config file
checkpoint_gen_path = "D:\FYP-101s\experiments\depth_estimation_test1_250319_171929\checkpoint\I5000_E2_depth_gen.pth"
checkpoint_opt_path = "D:\FYP-101s\experiments\depth_estimation_test1_250319_171929\checkpoint\I5000_E2_depth_opt.pth"
config_path = "config\shadow.json"  # Path to your config file

# Load the configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Load the checkpoints safely
def load_checkpoint(checkpoint_path):
    # Load checkpoint with weights_only=True to avoid unsafe objects in unpickling
    return torch.load(checkpoint_path, map_location='cpu', weights_only=True)

# Load the checkpoint data
checkpoint_gen = load_checkpoint(checkpoint_gen_path)
checkpoint_opt = load_checkpoint(checkpoint_opt_path)

# Print the checkpoint keys for debugging
print("Checkpoint Gen Keys:", checkpoint_gen.keys())
print("Checkpoint Opt Keys:", checkpoint_opt.keys())

# Load configuration from JSON
opt = load_config(config_path)

# Create the model architecture and pass in the configuration (opt)
model = Model.create_model(opt)  # Pass the 'opt' argument to create_model

# Now, load the weights from the checkpoint into the model
model.load_state_dict(checkpoint_gen, strict=False)  # load all weights into the model

# Move the model to the correct device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully.")

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale (1 channel)
    
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),  # Convert to tensor
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)  # Move image to device
    return image_tensor

# Generate depth map
def generate_depth_map(image_path):
    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Generate the depth map using the sample method
    with torch.no_grad():
        batch_size = 1
        image_size = image_tensor.shape[2]  # Assuming image is (batch_size, channels, height, width)
        
        # Generate random noise with the same shape as the input image
        random_noise = torch.randn((batch_size, 1, image_size, image_size)).to(device)

        # Ensure the noise tensor has the correct shape
        random_noise = random_noise.expand_as(image_tensor)

        # Use the sample method with the random noise tensor
        output = model.sample(batch_size=batch_size, continous=False)  # This will use the sample method

        # Debug: Check if output is None
        if output is None:
            print("Error: The model output is None. Check the model and sampling process.")
            return

        print("Sample output:", output)  # Add this line to check if the output is None or a valid tensor

    # Convert the output tensor to numpy for visualization
    depth_map = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    
    # Visualize the depth map
    plt.imshow(depth_map, cmap='plasma')  # You can change the colormap to suit your preference
    plt.colorbar()
    plt.show()

# Path to the input image you want to generate a depth map for
image_path = '000001.png'

# Call the function to generate the depth map
generate_depth_map(image_path)