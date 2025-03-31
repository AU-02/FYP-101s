import torch
import model as Model
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Path to the checkpoint and config file
checkpoint_path = "D:/FYP-101s/experiments/depth_estimation_test1_250327_220026/checkpoint/final_model.pth"
config_path = "config/shadow.json"  # Path to your config file

# Load the configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check the keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# Load configuration from JSON
opt = load_config(config_path)

# Create the model architecture and pass in the configuration (opt)
model = Model.create_model(opt)  # Pass the 'opt' argument to create_model

# Now, load the weights from the checkpoint into the model
model.load_state_dict(checkpoint, strict=False)  # load all weights into the model

# Move the model to the correct device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully.")

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)  # No need to convert to grayscale since it's already a thermal image
    
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 (adjust to model expected size)
        transforms.ToTensor(),  # Convert to tensor
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)  # Move image to device
    
    return image_tensor

# Generate depth map
def generate_depth_map(image_path):
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(torch.float32)  # Ensure proper dtype

    # Convert the input tensor to a dictionary
    input_data = {
        'HR': image_tensor,  # Assuming 'HR' is the high-resolution image
        'SR': image_tensor   # Assuming 'SR' is the super-resolution image (or same as HR here)
    }

    # Print input data shapes for debugging
    print("Input data keys:", input_data.keys())
    print("x_in['HR'] shape:", input_data['HR'].shape)
    print("x_in['SR'] shape:", input_data['SR'].shape)

    # Test forward pass
    with torch.no_grad():
        try:
            output = model(input_data)  # Pass the dictionary to the model
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            return

        if output is None:
            print("Error: Model returned None")
            return

    # Check the shape of the output before proceeding
    print(f"Model output shape: {output.shape}")

    # If the output is a 4D tensor, remove the batch dimension (if batch size is 1)
    if output.dim() == 4:
        output = output.squeeze(0)  # Remove the batch dimension (if batch size is 1)

    # Apply a final 1x1 convolution to get the depth map (single channel output)
    final_depth_map = output[0, 0, :, :]  # Taking the first channel of the output (assuming the first channel corresponds to depth)

    # Check if the final depth map has the right shape
    print(f"Final depth map shape: {final_depth_map.shape}")

    # Ensure the depth map is a 2D array
    depth_map = final_depth_map.cpu().numpy()  # Move to CPU for visualization

    # Check if depth_map has the right shape
    if depth_map.ndim != 2:
        print(f"Error: Expected 2D depth map, but got shape {depth_map.shape}")
        return

    # Visualize the depth map
    plt.imshow(depth_map, cmap='plasma')  # You can change the colormap to suit your preference
    plt.colorbar()
    plt.show()

# Path to the input image you want to generate a depth map for
image_path = '000001.png'

# Call the function to generate the depth map
generate_depth_map(image_path)







