import os
import torch
import json
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
from model import GCN_OLD_scheduling
from torch_geometric.data import Data

base_path = os.path.dirname(os.path.abspath(__file__))

# Load the DETR model
model_weights = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/best_models/model_best_sched.pth"

# Load the model weights onto the CPU
device = torch.device('cpu')
BESTboh = GCN_OLD_scheduling(hidden_layers=8, num_layers=0).to(device)
state_dict = torch.load(model_weights, map_location=device)
BESTboh.load_state_dict(state_dict)

# Define the input folder path
input_folder = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/StrawDI_Db1/val/img"

# Define the output folder path
output_folder = "/home/clara/Thesis/strawberry-pp-w-r-dataset-master/StrawDI_Db1/output"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


# Define the preprocessing function
def preprocess_image(image):
    # Resize the image to the desired dimensions
    resized_image = image.resize((1280, 720), Image.Resampling.LANCZOS)
    
    # Convert the image to a tensor
    image_tensor = F.to_tensor(resized_image)
    
    # Normalize the pixel values to the range [0, 1]
    image_tensor = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return image_tensor
    
# Iterate over the images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Load and preprocess the image
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0)
       
        # Create a Data object with the required attributes
        data = Data(x=image_tensor, edge_index=None, edge_weight=None, batch=None)
       
        # Perform segmentation using GCN_OLD_scheduling
        with torch.no_grad():
            output = BESTboh(data)
            
            
        # Retrieve the predicted segmentation masks
        masks = output["instances"].cpu().numpy()

        # Convert masks to PIL images and save them
        for i, mask in enumerate(masks):
            mask_pil = F.to_pil_image(mask)
            mask_pil.save(os.path.join(output_folder, f"{filename}_mask_{i}.png"))
            


            

