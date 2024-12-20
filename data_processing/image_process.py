from PIL import Image
import os
import torch
import torchvision.transforms as transforms


def resize_image(path, target_path, file_name, factor):
    image = Image.open(path)

    width, height = image.size
    new_size = (width//factor, height//factor)
    resized_image = image.resize(new_size)

    target_file = os.path.join(target_path, file_name)
    resized_image.save(target_file, optimize=True, quality=100)

    return

def convert_to_greyscale_image(path, target_path, file_name):
    
    image = Image.open(path).convert("L")

    target_file = os.path.join(target_path, file_name)
    image.save(target_file, optimize=True, quality=100)

    return


def convert_to_tensor(path, target_path, file_name):
    
    image = Image.open(path)

    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.squeeze(dim=0)

    new_filename = os.path.splitext(file_name)[0] + ".pt"
    target_file = os.path.join(target_path, new_filename)
    torch.save(image_tensor, target_file)

    return



def dimension_reduce():
    root_dir = os.path.join("data", "image_original")
    target_dir = os.path.join("data", "vision_cleaned")

    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_folder = os.path.basename(dirpath)
        curr_target_dir = os.path.join(target_dir, current_folder)
        if not os.path.exists(curr_target_dir):
            os.makedirs(curr_target_dir)
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(dirpath, filename)

            resize_image(file_path, curr_target_dir, filename, 16)
            
            count += 1
    print(count)


def convert_greyscale():
    root_dir = os.path.join("data", "vision_cleaned")
    target_dir = os.path.join("data", "vision_gs")

    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_folder = os.path.basename(dirpath)
        curr_target_dir = os.path.join(target_dir, current_folder)
        if not os.path.exists(curr_target_dir):
            os.makedirs(curr_target_dir)
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(dirpath, filename)

            convert_to_greyscale_image(file_path, curr_target_dir, filename)
            
            count += 1
    print(count)

def gs_to_tensor():
    convert_greyscale()
    root_dir = os.path.join("data", "vision_gs")
    target_dir = os.path.join("data", "vision_gs_tensor")

    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_folder = os.path.basename(dirpath)
        curr_target_dir = os.path.join(target_dir, current_folder)
        if not os.path.exists(curr_target_dir):
            os.makedirs(curr_target_dir)
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(dirpath, filename)

            convert_to_tensor(file_path, curr_target_dir, filename)
            
            count += 1

    print(count)


def big_gs_tensor():
    gs_to_tensor()
    root_dir = os.path.join("data", "vision_gs_tensor")
    count = 0
    tensor_list = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(dirpath, filename)
            curr_tensor = torch.load(file_path)
            curr_tensor = curr_tensor.flatten()
            tensor_list.append(curr_tensor.unsqueeze(dim=0))
            count += 1
            
    try:
        tensor = torch.cat(tensor_list, dim=0)
        print(tensor.shape)
    except RuntimeError as e:
        print(f"Error stacking tensors: {e}")
    
    torch.save(tensor, os.path.join("data", "gs_image_tensor.pt"))

    print("Total tensors loaded:", count)

def big_rgb_tensor():
    root_dir = os.path.join("data", "vision_cleaned")
    count = 0
    tensor_list = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(dirpath, filename)
            
            image = Image.open(file_path).convert("RGB")
    
            # Transform the image to a tensor with shape [3, H, W]
            transform = transforms.ToTensor()
            img_tensor = transform(image)  # Shape: [3, H, W]
            
            # Permute to [H, W, 3] and reshape to [H, 3*W]
            img_tensor = img_tensor.permute(1, 2, 0)  # Shape: [H, W, 3]
            h, w, c = img_tensor.shape
            unfolded_tensor = img_tensor.reshape(h, c * w)  # Shape: [H, 3W]

            unfolded_tensor = unfolded_tensor.flatten()
            
            tensor_list.append(unfolded_tensor.unsqueeze(dim=0))
            count += 1
            
    try:
        tensor = torch.cat(tensor_list, dim=0)
        print(tensor.shape)
    except RuntimeError as e:
        print(f"Error stacking tensors: {e}")
    
    torch.save(tensor, os.path.join("data", "rgb_image_tensor.pt"))

    print("Total tensors loaded:", count)


def convert_image_to_dataset(img_data_path, dataset_path, rgb=False):
    count = 0
    img_tensor_list = []
        # Get and sort all subfolders in the root folder
    subfolders = [f.path for f in os.scandir(img_data_path) if f.is_dir()]
    subfolders.sort()  # Sorting subfolders in lexicographical order

    for subfolder in subfolders:
        print(f"Reading files from: {subfolder}")
        # if (subfolder != "data/image_original\\airport"):
        #     print("skipping")
        #     continue

        # Get and sort all files in the current subfolder
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]
        files.sort()  # Sorting files in lexicographical order


        for file in files:
            if os.path.basename(file) == ".DS_Store":
                continue
            # Iterate through images
            # file_path = os.path.join(dirpath, filename)
            if rgb:
                image = Image.open(file).convert("RGB")

                image = image.resize((32, 32))
                transform = transforms.ToTensor()
                img_tensor = transform(image)  # Shape: [3, H, W]
                
                # Permute to [H, W, 3] and reshape to [H, 3*W]
                img_tensor = img_tensor.permute(1, 2, 0)  # Shape: [H, W, 3]
                h, w, c = img_tensor.shape
                img_tensor = img_tensor.reshape(h, c * w)  # Shape: [H, 3W]
            else:
                # load grayscale image  
                image = Image.open(file).convert("L")
                # resize
                image = image.resize((32, 32))
                # convert to tensor
                transform = transforms.ToTensor()
                img_tensor = transform(image)
            # append to list
            img_tensor = img_tensor.flatten()
            img_tensor_list.append(img_tensor.unsqueeze(dim=0))
            count += 1
            
    try:
        img_tensor = torch.cat(img_tensor_list, dim=0)
        img_tensor = img_tensor * 255
        print(img_tensor.shape)
    except RuntimeError as e:
        print(f"Error stacking tensors: {e}")
        
    torch.save(img_tensor, dataset_path)


# Run the function with relative paths
if __name__ == "__main__":
    # dimension_reduce()
    # big_gs_tensor()
    # big_rgb_tensor()
    # convert_image_to_dataset("data/image_original", "data/image_tensor.pt")
    convert_image_to_dataset("data/image_original", "data/image_tensor_rgb.pt", True)    

    #print(torch.load("data/DS_vision_gs1.pt"))