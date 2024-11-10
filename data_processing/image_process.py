from PIL import Image
import os

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

def dimension_reduce():
    root_dir = 'C:/Users/EricZ/Downloads/CSC2541/3828124/ADVANCE_vision/vision/'
    target_dir = 'C:/Users/EricZ/Downloads/CSC2541/3828124/ADVANCE_vision/vision_cleaned/'

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

            resize_image(file_path, curr_target_dir, filename, 4)
            
            count += 1
    print(count)


def convert_greyscale():
    root_dir = "C:/Users/EricZ/OneDrive/Documents/GitHub/Audio2Image/data/vision_cleaned"
    target_dir = "C:/Users/EricZ/OneDrive/Documents/GitHub/Audio2Image/data/vision_gs"

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

convert_greyscale()