from PIL import Image
import os

def process_image(path, target_path, file_name):
    image = Image.open(path)

    width, height = image.size
    new_size = (width//4, height//4)
    resized_image = image.resize(new_size)

    target_file = os.path.join(target_path, file_name)
    resized_image.save(target_file, optimize=True, quality=100)

    return


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

        process_image(file_path, curr_target_dir, filename)
        
        count += 1
print(count)