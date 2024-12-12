import os
import time

FILE_1GB_SIZE = 1024 * 1024 * 1024  # 1GB in bytes

def create_large_file(filename, size_in_bytes):
    """Creates a file of the specified size.

    Args:
    filename: The name of the file to create.
    size_in_bytes: The desired size of the file in bytes.
    """
    print(f"creating {filename}")
    try:
        f = open(filename, 'wb')
        f.seek(size_in_bytes - 1)
        f.write(b'\0')
        f.close()
    
    except OSError as e:
        if e.errno == 28:
            print("Error: Disk is full. Please free up some space.")
        raise

def delete_file(filename):
    """Deletes a file with the given filename.

    Args:
    filename: The name of the file to delete.
    """

    if os.path.isfile(filename):
        os.remove(filename)

# Create a 1GB file

def garbage_start():
    idx = 0
    while idx < 60:
        file_name = f'Garbages/1GB_file_{idx}.dat'
        if not os.path.isfile(file_name):
            create_large_file(file_name, FILE_1GB_SIZE)
            time.sleep(1)
        idx += 1

def waste_time_start():
    print("start wasting time~~")
    while True:
        time.sleep(60)

if __name__ == "__main__":
    try:
        garbage_start()
    except OSError as e:
        waste_time_start()
