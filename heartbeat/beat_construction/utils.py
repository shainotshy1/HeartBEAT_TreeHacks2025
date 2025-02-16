import os

def list_files_in_directory(directory):
    file_list = []
    
    # Recursively go through the directory and subdirectories
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        
        if os.path.isdir(full_path):  # If it's a directory, recurse into it
            file_list.extend(list_files_in_directory(full_path))
        else:  # If it's a file, add it to the list
            file_list.append(full_path)
    
    return file_list