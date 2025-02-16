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


def keep_last_folders(path):
    # Normalize the path to handle mixed separators
    normalized_path = os.path.normpath(path)
    
    # Split the path into its components
    path_components = normalized_path.split(os.sep)
    
    # Check if there are at least two components
    if len(path_components) >= 3:
        # Join the last two components (last 2 folders and file name)
        result = os.path.join(path_components[-3], os.path.join(path_components[-2], path_components[-1]))
        return result
    else:
        return normalized_path  # If the path has less than two components, return as is
