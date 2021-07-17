"""
write util functions to be used from each script

"""

import os


def check_file_exist(*files):
    
    if len(files) == 1:
        return os.path.isfile(files)
    
    else:
        for file in files:
            return os.path.isfile(file)
        
        
def get_class_list(class_path):
    class_list = []
    with open(class_path, 'r', encoding='UTF-8') as f:
        line = f.readline()
        while line:
            class_list.append(line.replace("\n",""))
            line = f.readline()
            
    return class_list