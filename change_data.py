import csv
import os
import random 
import numpy as np

name_colors = ['black', 'white', 'red', 'green']
color_mean = [[230, 25, 230]]

def random_data(length = 1000, scope = 25, value = [], name = ""):

    # Check parameter length.
    if not isinstance(length, int):
        raise ValueError(f"Parameter length should be type int.")

    assert length > 0, f"Parameter length should bigger than zero"

    # Check parameter range.
    if not isinstance(scope, int):
        raise ValueError(f"Parameter range should be type int.")

    assert 30 > scope > 0, f"Parameter range should bigger than zero and smaller than 30." 

    # Value color
    value_scope_color = []
    index = 0
    while index < length:
        # Random value
        
        rd_value = [random.randint(-scope, scope) for i in range(len(value))]
        v = list(np.array(value) - np.array(rd_value))
        if v not in value_scope_color:
            value_scope_color.append(v)
            index += 1

    return value_scope_color

# Write data
def write_data(path = "", dataset = None):
    assert  os.path.isfile(path), f"Not found file {path}."
    with open(path, 'w', newline="") as file:
        write = csv.writer(file)
        write.writerows(dataset)

if __name__ == '__main__':
    data = []
    for i, i_color in enumerate(color_mean):
        value = random_data(1000, 25, i_color, name_colors[i])
        data += value

    write_data(path = './data_max.csv',dataset = data)