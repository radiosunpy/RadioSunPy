from pathlib import Path
import numpy as np 
import matplotlib.colors as colors
import matplotlib

CMAPS_DIR = Path(__file__).absolute().parent.joinpath('colormap_files')

def cmap_from_file(file_name, cmap_name): 
    color_data = np.loadtxt(CMAPS_DIR.joinpath(file_name + '.csv'), delimiter=',')
    r, g, b = color_data[:, 0], color_data[:, 1], color_data[:, 2]
    i = np.linspace(0, 1, r.size)
    color_dict = {
        name: list(zip(i, el / 255.0, el / 255.0))
        for el, name in [(r, 'red'), (g, 'green'), (b, 'blue')]
    }
    return colors.LinearSegmentedColormap(cmap_name, color_dict)

def std_gamma_2():
    return cmap_from_file('std_gamma_2', 'std_gamma_2.csv')

colormaps_list = {
    'std_gamma_2': std_gamma_2()
}

for name, colormap in colormaps_list.items():
    matplotlib.colormaps.register(colormap, name=name)