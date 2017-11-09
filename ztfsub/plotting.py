
from matplotlib import pyplot as plt

import aplpy

def plot_image(fitsfile,plotName):

    fig = plt.figure()
    f1 = aplpy.FITSFigure(fitsfile,figure=fig)
    f1.set_tick_labels_font(size='x-small')
    f1.set_axis_labels_font(size='small')
    f1.show_grayscale()
    plt.savefig(plotName)
    plt.close()
   
