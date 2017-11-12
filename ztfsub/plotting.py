
from matplotlib import pyplot as plt

import aplpy

def plot_image(fitsfile,plotName,ra=None,dec=None,fwhm=None):

    fig = plt.figure()
    f1 = aplpy.FITSFigure(fitsfile,figure=fig)
    f1.set_tick_labels_font(size='x-small')
    f1.set_axis_labels_font(size='small')
    f1.show_grayscale()
    if not ra == None:
        f1.show_circles(ra,dec,fwhm/3600.0,zorder=99,linestyle='dashed', edgecolor='white')
    fig.canvas.draw()
    plt.savefig(plotName)
    plt.close()
   
