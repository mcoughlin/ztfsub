
import numpy as np

from matplotlib import pyplot as plt

import aplpy

def plot_image(fitsfile,plotName,ra=None,dec=None,fwhm=None,catfile=None):

    fig = plt.figure()
    f1 = aplpy.FITSFigure(fitsfile,figure=fig)
    f1.set_tick_labels_font(size='x-small')
    f1.set_axis_labels_font(size='small')
    f1.show_grayscale()
    if not ra == None:
        f1.show_circles(ra,dec,fwhm/3600.0,zorder=99,linestyle='dashed', edgecolor='white')
    if not catfile == None:
        cat = np.loadtxt(catfile)
        if cat.size:
            ras, decs, fwhms = cat[:,3], cat[:,4], cat[:,6]
            f1.show_circles(ras,decs,fwhms,zorder=99,linestyle='dashed', edgecolor='red')
    fig.canvas.draw()
    plt.savefig(plotName)
    plt.close()
  
def plot_images(fitsfiles,plotName,ra=None,dec=None,fwhm=None,catfiles=False):

    fig = plt.figure(figsize=(12,12))
    for ii,filt in enumerate(fitsfiles.iterkeys()):
        fitsfile = fitsfiles[filt]
        if ii == 0:
            subplot = subplot=[0.1,0.1,0.35,0.35]
        elif ii == 1:
            subplot = subplot=[0.5,0.1,0.35,0.35]
        elif ii == 2:
            subplot = subplot=[0.1,0.5,0.35,0.35]
        elif ii == 3:
            subplot = subplot=[0.5,0.5,0.35,0.35]

        f1 = aplpy.FITSFigure(fitsfile,figure=fig,subplot=subplot)
        f1.set_tick_labels_font(size='x-small')
        f1.set_axis_labels_font(size='small')
        f1.show_grayscale()
        if not ra == None:
            f1.show_circles(ra,dec,fwhm/3600.0,zorder=99,linestyle='dashed', edgecolor='white')
        if catfiles:
            catfile = fitsfile.replace(".fits",".cat")
            cat = np.loadtxt(catfile)
            if cat.size:
                ras, decs, fwhms = cat[:,3], cat[:,4], cat[:,6]
                f1.show_circles(ras,decs,fwhms,zorder=99,linestyle='dashed', edgecolor='red')
        f1.set_title(filt)
    fig.canvas.draw()
    plt.savefig(plotName)
    plt.close()
 
