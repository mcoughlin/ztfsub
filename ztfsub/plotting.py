
import os, sys
import numpy as np
import astropy
from matplotlib import pyplot as plt
from astropy.io import fits
import aplpy

def plot_cube(fitsfile,plotDir,ra=None,dec=None,fwhm=None,catfile=None):

    cat = np.loadtxt(catfile)
    xs, ys, fluxes, fluxerrs, mags, magerrs, ras, decs, A, B, A_world, B_world, theta, theta_world, fwhms, fwhms_world, extnumber = cat[:,0], cat[:,1], cat[:,2], cat[:,3], cat[:,4], cat[:,5], cat[:,6], cat[:,7], cat[:,8], cat[:,9], cat[:,10], cat[:,11], cat[:,12], cat[:,13], cat[:,14], cat[:,15], cat[:,16]

    hdulist = fits.open(fitsfile)
    for ii,hdu in enumerate(hdulist):
        if ii == 0: continue

        idx = np.where(extnumber == ii)[0]

        plotName = os.path.join(plotDir,'%04d.png'%ii)
        fig = plt.figure(figsize=(10,10))
        f1 = aplpy.FITSFigure(fitsfile,figure=fig,hdu=ii)
        f1.set_tick_labels_font(size='x-small')
        f1.set_axis_labels_font(size='small')
        f1.show_grayscale()
        plt.scatter(xs[idx],ys[idx],s=fwhms[idx],zorder=99,facecolors='none', edgecolors='green')
        #f1.show_circles(xs[idx],ys[idx],fwhms[idx],zorder=99,linestyle='dashed', edgecolor='white')
        fig.canvas.draw()
        plt.savefig(plotName)
        plt.close()

def plot_image(fitsfile,plotName,ra=None,dec=None,fwhm=None,catfile=None):

    fig = plt.figure(figsize=(10,10))
    f1 = aplpy.FITSFigure(fitsfile,figure=fig)
    f1.set_tick_labels_font(size='x-small')
    f1.set_axis_labels_font(size='small')
    f1.show_grayscale()
    if not ra == None:
        f1.show_circles(ra,dec,fwhm/3600.0,zorder=99,linestyle='dashed', edgecolor='white')
    if not catfile == None:
        try:
            cat = np.loadtxt(catfile)
            if cat.size:
                if cat.size > 8:
                    xs, ys, fluxes, fluxerrs, mags, magerrs, ras, decs, A, B, A_world, B_world, theta, theta_world, fwhms, fwhms_world, extnumber = cat[:,0], cat[:,1], cat[:,2], cat[:,3], cat[:,4], cat[:,5], cat[:,6], cat[:,7], cat[:,8], cat[:,9], cat[:,10], cat[:,11], cat[:,12], cat[:,13], cat[:,14], cat[:,15], cat[:,16]
                else:
                    xs, ys, fluxes, fluxerrs, mags, magerrs, ras, decs, A, B, A_world, B_world, theta, theta_world, fwhms, fwhms_world, extnumber = cat[0], cat[1], cat[2], cat[3], cat[4], cat[5], cat[6], cat[7], cat[8], cat[9], cat[10], cat[11], cat[12], cat[13], cat[14], cat[15], cat[16]

                try:
                    f1.show_circles(ras,decs,fwhms,zorder=99,linestyle='dashed', edgecolor='red')
                except:
                    plt.scatter(xs,ys,s=fwhms,zorder=99,facecolors='none', edgecolors='green')

        except:
            hdulist=astropy.io.fits.open(catfile)
            header = hdulist[1].header
            data = hdulist[1].data
 
            ras, decs, fwhms = [], [], []
            for row in data:
                ragalaxy, decgalaxy, fwhm = row[35], row[36], row[13]
                ras.append(ragalaxy)
                decs.append(decgalaxy)
                fwhms.append(fwhm/3600.0)
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
            try:
                cat = np.loadtxt(catfile)
                if cat.size:
                    ras, decs, fwhms = cat[:,3], cat[:,4], cat[:,6]
                    f1.show_circles(ras,decs,fwhms,zorder=99,linestyle='dashed', edgecolor='red')
            except:
                hdulist=astropy.io.fits.open(catfile)
                header = hdulist[1].header
                data = hdulist[1].data

                ras, decs, fwhms = [], [], []
                for row in data:
                    ragalaxy, decgalaxy, fwhm = row[35], row[36], row[13]
                    ras.append(ragalaxy)
                    decs.append(decgalaxy)
                    fwhms.append(fwhm/3600.0) 
                f1.show_circles(ras,decs,fwhms,zorder=99,linestyle='dashed', edgecolor='red')

        f1.set_title(filt)
    fig.canvas.draw()
    plt.savefig(plotName)
    plt.close()
 
