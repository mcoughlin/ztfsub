
import os, sys, optparse, shutil, glob
import random, string
import numpy as np
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

import matplotlib.path as mplPath
import h5py
from astropy.io import fits
import aplpy

import requests
from lxml.html import fromstring

import ztfsub.utils

def get_ps1(opts,imagefile,ra,dec,filt):

    ps1Dir = '%s/ps1'%opts.dataDir
    if not os.path.isdir(ps1Dir):
        os.makedirs(ps1Dir)

    ps1ResampleDir = '%s/ps1_resample'%opts.dataDir
    if not os.path.isdir(ps1ResampleDir):
        os.makedirs(ps1ResampleDir)

    if os.path.isfile(imagefile):
        return

    BaseURL = "http://ps1images.stsci.edu/"

    N = 10
    listfile = opts.tmpDir + "/ps1_" + filt + "_list_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N)) + '.txt'

    wget_command = 'wget "http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=%.5f&dec=%.5f&filters=%s" -O %s'%(ra,dec,filt,listfile)
    os.system(wget_command)

    lines = [line.rstrip('\n') for line in open(listfile)]
    lines = lines[1:]

    if len(lines) == 0:
        rm_command = "rm %s"%listfile
        os.system(rm_command)
        print "No PS1 images available... returning."
        return False

    fid = open(listfile,'w')
    for line in lines:
        lineSplit = line.split(" ")
        datafits = lineSplit[-2]
        datafitsshort = lineSplit[-1]
        datafitsshort = datafitsshort.replace(".","_").replace("_fits",".fits")

        Link = '%s%s'%(BaseURL,datafits)
        FileNameFitsPath = '%s/%s'%(ps1Dir,datafitsshort)
  
        fid.write('%s\n'%(FileNameFitsPath))
        if os.path.isfile(FileNameFitsPath): continue

        wget_command = "wget %s -O %s"%(Link,FileNameFitsPath)
        os.system(wget_command)

        funpack_command = "fpack %s; rm %s; funpack %s.fz"%(FileNameFitsPath,FileNameFitsPath,FileNameFitsPath)
        os.system(funpack_command)

        rm_command = "rm funpack %s.fz"%(FileNameFitsPath)
        os.system(rm_command)

        #NAXIS1, NAXIS2 = ztfsub.utils.get_head(FileNameFitsPath,['NAXIS1','NAXIS2'],hdunum=1)
         
        #swarpcmd='swarp %s -CENTER_TYPE ALL -PIXELSCALE_TYPE MEDIAN -IMAGE_SIZE "%i, %i" -SUBTRACT_BACK N -IMAGEOUT_NAME %s -COPY_KEYWORDS FILTER' % (FileNameFitsPath, NAXIS1, NAXIS2, FileNameFitsPath)
        #os.system(swarpcmd)

    fid.close()

    image_scale = float(1.0/0.5)
    image_size = int(opts.image_size*image_scale)

    swarp_command = 'swarp @%s -c %s/swarp.conf -CENTER %.5f,%.5f -IMAGE_SIZE %d,%d  -PIXEL_SCALE 0.258 -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s/coadd.weight.fits -RESAMPLE_DIR %s -XML_NAME %s/swarp.xml -COPY_KEYWORDS PIXEL_SCALE'%(listfile,opts.defaultsDir,ra,dec,image_size,image_size,imagefile,opts.tmpDir,ps1ResampleDir,opts.tmpDir)
    os.system(swarp_command)

    # replace borders with NaNs in ref image if there are any that are == 0,
    hdulist=fits.open(imagefile)
    hdulist[0].data[hdulist[0].data==0]=np.nan
    hdulist.writeto(imagefile,overwrite=True)

    rm_command = "rm *.fits"
    os.system(rm_command)
    rm_command = "rm *.bz2"
    os.system(rm_command)

    return True

def get_sdss(opts,imagefile,ra,dec,filt):

    sdssDir = '%s/sdss'%opts.dataDir
    if not os.path.isdir(sdssDir):
        os.makedirs(sdssDir)

    sdssResampleDir = '%s/sdss_resample'%opts.dataDir
    if not os.path.isdir(sdssResampleDir):
        os.makedirs(sdssResampleDir)

    hdf5file = "%s/SDSS_DR9_Fields_All_PolySort.hdf5"%opts.inputDir
    f = h5py.File(hdf5file, 'r')
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])
    Run, Rerun, Camcol, Field, MJD_u, MJD_g, MJD_r, MJD_i, MJD_z, RA1, RA2, RA3, RA4, Dec1, Dec2, Dec3, Dec4, IndexPoly = data
    RAAve = (RA1+RA2+RA3+RA4)/4.0
    DecAve = (Dec1+Dec2+Dec3+Dec4)/4.0

    catcoord = SkyCoord(RA1*u.rad,Dec1*u.rad,frame='icrs')
    #%%%%%%%%%%%%check if the sdss ref already exists in ../SN/refs/, each band
    #sdss_bands=['u','g','r','i','z']
    #sdss_bands=['g','r']
    ra_box = np.arange(-0.2,0.3,0.1)
    dec_box = np.arange(-0.2,0.3,0.1)

    BaseURL = 'http://data.sdss3.org/sas/dr9/boss/photoObj/frames/'

    Threshold  = 0.3
    if os.path.isfile(imagefile):
        return

    N = 10
    listfile = opts.tmpDir + "/sdss_" + filt + "_list_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N)) + '.txt'

    fid = open(listfile,'w')
    for ii in xrange(len(ra_box)):
        for jj in xrange(len(dec_box)):
            thisra = np.deg2rad(ra + ra_box[ii])
            thisdec = np.deg2rad(dec + dec_box[jj])

            coord = SkyCoord(thisra*u.rad,thisdec*u.rad,frame='icrs')
            coord_distance = catcoord.separation(coord)

            Iids = np.where(coord_distance.deg <= Threshold)[0]
            for Iid in Iids:
                bbPath = mplPath.Path(np.array([[RA1[Iid],Dec1[Iid]],[RA2[Iid],Dec2[Iid]],[RA3[Iid],Dec3[Iid]],[RA4[Iid],Dec4[Iid]]]))
                check1 = bbPath.contains_point((thisra, thisdec))
                check2 = bbPath.contains_point((thisra-2*np.pi, thisdec))
                check3 = bbPath.contains_point((thisra+2*np.pi, thisdec))

                check = check1 or check2 or check3
                if not check: continue

                URL = '%s/%d/%d/%d/'%(BaseURL,Rerun[Iid],Run[Iid],Camcol[Iid])
                FileName = 'frame-%s-%06d-%d-%04d.fits.bz2'%(filt,Run[Iid],Camcol[Iid],Field[Iid])
                FileNameFits = 'frame-%s-%06d-%d-%04d.fits'%(filt,Run[Iid],Camcol[Iid],Field[Iid])
                FileNameFitsPath = '%s/frame-%s-%06d-%d-%04d.fits'%(sdssDir,filt,Run[Iid],Camcol[Iid],Field[Iid])
                Link = '%s%s'%(URL,FileName)

                fid.write('%s\n'%(FileNameFitsPath))
                if os.path.isfile(FileNameFitsPath): continue

                wget_command = "wget %s"%Link
                os.system(wget_command)
                bunzip2_command = "bunzip2 %s"%FileName
                os.system(bunzip2_command)

                mv_command = 'mv %s %s'%(FileNameFits,sdssDir)
                os.system(mv_command)

    fid.close()

    lines = [line.rstrip('\n') for line in open(listfile)]
    if len(lines) == 0:
        rm_command = "rm %s"%listfile
        os.system(rm_command)
        print "No SDSS images available... returning."
        return False

    swarp_command = 'swarp @%s -c %s/swarp.conf -CENTER %.5f,%.5f -IMAGE_SIZE %d,%d  -PIXEL_SCALE 0.396127 -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s/coadd.weight.fits -RESAMPLE_DIR %s -XML_NAME %s/swarp.xml'%(listfile,opts.defaultsDir,ra,dec,opts.image_size,opts.image_size,imagefile,opts.tmpDir,sdssResampleDir,opts.tmpDir)
    os.system(swarp_command)

    # replace borders with NaNs in ref image if there are any that are == 0,
    hdulist=fits.open(imagefile)
    hdulist[0].data[hdulist[0].data==0]=np.nan
    # scale values
    #hdulist[0].data = hdulist[0].data*100.0/np.nanstd(hdulist[0].data)
    hdulist[0].data = hdulist[0].data + 1000
    #hdulist[0].data = hdulist[0].data*100.0/np.nanstd(hdulist[0].data)
    hdulist.writeto(imagefile,overwrite=True)

    rm_command = "rm *.fits"
    os.system(rm_command)
    rm_command = "rm *.bz2"
    os.system(rm_command)

    return True

def get_ztf(opts,imagefile,imagenum):
    linksFile = '%s/links.txt'%opts.outputDir
    links = [line.rstrip('\n') for line in open(linksFile)]

    ztfDir = '%s/ztf'%opts.dataDir
    if not os.path.isdir(ztfDir):
        os.makedirs(ztfDir)

    ztfResampleDir = '%s/ztf_resample'%opts.dataDir
    if not os.path.isdir(ztfResampleDir):
        os.makedirs(ztfResampleDir)

    images = []
    for link in links:
        linkSplit = link.split("/")
        imagenumSplit = imagenum.split("_")
        imageday = int(imagenumSplit[0])
        imagedaynum = int(imagenumSplit[1])

        if (imageday == int(linkSplit[-3])) and (imagedaynum == int(linkSplit[-2])):
            images.append(link)

    tilesFile = '%s/ZTF_Fields.txt'%opts.inputDir
    tiles = np.loadtxt(tilesFile,comments='%')

    imageSplit = images[0].replace(".fits","").split("_")
    fieldID = int(imageSplit[-6])
    #idx = np.where(fieldID == tiles[:,0])[0]
    #tile = tiles[idx][0]
    #fieldID, tilera, tiledec = tile[0], tile[1], tile[2]

    N = 10
    listfile = opts.tmpDir + "/ztf_list_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N)) + '.txt'

    raimages, decimages = [], []
    for image in images:
        imagefinal = '%s/%s'%(ztfDir,image.split("/")[-1])
        if not os.path.isfile(imagefinal):
            wget_command = "wget %s --user %s --password %s -O %s"%(image,os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"],imagefinal)
            os.system(wget_command)
        raimage, decimage = ztfsub.utils.get_radec_from_wcs(imagefinal)
        raimages.append(raimage)
        decimages.append(decimage)

    if not opts.ra == None:
        ra = opts.ra
    else:
        ra = np.mean(raimages)

    if not opts.declination == None:
        dec = opts.declination
    else:
        dec = np.mean(decimages)

    coord = SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')
 
    Threshold  = 0.5
    catcoord = SkyCoord(np.array(raimages)*u.deg,np.array(decimages)*u.deg,frame='icrs') 
    coord_distance = catcoord.separation(coord) 
    Iids = np.where(coord_distance.deg <= Threshold)[0]

    if len(Iids) == 0:
        print "No ZTF images available... returning."
        return False

    fid = open(listfile,'w')
    for Iid in Iids:
        image = images[Iid]
        imagefinal = '%s/%s'%(ztfDir,image.split("/")[-1])
        fid.write('%s\n'%(imagefinal))
    fid.close()

    swarp_command = 'swarp @%s -c %s/swarp.conf -CENTER %.5f,%.5f -IMAGE_SIZE %d,%d -PIXEL_SCALE 1.0 -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s/coadd.weight.fits -RESAMPLE_DIR %s -XML_NAME %s/swarp.xml'%(listfile,opts.defaultsDir,ra,dec,opts.image_size,opts.image_size,imagefile,opts.tmpDir,ztfResampleDir,opts.tmpDir)
    os.system(swarp_command)

    # replace borders with NaNs in ref image if there are any that are == 0,
    hdulist=fits.open(imagefile)
    hdulist[0].data[hdulist[0].data==0]=np.nan
    hdulist.writeto(imagefile,overwrite=True)

    return True

def get_ptf(opts,imageDir,ra=None,ra_size=None,dec=None,dec_size=None):

    ptfResampleDir = '%s/ptf_resample/%s'%(opts.dataDir,opts.field)
    if not os.path.isdir(ptfResampleDir):
        os.makedirs(ptfResampleDir)

    images = glob.glob('%s/ptf/%s/*.fits'%(opts.dataDir,opts.field))

    raimages, decimages = [], []
    data = {}
    for image in images:
        imagefile = image.split("/")[-1]
        if "201603213319" in imagefile: continue
        if "mask" in imagefile: continue

        hdulist=fits.open(image)
        header = hdulist[0].header
        filt = header['FILTER']
        raimage, decimage = ztfsub.utils.get_radec_from_wcs(image)

        if not filt in data:
            data[filt] = {}
            data[filt]["files"] = []
            data[filt]["ras"] = []
            data[filt]["decs"] = []

        data[filt]["files"].append(image)
        data[filt]["ras"].append(raimage)
        data[filt]["decs"].append(decimage)

        raimages.append(raimage)
        decimages.append(decimage)

    if ra == None:
        ra = np.mean(raimages)
    if dec == None:
        dec = np.mean(decimages)
    if ra_size == None:
        ra_size = opts.image_size
    if dec == None:
        dec_size = opts.image_size

    coord = SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')

    N = 10

    goodimages = True
    imagefiles = {}

    if ra_size > 200:
        Threshold  = 10.0
    else:
        Threshold  = 0.5
    for filt in data.iterkeys():
        imagefile = "%s/ptf_%s.fits"%(imageDir,filt) 
        imagefiles[filt] = imagefile

        listfile = opts.tmpDir + "/ptf_list_" + filt + '_' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N)) + '.txt'

        raimages, decimages = data[filt]["ras"], data[filt]["decs"]
        images = data[filt]["files"]

        catcoord = SkyCoord(np.array(raimages)*u.deg,np.array(decimages)*u.deg,frame='icrs')
        coord_distance = catcoord.separation(coord)
        Iids = np.where(coord_distance.deg <= Threshold)[0]

        if len(Iids) == 0:
            print "No PTF images available... returning."
            return [], False

        fid = open(listfile,'w')
        for Iid in Iids:
            image = images[Iid]
            fid.write('%s\n'%(image))
        fid.close()

        swarp_command = 'swarp @%s -c %s/swarp.conf -CENTER %.5f,%.5f -IMAGE_SIZE %d,%d -PIXEL_SCALE 1.0 -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s/coadd.weight.fits -RESAMPLE_DIR %s -XML_NAME %s/swarp.xml'%(listfile,opts.defaultsDir,ra,dec,ra_size,dec_size,imagefile,opts.tmpDir,ptfResampleDir,opts.tmpDir)
        os.system(swarp_command)

        # replace borders with NaNs in ref image if there are any that are == 0,
        hdulist=fits.open(imagefile)
 
        if np.nansum(hdulist[0].data) == 0:
            goodimages = False

        hdulist[0].data[hdulist[0].data==0]=np.nan
        hdulist.writeto(imagefile,overwrite=True)

    return imagefiles, goodimages
