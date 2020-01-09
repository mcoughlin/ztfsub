#import pyraf
#from pyraf import iraf
import copy, os, shutil, glob, sys, string, re, math, operator, numpy, time
#import pyfits
from types import *

import numpy as np
import datetime

from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

import requests
from lxml.html import fromstring

import PythonPhot as pp

P60DISTORT = "P60distort.head"
SUBREGION = "[1:1024,1:1024]"

######################################################################

def get_radec_from_header(fitsfile,ext=0):
    header = fits.getheader(fitsfile,ext=ext)

    ra, dec = header["CRVAL1"], header["CRVAL2"]
    return ra, dec

def get_radec_from_wcs(fitsfile,ext=0):
    header = fits.getheader(fitsfile,ext=ext)
 
    w = WCS(header)
    ra, dec = w.wcs_pix2world(float(header['NAXIS1'])/2.0, float(header['NAXIS2'])/2.0,1)

    return ra, dec

def get_radec_limits_from_wcs(fitsfile):
    header = fits.getheader(fitsfile)

    w = WCS(header)
    ra1, dec1 = w.wcs_pix2world(0,0,1)
    ra2, dec2 = w.wcs_pix2world(float(header['NAXIS1']), float(header['NAXIS2']),1) 

    ra_min, ra_max = np.min([ra1,ra2]), np.max([ra1,ra2])
    dec_min, dec_max = np.min([dec1,dec2]), np.max([dec1,dec2])

    return [ra_min,ra_max], [dec_min,dec_max]

def get_head(imagefile,keywords,hdunum=0):

    hdulist=fits.open(imagefile)
    header = hdulist[hdunum].header
    vals = []
    for key in keywords:
        try:
            vals.append(float(header[key]))
        except:
            vals.append(header[key])
    
    return vals

def p60hotpants(inlis, refimage, outimage, tu=50000, iu=50000, ig=2.3, tg=2.3,
                stamps=None, nsx=4, nsy=4, ko=0, bgo=0, radius=10, 
                tlow=0, ilow=0, sthresh=5.0, ng=None,
                ngref=[3, 6, 0.70, 4, 1.50, 2, 3.00], scimage=False,
                defaultsDir = "defaults"):

    '''P60 Subtraction using HOTPANTS'''

    subimages=[inlis]

    hdulist=fits.open(refimage)
    hdulist[0].data[np.isnan(hdulist[0].data)]=0.0
    hdulist.writeto(refimage,clobber=True)

    for image in subimages:

        hdulist=fits.open(image)
        hdulist[0].data[np.isnan(hdulist[0].data)]=0.0
        hdulist.writeto(image,clobber=True)

        root = image.split('.')[0]
        scmd="hotpants -inim %s -tmplim %s -outim %s -tu %.2f -tuk %.2f -iu %.2f -iuk %.2f -ig %.2f -tg %.2f -savexy %s.st -ko %i -bgo %i -nsx %i -nsy %i -r %i -rss %i -tl %f -il %f -ft %f -v 0 -c t -n i" % (image, refimage, outimage, tu, tu, iu, iu, ig, tg, root, ko, bgo, nsx, nsy, radius, radius*1.5, tlow, ilow, sthresh)
        #scmd="hotpants -inim %s -tmplim %s -outim %s" % (image, refimage, outimage)
        if not (stamps==None):
            scmd += " -ssf %s -afssc 0" % stamps 
        if (ng==None):
            scmd += " -ng %i " % ngref[0]
            #seepix=get_head(image,['SEEPIX'])
            seepix=2
            for k in range(1, len(ngref)-1, 2):
                scmd+="%i %.2f " % (ngref[k], ngref[k+1]*float(seepix)/3.0)
        if (scimage==True):
            scmd += ' -n i '

        #scmd += "-oni noise.fits -ond noise_diff.fits -oci convolved.fits"

        cmd=os.popen(scmd,'r')
        hlines=cmd.readlines()
        cmd.close()

    return

############################################################################

def p60swarp(opts, image, outfile, ractr=None, dcctr=None, pixscale=None, size=None,backsub=False):

    '''Run SWarp on P60 images'''

    resampleDir = "/".join(image.split("/")[:-1])

    swarpcmd='swarp %s ' % image
    if ractr!=None or dcctr!=None:
        swarpcmd+='-CENTER_TYPE MANUAL -CENTER "%s, %s" ' % (ractr, dcctr)
    if pixscale!=None:
        swarpcmd+='-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE %f ' % pixscale
    if size!=None:
        swarpcmd+='-IMAGE_SIZE "%i, %i" ' % (size[0], size[1])
    if backsub==False:
        swarpcmd+='-SUBTRACT_BACK N '
    swarpcmd+='-COPY_KEYWORDS OBJECT,SKYSUB,SKYBKG,SEEPIX,PIXEL_SCALE '
    swarpcmd+='-c %s/swarp.conf -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s/coadd.weight.fits -RESAMPLE_DIR %s -XML_NAME %s/swarp.xml'%(opts.defaultsDir,outfile,opts.tmpDir,resampleDir,opts.tmpDir)

    scmd=os.popen(swarpcmd, 'r', -1)
    scmd.readlines()

def getsdssfits(object, ra, dec, filters=['g', 'r', 'i', 'z'], width=0.15, 
                pix=0.378, tmpdir='tmp'):

    '''Use Montage to create SDSS templates for given position.'''

    # Remove old files if necessary
    if os.path.exists("%s.hdr" % object):
        os.remove("%s.hdr" % object)
    if os.path.exists(tmpdir):
        os.system("rm -rf %s" % tmpdir)

    hdrcmd = "mHdr -p %.3f \"%s %s\" %.2f %s.hdr" % (pix, ra, dec, width, 
              object)
    hcmd = os.popen(hdrcmd,'r')
    hlines = hcmd.readlines()
    if re.search("ERROR", hlines[0]):
        print "Error constructing SDSS reference: %s" % hlines[0]
        return -1

    # Loop through filters
    for filter in filters:

        excmd = "mExec -o %s-%s.fits -f %s.hdr SDSS %s %s" % (object, filter,
                 object, filter, tmpdir)
        ecmd = os.popen(excmd, 'r')
        elines = ecmd.readlines()
        if re.search("ERROR", elines[0]):
            print "Error constructing SDSS reference: %s" % elines[0]
            return -1

    # Return success
    return 1

############################################################################

def getrefstars(object, ra, dec, width=0.15, pmcut=True):

    '''Get SDSS reference objects for PSF matching and photometry'''

    sdsscmd = "getsdss.pl -r %.2f " % (width * 40.0)
    if pmcut:
        sdsscmd += "-u -f %s-nopm.reg %s %s %s-nopm.txt" % (object,
                    ra, dec, object)
    else:
        sdsscmd += "-b -f %s-allpm.reg %s %s %s-allpm.txt" % (object,
                    ra, dec, object)
    scmd = os.popen(sdsscmd, 'r')
    lines = scmd.readlines()

    if pmcut:
        stars=Starlist("%s-nopm.reg" % object)
    else:
        stars=Starlist("%s-allpm.reg" % object)
    if len(stars) < 5:
        print "Error retreiving SDSS reference stars"
        return -1
    else:
        return 1

############################################################################

def p60sdsssub(opts, inlis, refimage, ot, distortdeg=1, scthresh1=3.0, 
               scthresh2=10.0, tu=50000, iu=50000, ig=2.3, tg=1.0, 
               stamps=None, nsx=4, nsy=4, ko=0, bgo=0, radius=10, 
               tlow=0.0, ilow=None, sthresh=5.0, ng=None, aperture=10.0,
               distfile=P60DISTORT, defaultsDir = "defaults"):

    '''P60 Subtraction using SDSS image as reference'''

    #images = iraffiles(inlis)
    images = [inlis]
    images.sort()

    # Get WCS center, pixel scale, and pixel extent of reference
    [n1,n2]=get_head(refimage, ['NAXIS1','NAXIS2'])
    ractr,dcctr = get_radec_from_wcs(refimage)
    #[ractr,dcctr]=get_head(refimage, ['CRVAL1','CRVAL2'])
    #[[ractr,dcctr]]=impix2wcs(refimage,n1/2.0,n2/2.0)
    #if not (check_head(refimage,'PIXSCALE')):
    #    print 'Error: Please add PIXSCALE keyword to reference image'
    #    return 0
    #else:
    #    pix=get_head(refimage,'PIXSCALE')

    try:
        pix = get_head(refimage,['PIXSCALE'])[0]
    except:
        FILE0001 = get_head(refimage,['FILE0001'])[0]
        if ("ztf" in FILE0001) or ("PTF" in FILE0001):
            pix = 1.0
        elif "rings" in FILE0001:
            pix = 0.26
        elif "legacy" in FILE0001:
            pix = 0.30
        else:
            pix=0.396

    for image in images:

        root=image.split('.fits')[0]

        # Run scamp
        #p60scamp('%s.dist.fits' % root, refimage=refimage, 
        #p60scamp('%s.fits' % root, refimage=refimage, 
        #         distortdeg=distortdeg, scthresh1=scthresh1, 
        #         scthresh2=scthresh2,defaultsDir=defaultsDir)

        # Run Swarp
        #p60swarp('%s.dist.fits' % root, '%s.shift.fits' % root, ractr=ractr, 
        p60swarp(opts, '%s.fits' % root, '%s.shift.fits' % root, ractr=ractr,
                 dcctr=dcctr, pixscale=pix, size=[n1, n2], backsub=False)

        # Subtract
        if (ilow==None):
            ilow2=float(iskybkg) - 5 * math.sqrt(float(iskybkg))
        else:
            ilow2=ilow
        p60hotpants('%s.shift.fits' % root, refimage, '%s.sub.fits' % root,
                    tu=tu, iu=iu, ig=ig, tg=tg, ko=ko, bgo=bgo, nsx=nsx, 
                    nsy=nsy, radius=radius, tlow=tlow, ilow=ilow2, 
                    sthresh=sthresh, ng=ng, stamps=None, scimage=False)

        #p60hotpants(refimage, '%s.shift.fits' % root, '%s.sub2.fits' % root,
        #            tu=tu, iu=iu, ig=ig, tg=tg, ko=ko, bgo=bgo, nsx=nsx,
        #            nsy=nsy, radius=radius, tlow=tlow, ilow=ilow2,
        #            sthresh=sthresh, ng=ng, stamps=None, scimage=False)

        # Photometer subtracted image
        #iraf.phot('%s.sub.fits' % root, coords='ref.coo', output='%s.mag' % 
                  #root, epadu=ig, exposure='', calgorithm='none',
                  #salgorithm='median', annulus=30.0, dannulus=10.0, 
                  #weighting='constant', apertures=aperture, zmag=25.0, 
                  #interactive=no) 

    print "Exiting successfully"
    return
    
def p60scamp(opts, inlis, refimage=None, distortdeg=3, scthresh1=5.0, 
             scthresh2=10.0, match=False, cat="SDSS-R6", 
             defaultsDir = "defaults"):

    '''P60 Subtraction using scamp for alignment and HOTPANTS for
    subtraction.'''

    subimages=[inlis]

    # Create WCS catalog from reference image (if necessary)
    if not (refimage==None):
        refroot = refimage.replace(".fits","")
        ext = "fits"
        if not os.path.exists('%s.cat' % refroot):
            os.system('sex %s -c %s/default.sex -PARAMETERS_NAME %s/daofind.param -FILTER_NAME %s/default.conv' % (refimage,defaultsDir,defaultsDir,defaultsDir))
            shutil.move('test.cat', '%s.cat' % refroot)

    for image in subimages:

        # Extract image root
        root=image.split('.')[0]

        # Create FITS-LDAC file from SExtractor
        os.system('sex %s -c %s/default.sex -PARAMETERS_NAME %s/daofind.param -FILTER_NAME %s/default.conv' % (image,defaultsDir,defaultsDir,defaultsDir))
        shutil.move('test.cat', '%s.cat' % root)

        # Run scamp
        scampcmd="scamp %s.cat -DISTORT_DEGREES %i -SOLVE_PHOTOM N -SN_THRESHOLDS %f,%f -CHECKPLOT_DEV NULL " % (root, distortdeg, scthresh1, scthresh2)
        if refimage==None:
            scampcmd+="-ASTREF_CATALOG %s " % cat
        else:
            scampcmd+="-ASTREF_CATALOG FILE -ASTREFCENT_KEYS XWIN_WORLD,YWIN_WORLD -ASTREFCAT_NAME %s.cat -ASTREFERR_KEYS ERRAWIN_WORLD,ERRBWIN_WORLD,ERRTHETAWIN_WORLD -ASTREFMAG_KEY MAG_AUTO" % refroot
        if match:
            scampcmd+=" -MATCH Y -POSITION_MAXERR 6.0"
        else:
            scampcmd+=" -MATCH N"
        scmd=os.popen(scampcmd, 'r', -1)
        scmd.readlines()

    print "Exiting successfully"

def astrometrynet(imagefile,pixel_scale=0.18,ra=None, dec=None, radius=1.0,depth=None, index_xyls=None, ext=0, cutedges=0):

    if cutedges>0:
        hdulist=fits.open(imagefile)
        for ii in range(len(hdulist)):
            if hdulist[ii].data is None: continue
            hdulist[ii].data=hdulist[ii].data[cutedges:-cutedges,cutedges:-cutedges] 
        hdulist.writeto(imagefile,clobber=True)

    if not index_xyls == None:
        hdulist=fits.open(imagefile)
        xsize,ysize = hdulist[1].data.shape
        system_command = 'solve-field --odds-to-solve 10000.0 --guess-scale --no-plots --overwrite --scale-units arcsecperpix --scale-low %.5f --scale-high %.5f --ra %.5f --dec %.5f --radius %.5f --x-column XIMAGE --y-column YIMAGE --sort-ascending --sort-column MAG --width %d --height %d %s' % (pixel_scale/2.0,pixel_scale*2.0,ra,dec,radius,xsize,ysize,index_xyls)
        print(system_command)
    else:
        if not depth == None:
            if not ra == None:
                system_command = 'solve-field --guess-scale --no-plots --overwrite %s --scale-units arcsecperpix --scale-low %.5f --scale-high %.5f --ra %.5f --dec %.5f --radius %.5f --ext %d' % (imagefile,pixel_scale/2.0,pixel_scale*2.0,ra,dec,radius,ext)
            else:
                system_command = 'solve-field --guess-scale --no-plots --overwrite %s --scale-units arcsecperpix --scale-low %.5f --scale-high %.5f --ext %d' % (imagefile,pixel_scale/2.0,pixel_scale*2.0,ext)
        else:
            if not ra == None:
                system_command = 'solve-field --guess-scale --no-plots --overwrite %s --scale-units arcsecperpix --scale-low %.5f --scale-high %.5f --ra %.5f --dec %.5f --radius %.5f --ext %d'% (imagefile,pixel_scale/2.0,pixel_scale*2.0,ra,dec,radius,ext)
            else:
                system_command = 'solve-field --guess-scale --no-plots --overwrite %s --scale-units arcsecperpix --scale-low %.5f --scale-high %.5f --ext %d' % (imagefile,pixel_scale/2.0,pixel_scale*2.0,ext)
    os.system(system_command)

    try:
        shutil.move(imagefile.replace(".fits",".new"), imagefile)
    except:
        pass

def sextractor(imagefile,defaultsDir,doSubtractBackground=False,doPS1Params=False,zp=0.0,catfile=None,backfile=None):

    if catfile == None:
        catfile = imagefile.replace(".fits",".cat")
    if backfile == None:
        backfile = imagefile.replace(".fits",".background.fits")
    if doPS1Params:
        cmd_sex = 'sex %s -c %s/withPS1.sex -PARAMETERS_NAME %s/withPS1.param -FILTER_NAME %s/gauss_2.0_5x5.conv -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s -CATALOG_NAME %s -PSF_NAME %s/default.psf -STARNNW_NAME %s/default.nnw -MAG_ZEROPOINT %.5f'%(imagefile,defaultsDir,defaultsDir,defaultsDir,backfile,catfile,defaultsDir,defaultsDir,zp)
    else:
        cmd_sex = 'sex %s -c %s/default.sex -PARAMETERS_NAME %s/daofind.param -FILTER_NAME %s/default.conv -CHECKIMAGE_TYPE BACKGROUND -CHECKIMAGE_NAME %s -CATALOG_NAME %s -MAG_ZEROPOINT %.5f'%(imagefile,defaultsDir,defaultsDir,defaultsDir,backfile,catfile,zp)    
    os.system(cmd_sex)

    if doSubtractBackground:
        hdulist=fits.open(imagefile)
        hdulistback=fits.open(backfile)

        for ii in range(len(hdulist)):
            if hdulist[ii].data is None: continue
            hdulist[ii].data=hdulist[ii].data-hdulistback[ii].data
        hdulist.writeto(imagefile,clobber=True)        

def utcparser(utcstart):
        """
        Datetime parser for CHIMERA UTCSTART header keyword.

        Parameters
        ----------
        utcstart : string
            Datetime for start of frame (in UTC)

        Returns
        -------
        dt : datetime struct
            Datetime structure
        """

        MONTHS = {"Jan": 1, "Feb": 2, "March": 3, "Mar": 3, "April": 4, "May": 5, "June": 6, "July": 7, "Aug": 8, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}

        month, date, year, time = utcstart.split("-")
        month = MONTHS[month]
        date = int(date)
        year = int(year)

        hour, minu, sec = time.split(":")
        hour = int(hour)
        minu = int(minu)
        sec, ms = sec.split(".")
        sec = int(sec)
        ms = int(ms) * 1000

        dt = datetime.datetime(year, month, date, hour, minu, sec, ms)

        return Time(dt)

def forcedphotometry(imagefile,ra=None,dec=None,x=None,y=None,fwhm=5.0,zp=0.0,gain=1.0):

    hdulist=fits.open(imagefile)
    header = fits.getheader(imagefile)

    if x == None:
        w = WCS(header)
        x0,y0 = w.wcs_world2pix(ra,dec,1)
        gain = 1.0
    else:
        x0,y0 = x, y

    if len(hdulist) > 3:
       image = hdulist[1].data
    elif len(hdulist) == 2:
       image = hdulist[0].data
    else:
       image = hdulist[0].data
    image_shape = image.shape

    forcedfile = imagefile.replace(".fits",".forced")
    fid = open(forcedfile,'w')

    if len(image_shape) == 3:

        nhdu, xshape, yshape = image.shape
        dateobs = utcparser(hdulist[0].header["UTCSTART"])
        mjd = dateobs.mjd

        if "KINCYCTI" in hdulist[0].header:
            mjdall = mjd + np.arange(nhdu)*hdulist[0].header["KINCYCTI"]/86400.0
        else:
            mjdall = mjd + np.arange(nhdu)*hdulist[0].header["EXPTIME"]/86400.0

        mjds, mags, magerrs, fluxes, fluxerrs = [], [], [], [], []
        for jj in range(nhdu):
            image = hdulist[0].data[jj,:,:]
            mjd = mjdall[jj]

            mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = pp.aper.aper(image,x0,y0,phpadu=gain,apr=fwhm,zeropoint=zp,skyrad=[3*fwhm,5*fwhm],exact=False)

            mjds.append(mjd)
            mags.append(mag)
            magerrs.append(magerr)
            fluxes.append(flux)
            fluxerrs.append(fluxerr)

            fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(dateobs.mjd,mag,magerr,flux,fluxerr))
        fid.close()

        return np.array(mjds), np.array(mags), np.array(magerrs), np.array(fluxes), np.array(fluxerrs)

    elif len(hdulist) == 1:
        header = hdulist[0].header
        image = hdulist[0].data
        exptime = header["EXPTIME"]

        try:
            mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = pp.aper.aper(image,x0,y0,phpadu=gain,apr=fwhm,zeropoint=zp,skyrad=[3*fwhm,5*fwhm],exact=False,exptime=exptime)    
        except:
            mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = pp.aper.aper(image,x0,y0,phpadu=gain,apr=fwhm,zeropoint=zp,skyrad=[3*fwhm,5*fwhm],exact=False)

        if "UTCSTART" in header:
            dateobs = utcparser(header["UTCSTART"])
            mjd = dateobs.mjd
        else:
            mjd = -1

        return mjd, mag, magerr, flux, fluxerr

    elif len(hdulist) == 2:
        header = hdulist[0].header
        image = hdulist[0].data

        mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = pp.aper.aper(image,x0,y0,phpadu=gain,apr=fwhm,zeropoint=zp,skyrad=[3*fwhm,5*fwhm],exact=False)
        if "DATE" in header:
            dateobs = Time(header["DATE"])
            mjd = dateobs.mjd
        else:
            mjd = -1

        return mjd, mag, magerr, flux, fluxerr

    elif len(hdulist) == 3:
        header = hdulist[0].header
        image = hdulist[0].data

        mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = pp.aper.aper(image,x0,y0,phpadu=gain,apr=fwhm,zeropoint=zp,skyrad=[3*fwhm,5*fwhm],exact=False)
        if "DATE" in header:
            dateobs = Time(header["DATE"])
            mjd = dateobs.mjd
        else:
            mjd = -1

        return mjd, mag, magerr, flux, fluxerr

    else: 
        mjds, mags, magerrs, fluxes, fluxerrs = [], [], [], [], []
        for ii, hdu in enumerate(hdulist):
            if ii == 0: continue
            header = hdulist[ii].header
            image = hdulist[ii].data
            if not "DATE" in header:
                print("Warning: 'DATE missing from %s hdu %d/%d"%(imagefile,ii,len(hdulist)))
                continue

            dateobs = Time(header["DATE"])
            mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = pp.aper.aper(image,x0,y0,phpadu=gain,apr=fwhm,zeropoint=zp,skyrad=[3*fwhm,5*fwhm],exact=False)

            mjds.append(dateobs.mjd)
            mags.append(mag)
            magerrs.append(magerr)
            fluxes.append(flux)
            fluxerrs.append(fluxerr)

            fid.write('%.5f %.5f %.5f %.5f %.5f\n'%(dateobs.mjd,mag,magerr,flux,fluxerr))
        fid.close()

        return np.array(mjds), np.array(mags), np.array(magerrs), np.array(fluxes), np.array(fluxerrs)


def get_links(minday = -1, day = None):
    links = []

    url = "https://ztfweb.ipac.caltech.edu/ztf/archive/sci/2017/"
    doc = fromstring(requests.get(url,auth=(os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"])).content)
    doc.make_links_absolute(base_url=url)
    for l in doc.iterlinks():
        if not l[0].tag == 'a': continue
        if "?" in l[2]: continue
        if not url in l[2]: continue
        url2 = l[2]
        thisday = int(url2.split("/")[-2])
        if thisday < minday: continue
        if not day == None:
            if not thisday == day: continue

        doc2 = fromstring(requests.get(url2,auth=(os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"])).content)
        doc2.make_links_absolute(base_url=url2)
        for m in doc2.iterlinks():
            if not m[0].tag == 'a': continue
            if "?" in m[2]: continue
            if not url2 in m[2]: continue
            url3 = m[2]
            doc3 = fromstring(requests.get(url3,auth=(os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"])).content)
            doc3.make_links_absolute(base_url=url3)
            for n in doc3.iterlinks():
                url4 = n[2]
                if not "sciimg.fits" in url4: continue
                links.append(url4)
    return links

def get_fits_from_link(link):

    links = []
    doc3 = fromstring(requests.get(link,auth=(os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"])).content)
    doc3.make_links_absolute(base_url=link)
    for n in doc3.iterlinks():
        url4 = n[2]
        if not "sciimg.fits" in url4: continue
        if link not in url4: continue
        links.append(url4)
    return links

def download_images_from_links(links,ztfDir):

    raimages, decimages = [], []
    for image in links:
        imageSplit = image.split("/")
        year, day = imageSplit[-4], imageSplit[-3]
        if not os.path.isdir('%s/%s%s'%(ztfDir,year,day)):
            os.makedirs('%s/%s%s'%(ztfDir,year,day))

        imagefinal = '%s/%s%s/%s'%(ztfDir,year,day,imageSplit[-1])
        if not os.path.isfile(imagefinal):
            wget_command = "wget %s --user %s --password %s -O %s"%(image,os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"],imagefinal)
            os.system(wget_command)

        if not os.path.isfile(imagefinal): continue

        raimage, decimage = get_radec_from_wcs(imagefinal)
        raimages.append(raimage)
        decimages.append(decimage)

    return raimages, decimages

