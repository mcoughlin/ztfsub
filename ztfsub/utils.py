#import pyraf
#from pyraf import iraf
import copy, os, shutil, glob, sys, string, re, math, operator, numpy, time
#import pyfits
from types import *

import numpy as np

from astropy.io import fits

import requests
from lxml.html import fromstring

P60DISTORT = "P60distort.head"
SUBREGION = "[1:1024,1:1024]"

######################################################################

def get_head(imagefile,keywords):

    hdulist=fits.open(imagefile)
    header = hdulist[0].header
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

        cmd=os.popen(scmd,'r')
        hlines=cmd.readlines()
        cmd.close()

    return

############################################################################

def p60swarp(image, outfile, ractr=None, dcctr=None, pixscale=None, size=None,
             backsub=False):

    '''Run SWarp on P60 images'''

    swarpcmd='swarp %s ' % image
    if ractr!=None or dcctr!=None:
        swarpcmd+='-CENTER_TYPE MANUAL -CENTER "%s, %s" ' % (ractr, dcctr)
    if pixscale!=None:
        swarpcmd+='-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE %f ' % pixscale
    if size!=None:
        swarpcmd+='-IMAGE_SIZE "%i, %i" ' % (size[0], size[1])
    if backsub==False:
        swarpcmd+='-SUBTRACT_BACK N '
    swarpcmd+='-COPY_KEYWORDS OBJECT,SKYSUB,SKYBKG,SEEPIX '

    scmd=os.popen(swarpcmd, 'r', -1)
    scmd.readlines()
    shutil.move('coadd.fits', outfile)

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

def p60sdsssub(inlis, refimage, ot, distortdeg=1, scthresh1=3.0, 
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
    [ractr,dcctr]=get_head(refimage, ['CRVAL1','CRVAL2'])
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
        if "ztf" in FILE0001:
            pix = 1.0
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
        p60swarp('%s.fits' % root, '%s.shift.fits' % root, ractr=ractr,
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
    
def p60scamp(inlis, refimage=None, distortdeg=3, scthresh1=5.0, 
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

def astrometrynet(imagefile):

    os.system('solve-field --no-plots --overwrite %s' % (imagefile))
    shutil.move(imagefile.replace(".fits",".new"), imagefile)

def sextractor(imagefile,defaultsDir):

    catfile = imagefile.replace(".fits",".cat")
    cmd_sex = 'sex %s -c %s/default.sex -PARAMETERS_NAME %s/daofind.param -FILTER_NAME %s/default.conv'%(imagefile,defaultsDir,defaultsDir,defaultsDir)
    os.system(cmd_sex)
    mv_command = 'mv test.cat %s'%(catfile)
    os.system(mv_command)

def get_links():
    links = []

    url = "https://ztfweb.ipac.caltech.edu/ztf/archive/sci/2017/"
    doc = fromstring(requests.get(url,auth=(os.environ["ZTF_USERNAME"],os.environ["ZTF_PASSWORD"])).content)
    doc.make_links_absolute(base_url=url)
    for l in doc.iterlinks():
        if not l[0].tag == 'a': continue
        if "?" in l[2]: continue
        if not url in l[2]: continue
        url2 = l[2]
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


