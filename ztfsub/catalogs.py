
import numpy as np

from astropy.io import fits

from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.vizier import Vizier

def get_glade(ra,dec,rawidth,decwidth):

    result = Vizier.query_region(SkyCoord(ra=ra, dec=dec,
                                     unit=(u.deg, u.deg),
                                     frame='icrs'),
                                     width=[rawidth*u.deg,decwidth*u.deg],
                                     catalog=['VII/275'])

    for table_name in result.keys():
        table = result[table_name]
        appMs = table["Bmag"] # - (5*np.log10(table["Dist"]*1e6) - 5)

        idxs = np.where((table["Dist"] > 40.0) & (table["Dist"] < 300.0))[0]

    return table["RAJ2000"][idxs], table["DEJ2000"][idxs]

def get_clu(ra,dec,rawidth,decwidth,dataDir):

    datafile = "%s/clu/CLU_20170106_galexwise_DaveUpdate.fits"%dataDir
    hdulist=fits.open(datafile)
    header = hdulist[1].header
    data = hdulist[1].data

    ras, decs = [], []
    for row in data:
        ragalaxy, decgalaxy, distgalaxy = row[3], row[4], row[109]
        if (distgalaxy > 40.0) & (distgalaxy < 300.0):
            ras.append(ragalaxy)
            decs.append(decgalaxy)
    return ras, decs
