# Copyright European Space Agency, 2013

"""
Converts a downloaded dataset into a given format (CDF and netCDF currently)
and applies requested postprocessing steps like resampling.
"""

from __future__ import absolute_import, print_function
from six.moves import map
import argparse
from datetime import datetime
import os
import sys
from functools import partial

from auromat.mapping.themis import ThemisMappingProvider
from auromat.mapping.iss import ISSMappingProvider
from auromat.resample import resample, resampleMLatMLT
from auromat.util.os import makedirs

__all__ = ['main']

def date(datestring):
    return datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S')

description = '''
This tool converts downloaded data to a new format and optionally
applies postprocessing like resampling.
'''

epilog = '''
Examples:

Convert everything in the current directory with default settings:
auromat-convert --format cdf

Convert data in iss/seq-1 folder with 100km mapping altitude:
auromat-convert --data iss/seq-1 --format cdf --altitude 100

Resample and convert data:
auromat-convert --format cdf --resample --resolution 80

Don't store pixel corner coordinates:
auromat-convert --format cdf --without-bounds

Note that if you called auromat-download with --start and/or --end
dates, then you have to specify these here as well.
'''

class Grid(object):
    geo = 'geo'
    mag = 'mag'
    
class Format(object):
    cdf = 'cdf'
    netcdf = 'netcdf'

def getParser():
    parser = argparse.ArgumentParser(prog='auromat-convert', 
                                     epilog=epilog, description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', help='Data directory, by default the current directory. '
                                       'For ESA ISS sequences, this must be the subfolder '
                                       'corresponding to a single sequence, typically "seq-n". '
                                       'Under Unix-type shells, use "--data seq-*" to process '
                                       'all sequences.', 
                        default=os.getcwd())
    
    period = parser.add_argument_group('period', 
                                       'These arguments optionally specify which data to convert. '
                                       'If you used --start and/or --end in auromat-download, '
                                       'then the same values have to be used here as well.')
    period.add_argument('--start', help='UTC start date, format 2000-01-01T12:00:00', type=date)
    period.add_argument('--end', help='UTC end date (inclusive)', type=date)
    
    mappingArgs = parser.add_argument_group('mapping')
    mappingArgs.add_argument('--altitude', help='Altitude in km onto which to map the images, '
                                                'default is 110km', default=110)
    
    esaIssArgs = parser.add_argument_group('ESA ISS data')
    esaIssArgs.add_argument('--bps', help='bits per sample, default is 16', choices=[8,16], default=16)
    esaIssArgs.add_argument('--correctgamma', 
                            help='Applies BT.709 gamma correction for creating visually '
                                 'pleasing images. A nonlinear curve is applied which '
                                 'brightens dark pixels more than bright ones to match '
                                 'human perception. If --correctgamma is not used then '
                                 'a more scientifically useful linear image is produced '
                                 'where the number of photons hitting a pixel is in linear '
                                 'relationship with the resulting pixel value.', 
                            action='store_true')
    esaIssArgs.add_argument('--autobright', 
                            help='Automatically brightens the image such that 1%% of all pixels '
                                 'are fully saturated. This may destroy information in the image '
                                 'and should not be used when doing a scientific analysis.',
                            action='store_true')
    
    resampleArgs = parser.add_argument_group('resampling')
    resampleArgs.add_argument('--resample', help='Whether to resample or not', action='store_true')
    resampleArgs.add_argument('--resolution', metavar='RES', help='in arcsec/px, default 100', default=100, type=float)
    resampleArgs.add_argument('--grid', 
                              help='The grid which will be regular after resampling. '
                                   'Default is MLat/MLT grid. Use geo for geographical grid.', 
                              default=Grid.mag, choices=[Grid.geo, Grid.mag])

    outputArgs = parser.add_argument_group('output')
    outputArgs.add_argument('--out', 
                            help='Output directory, by default the "converted" '
                                 'subdirectory of --data')
    outputArgs.add_argument('--overwrite', 
                            help='Overwrites existing files.',
                            action='store_true')
    outputArgs.add_argument('--skip',
                            help='Skips already converted files.',
                            action='store_true')
    outputArgs.add_argument('--format', help='Data format of converted files', 
                            choices=[Format.cdf, Format.netcdf], required=True)
    outputArgs.add_argument('--without-bounds', dest='withoutBounds',
                            help='Do not include coordinates of pixel corners. '
                                 'If set, then only the pixel center coordinates are written, '
                                 'otherwise both.',
                            action='store_true')
    outputArgs.add_argument('--without-mag', dest='withoutMag',
                            help='Do not include MLat/MLT coordinates.', 
                            action='store_true')
    outputArgs.add_argument('--without-geo', dest='withoutGeo',
                            help='Do not include geodetic coordinates. Only usable with CDF output.', 
                            action='store_true')
    
    parser.add_argument('--version', action='version', version='auromat TODO')
    return parser

def parseargs():
    parser = getParser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    if not args.out:
        args.out = os.path.join(args.data, 'converted')
    if args.overwrite and args.skip:
        parser.error('only one of --overwrite and --skip is allowed')
    if args.withoutGeo and args.format == Format.netcdf:
        parser.error('--without-geo is only usable with --format cdf')
    
    return args

def main():
    args = parseargs()
        
    # first, we try to figure out from which provider the data came from
    dataPath = args.data
    dataFiles = os.listdir(dataPath)
    if any(f == 'api.json' for f in dataFiles):
        provider = ISSMappingProvider(dataPath, altitude=args.altitude,
                                      noRawPostprocessCaching=True,
                                      raw_bps=args.bps, raw_auto_bright=args.autobright,
                                      raw_gamma=None if args.correctgamma else (1,1),
                                      offline=True)
    
    elif any(f.startswith('thg_l1_') for f in dataFiles):
        if args.format == Format.cdf:
            print ('Note that THEMIS files are already in CDF format where each file '
                   'contains 1h of images. With this script a CDF file for each image '
                   'is created.')
        if not (args.start and args.end):
            print('For THEMIS data you have to specify --start and --end')
            sys.exit(1)
                        
        provider = ThemisMappingProvider(dataPath, dataPath, altitude=args.altitude,
                                         offline=True)
    
    else:
        raise NotImplementedError('Not recognized as THEMIS or ESA ISS data')
    
    mappings = provider.getSequence(args.start, args.end)
    
    if args.resample:
        if args.grid == Grid.geo:
            resample_ = resample
        elif args.grid == Grid.mag:
            resample_ = resampleMLatMLT
        resample_ = partial(resample_, arcsecPerPx=args.resolution)
            
        mappings = map(resample_, mappings)
    
    if args.format == Format.cdf:
        import auromat.export.cdf
        export = auromat.export.cdf.write
        ext = '.cdf'
        
    elif args.format == Format.netcdf:
        import auromat.export.netcdf
        export = auromat.export.netcdf.write
        ext = '.nc'
    
    export = partial(export, includeBounds=not args.withoutBounds, includeMagCoords=not args.withoutMag,
                     includeGeoCoords=not args.withoutGeo)
        
    makedirs(args.out)
    for mapping in mappings:
        path = os.path.join(args.out, mapping.identifier + ext)
        if os.path.exists(path):
            if args.skip:
                print('skipping', path)
                continue
            elif args.overwrite:
                os.remove(path)
            else:
                print('The file', path, 'already exists.\n'
                      'Please use --skip or --overwrite, or a different output folder.',
                      file=sys.stderr)
                sys.exit(1)
            
        print('storing', path)
        export(path, mapping)
    
    print('Done.')

main.__doc__ = """
::
  
  {}
  

""".format(getParser().format_help().replace('\n', '\n  '))

if __name__ == '__main__':
    main()
    