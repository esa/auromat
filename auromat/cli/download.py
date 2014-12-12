# Copyright European Space Agency, 2013

"""
Downloads a dataset from one of the following providers:

- ESA ISS Archive
- THEMIS Archive
"""

from __future__ import absolute_import, print_function
import argparse
from datetime import datetime
import os
import sys

__all__ = ['main']

def date(datestring):
    return datetime.strptime(datestring, '%Y-%m-%dT%H:%M:%S')

class Providers(object):
    ESA_ISS = 'esa-iss'
    THEMIS = 'themis'

description = '''
This tool downloads data of a given period from a provider and
stores it on disk without applying any postprocessing.
'''

epilog = '''
Examples:

auromat-download esa-iss --id 5
auromat-download esa-iss --id 5 --start 2000-01-01T12:00:00
auromat-download esa-iss --id 5 --id 6 --id 7
auromat-download esa-iss --id 5 --dir sequences/iss

auromat-download themis --start 2000-01-01T12:00:00 --end 2000-01-01T12:10:00
'''

def getParser():
    parser = argparse.ArgumentParser(prog='auromat-download', 
                                     epilog=epilog, description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('provider', help='The provider to get data from', 
                        choices=[Providers.ESA_ISS, Providers.THEMIS])
    period = parser.add_argument_group('period', 
                                       'These arguments specify which data to download.\n'
                                       'For the THEMIS provider, --start and --end are required.\n'
                                       'For the ESA ISS provider, --id is required and \n'
                                       '--start --end can be used to further constrain the period.')
    period.add_argument('--start', help='UTC start date, format 2000-01-01T12:00:00', type=date)
    period.add_argument('--end', help='UTC end date (inclusive)', type=date)
    period.add_argument('--id', help='Sequence ID(s) in ESA ISS archive', type=int, action='append')
    
    parser.add_argument('--dir', help='Directory to store data in, by default the current one', default=os.getcwd())
    parser.add_argument('--version', action='version', version='auromat TODO')
    return parser

def parseargs():
    parser = getParser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    
    if args.provider == Providers.THEMIS:
        if not (args.start and args.end):
            parser.error('--start and --end are required for the THEMIS provider')
    
    if args.provider == Providers.ESA_ISS:
        if not args.id:
            parser.error('--id is required for the ESA ISS provider')
    return args

def main():
    args = parseargs()
    
    if args.provider == Providers.ESA_ISS:
        from auromat.mapping.iss import ISSMappingProvider
        providers = [ISSMappingProvider(cacheFolder=os.path.join(args.dir, 'seq-{}'.format(id_)),
                                        id_=id_)
                     for id_ in args.id]
    
    elif args.provider == Providers.THEMIS:
        from auromat.mapping.themis import ThemisMappingProvider
        providers = [ThemisMappingProvider(cdfL1CacheFolder=args.dir, 
                                           cdfL2CacheFolder=args.dir)]
    
    for provider in providers:
        provider.download(args.start, args.end)
    
    print('Done.')
    
main.__doc__ = """
::
  
  {}
  

""".format(getParser().format_help().replace('\n', '\n  '))
    
if __name__ == '__main__':
    main()
    