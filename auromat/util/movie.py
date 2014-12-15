# Copyright European Space Agency, 2013

"""
A little helper module to create movies out of images using ffmpeg.
"""

from __future__ import absolute_import, print_function

import os
import errno
import tempfile
import shutil
import subprocess

def createMovie(moviePath, imagePaths, frameRate=25, crf=18, maxBitrate=2000, width=None, height=None,
                tempFolder=None):
    """
    
    :param moviePath: path with .mp4 or .webm ending, will be overridden if existing
    :param imagePaths: paths of images in correct order
    :param frameRate: the frame rate of the movie
    :param crf: -crf parameter of ffmpeg, lower is better
    :param maxBitrate: maximum bitrate in kbit/s
    :param width, height: width and height of movie in pixels; if not given, then image dimensions are used
    :param tempFolder: the folder to store temporary data in (symlinks), will be removed afterwards
    """
    if not tempFolder:
        tempFolder = tempfile.mkdtemp()
    elif not os.path.exists(tempFolder):
        os.makedirs(tempFolder)
    
    ext = os.path.splitext(imagePaths[0])[1]
    framePaths = [os.path.join(tempFolder, str(frame) + ext) for frame in range(len(imagePaths))]
    
    movieExt = os.path.splitext(moviePath)[1]
    
    try:
        # create symlinks in the temporary folder to the input images
        for imagePath, framePath in zip(imagePaths, framePaths):     
            # TODO doesn't work on Windows       
            os.symlink(imagePath, framePath)
        
        # run ffmpeg
        args = ['ffmpeg', '-y', '-r', str(frameRate), '-i', os.path.join(tempFolder, '%d' + ext),
                '-r', str(frameRate)]
        
        if width or height:
            if width and height:
                s = '{}:{}'.format(width, height)
            elif width:
                # makes sure that height is divisible by 2
                s = '{}:trunc(ow/a/2)*2'.format(width)
            elif height:
                # makes sure that width is divisible by 2
                s = 'trunc(oh/a/2)*2:{}'.format(height)
            args += ['-vf', 'scale=' + s]
        
        if movieExt == '.mp4':
            args += ['-codec:v', 'libx264', '-preset', 'slow', '-profile:v', 'baseline',
                     '-pix_fmt', 'yuv420p', '-movflags', 'faststart'] # faststart needs ffmpeg >= 1.0
            
        elif movieExt == '.webm':
            args += ['-codec:v', 'libvpx', '-quality', 'good', '-cpu-used', '0',
                     '-b:v', str(maxBitrate) + 'k', '-qmin', '10', '-qmax', '42']
        else:
            raise NotImplementedError('Unsupported video format: ' + ext)
        
        args += ['-crf', str(crf), '-maxrate', str(maxBitrate) + 'k', 
                 '-bufsize', str(maxBitrate*2) + 'k']
        args += [moviePath]
        
        print('running ' + ' '.join(args))
                
        try:
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise RuntimeError('ffmpeg could not be launched. Make sure it is in the PATH!')
            else:
                raise
        
        stdout, _ = process.communicate()
            
        if process.returncode != 0:
            raise RuntimeError('ffmpeg returned exit code ' + str(process.returncode) + '; cmd line: ' + ' '.join(args) + '; ' +
                               'stdout&stderr output follows: ' + stdout.decode('ascii'))
    finally:
        shutil.rmtree(tempFolder)
    