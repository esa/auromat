#Copyright (c) 2012, Tolga Birdal
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the distribution
#    * Neither the name of the Gravi Information Technologies and Consultancy Ltd nor the names
#      of its contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#POSSIBILITY OF SUCH DAMAGE.

from math import sqrt, pi
import numpy as np
from scipy.ndimage.filters import correlate1d


def _estimateNoiseLevel(imgray):
    """
    Estimates the noise level of the given one-channel image.
    
    This code is adapted from
    http://www.mathworks.com/matlabcentral/fileexchange/36941-fast-noise-estimation-in-images
    
    The original description follows:
    
        by Tolga Birdal
        
        This is an extremely simple m-file which implements the method described
        in : J. Immerkaer, "Fast Noise Variance Estimation", Computer Vision and
        Image Understanding, Vol. 64, No. 2, pp. 300-302, Sep. 1996
                
        The advantage of this method is that it includes a Laplacian operation 
        which is almost insensitive to image structure but only depends on the 
        noise in the image. 
    """    
    h,w = imgray.shape[0], imgray.shape[1]
    
    # compute sum of absolute values of Laplacian
    kernel = np.array([1,-2,1])
    conv = correlate1d(imgray.astype(np.float), kernel, axis=0)
    conv = correlate1d(conv, kernel, axis=1)
    sigma = np.sum(np.abs(conv))
    
    # scale sigma with proposed coefficients
    sigma = sigma*sqrt(0.5*pi)/(6*(w-2)*(h-2))
    return sigma