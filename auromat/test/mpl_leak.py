# Copyright European Space Agency, 2013

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

# see https://github.com/matplotlib/matplotlib/issues/3197

# mem leak only occurs when backend is agg, rasterized=True, 
# and when savefig is called with SVG! for png there's no leak

for i in range(10000):
    fig, ax = plt.subplots()            

    circle1=plt.Circle((0,0),2,color='r',rasterized=True)
    ax.add_artist(circle1)

    fig.savefig('test.svg')
    plt.close(fig)
    