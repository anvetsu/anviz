"""
julia.py - Implement the Julia set in Python using matplotlib & numpy

Ref: http://localhost:1313/posts/fractals_with_python1/

"""

import matplotlib.pyplot as plt
import numpy as np

def julia_set(width, height, zoom=1, niter=256):
    """ A julia set of geometry (wxh) and iterations 'niter' """

    w,h = width, height
    pixels = np.arange(w*h, dtype=np.uint16).reshape(h, w)

    # Pick some defaults for the real and imaginary constants
    # This determines the shape of the Julia set.
    c_real, c_imag = -0.7, 0.27
    
    for x in range(w): 
        for y in range(h):
            # calculate the initial real and imaginary part of z,
            # based on the pixel location and zoom and position values
            zx = 1.5*(x - w/2)/(0.5*zoom*w) 
            zy = 1.0*(y - h/2)/(0.5*zoom*h)

            for i in range(niter):
                radius_sqr = zx*zx + zy*zy
                # Iterate till the point is outside
                # the circle with radius 2.             
                if radius_sqr > 4: break
                # Calculate new positions
                zy,zx = 2.0*zx*zy + c_imag, zx*zx - zy*zy + c_real

            color = (i >> 21) + (i >> 10)  + i * 8
            pixels[y,x] = color
  
    # to display the created fractal 
    plt.imshow(pixels)
    plt.show()
    
if __name__ == "__main__":
    julia_set(1920, 1440)
