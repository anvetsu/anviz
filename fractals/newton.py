"""
newton.py - Implement fractals using newton-raphson approximation method
of solving functions.

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import functools
import multiprocessing as mp

from PIL import Image

def rgb2int(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red << 16) + (green << 8) + blue
    return RGBint

def int2rgb(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return (red, green, blue)

# put any complex function here to generate a fractal for it!
def fcube(z):
    return z ** 3 - 1

def f8(z):
    return z**8 + 15*z**4 - 16

def sin(z):
    return math.sin(abs(z))

def tan(z):
    return math.tan(abs(z))

def cot(z):
    return 1.0/math.tan(abs(z))

def fcube2(z):
    return z**3 - 2*z

def fsquare(z):
    return z**2 - 1

def fsix(z):
    return z**6  + z**3 - 1

def fcomplex(z):
    return z**(4+3j) - 1

def ftrig1(z):
    return math.cos(abs(z)) - math.sin(abs(z))

def newton_set_calc_row(y, width, height, function, niter=256, zoom=1):
    """ Calculate one row of the julia set with size wxh """

    row_pixels = np.arange(width, dtype=np.uint32)
    # drawing area
    xa, xb, ya, yb = -2.0, 2.0, -2.0, 2.0

    zy = y * (yb - ya) / (zoom*(height - 1)) + ya   
    
    h = 1e-7 # step size for numerical derivative
    eps = 1e-7 # max error allowed
    a = complex(1, 0)
    
    for x in range(width): 
        # calculate the initial real and imaginary part of z,
        # based on the pixel location and zoom and position values
            zx = x * (xb - xa) / (zoom*(width - 1)) + xa
            z = complex(zx, zy)
            
            for i in range(niter):
                # complex numerical derivative
                dz = (function(z + complex(h, h)) - function(z)) / complex(h, h)
                if dz == 0:
                    break

                z0 = z - a*function(z) / dz # Newton iteration
                if abs(z0 - z) < eps: # stop when close enough to any root
                    break
                
                z = z0

            rgb = (i % 32 * 32, i % 16 * 16, i % 32 * 8)
            row_pixels[x] = rgb2int(rgb)


    return y,row_pixels

# draw the fractal
def newton_set(width, height, function, zoom=1, niter=256):
    """ Fractals using newton-raphson """
    
    # drawing area
    xa, xb, ya, yb = -2.0, 2.0, -2.0, 2.0
    # Image object
    pimg = Image.new("RGB", (width, height))
    pixels = np.arange(width*height*3, dtype=np.uint32).reshape(height, width, 3)
    
    h = 1e-6 # step size for numerical derivative
    eps = 1e-5 # max error allowed
    a = complex(1, 1)
    
    for y in range(height):
        zy = y * (yb - ya) / (zoom*(height - 1)) + ya

        for x in range(width):
            zx = x * (xb - xa) / (zoom*(width - 1)) + xa
            z = complex(zx, zy)
            flag  = True # False
            
            for i in range(niter):
                # complex numerical derivative
                dz = (function(z + complex(h, h)) - function(z)) / complex(h, h)
                if dz == 0:
                    break

                z0 = z - a*function(z) / dz # Newton iteration
                if abs(z0 - z) < eps: # stop when close enough to any root
                    flag = True
                    break
                
                z = z0

            # schemes
            # 1. nice bluish tinge
            # rgb = (n % 8 * 4, n % 16 * 8, n % 32 * 16)
            # 2. reddish green
            # rgb = (n % 8 * 32, n % 16 * 16, n % 32 * 8)
            # 3. very dark red and green
            # rgb = (n % 8 * 32, n % 16 * 8, n % 32 * 2)
            # 4. blue fractal, green red curves
            # rgb = (n % 4 * 64, n % 8 * 32, n % 16 * 16)
            # 5. reddish pink fractal with blue curves
            # rgb = (n % 64 * 16, n % 16 * 8, n % 8 * 32)
            # 6. Greenish n red fractals, blue in between
            # rgb = (n % 32 * 4, n % 16 * 8, n % 4 * 16)
            # 7. Blue n pink
            # rgb = (n % 8 * 32, n % 16 * 16, n % 64 * 32)
            # 8. Bluish black
            # rgb = (n % 8 * 8, n % 8 * 16, n % 16 * 32)
            # 9. Black n white
            # rgb = (n % 8 * 32, n % 8 * 32, n % 16 * 32)
            # 10. Mix of colors evenly
            # rgb = (n % 8 * 64, n % 8 * 32, n % 16 * 32)
            # 11. Very greenish red
            # rgb = (n % 32 * 64, n % 8 * 32, n % 8 * 16)
            
            # rgb = (n % 4 * 64, n % 8 * 32, n % 16 * 16)
            rgb = (i % 8 * 64, i % 8 * 32, i % 16 * 32)
            
            pimg.putpixel((x,y), rgb)
            pixels[y,x] = rgb
            # return pimg
        
    return pixels

def newton_set_mp(width, height, function, zoom=1, niter=256):
    """ Newton-raphson fractal set with multiprocessing """
    
    w,h = width, height
    pixels = np.arange(w*h*3, dtype=np.uint32).reshape(h, w, 3)  

    # print('Starting calculation using',width, height,cx,cy)
    pool = mp.Pool(mp.cpu_count())

    newton_partial = functools.partial(newton_set_calc_row, 
                                      width=width,height=height, function=function,
                                      niter=niter,zoom=zoom)

    for y,row_pixel in pool.map(newton_partial, range(h)):
        for x in range(w):
            pixels[y, x] = np.array(int2rgb(row_pixel[x]))

    return pixels
        
def display(width=1024, height=1024, niter=120, zoom=1, cmap='viridis'):
    """ Display a newton-raphson fractal """

    pimg = newton_set_mp(width, height, fcube, zoom=zoom, niter=niter)
    plt.axis('off') 
    plt.imshow(pimg, cmap=cmap) 
    plt.show()

if __name__ == "__main__":
    display(cmap='magma')
