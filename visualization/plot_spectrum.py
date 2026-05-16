import numpy as np
import matplotlib
from matplotlib import pyplot as plt
def wavelength_to_rgb(wavelength: float, gamma: float=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


_visible_range_plus = (350,780)
_norm = plt.Normalize(*_visible_range_plus)
_wl = np.arange(_visible_range_plus[0],_visible_range_plus[1]+1,2)
_colorlist = list(zip(_norm(_wl),[wavelength_to_rgb(w) for w in _wl]))
spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", _colorlist)


def plot_spectrum(x, y=None, lo=400, hi=700, ylabel=None):
    if y is None:
        y = x
        interval = (hi-lo)//(len(spectrum)-1)
        x = np.arange(lo, hi+interval, interval)
    plt.plot(x, y, color='darkred')

    y_max = max(1, max(y))
    y_steps = np.linspace(0, y_max, 100)
    X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), y_steps)

    extent = (min(x), max(x), min(y_steps), y_max)

    plt.imshow(X, clim=_visible_range_plus, extent=extent, cmap=spectralmap, aspect='auto')
    plt.xlabel('$\lambda$ (Wavelength nm)')
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.fill_between(x, y, y_max, color='w')
