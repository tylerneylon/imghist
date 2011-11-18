imghist.py
==========

This is a python script to generate hue histograms
and color pies from color png images.


installation
------------

1. Install pypng from here:

http://code.google.com/p/pypng/downloads/list

2. Install pycairo from here:

http://cairographics.org/pycairo/

3. Download or clone the python script from this github page:

https://github.com/tylerneylon/imghist


usage
-----

    ./imghist.py [options] <input png> <output png>

For example, if you have a file named myImg.png, you could generate a
color pie with surrounding hue histogram with this command:

    ./imghist.py -mboth -s600x600 myImg.png myHueHist.png

For more usage details, type:

    ./imghist.py -h

