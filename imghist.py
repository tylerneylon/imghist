#!/usr/bin/python
#
# imghist.py
#
# Usage: ./imghist.py [options] <input png> <output png>
#
# Generates a color pie and/or hue histogram for a color image.
# Here's the blog post introducing the ideas behind this code:
#
# http://blog.zillabyte.com/post/11193458776/color-as-data
#

import cairo
import colorsys
import math
import operator
from optparse import OptionParser
import png
import random
import sys

# globals
# =======

# Many of these are setup in setupParameters().

colorBuckets = None
numBuckets = 0
numPixels = 0
maxCount = 0
pixPts = None

width = 600
height = 600
lineWidth = 1.0
histInnerRadius = 125
margin = 10

gaussParam = 0
lineLenExp = 0
smoothVec = None

lightnessCutoff = 0.03  # Higher values = omit more whitish/blackish colors from hist.
saturationCutoff = 0.05  # Higher values = omit more grayish colors from hist.

# This can be 'hist', 'pie', or 'both'.  Set via -m on the command-line.
outputMode = 'hist'


# functions
# =========

def sqDist(a, b):
  diffs = map(lambda x,y: x-y, a, b)
  sqDiffs = map(lambda x: x*x, diffs)
  return sum(sqDiffs)

def findCentroids(pts, centers):
  dim = len(pts[0])
  k = len(centers)
  sumForCenters = [[0 for i in xrange(dim)] for j in xrange(k)]
  numForCenters = [0 for j in xrange(k)]
  for pt in pts:
    closest = 0
    closestSqDist = sqDist(pt, centers[0])
    for i in xrange(1, k):
      c = centers[i]
      s = sqDist(pt, c)
      if s < closestSqDist:
        closest = i
        closestSqDist = s
    sumForCenters[closest] = map(lambda x,y: x+y, sumForCenters[closest], pt)
    numForCenters[closest] += 1
  c = [[float(sumForCenters[i][j]) / numForCenters[i] for j in xrange(dim)] for i in xrange(k) if numForCenters[i]]
  n = [numForCenters[i] for i in xrange(k) if numForCenters[i]]
  return (c, n)

def kmeans(pts, k):
  iters = 0
  centers = random.sample(pts, k)
  maxIters = 3  # 8 TEMP
  while iters < maxIters:
    print "Starting iter %d" % iters
    newCenters, nums = findCentroids(pts, centers)
    if centers == newCenters:
      return centers
    centers = newCenters
    iters += 1
  return (centers, nums)

def addHLSPixel(h, l, s):
  global colorBuckets
  global numBuckets
  global numPixels
  global maxCount
  if l < lightnessCutoff or l > (1 - lightnessCutoff) or s < saturationCutoff:
    return
  if colorBuckets is None:
    colorBuckets = [(0, 0, 0) for i in xrange(numBuckets)]
  index = int(h * numBuckets)
  if index == numBuckets: index -= 1
  (count, lsum, ssum) = colorBuckets[index]
  colorBuckets[index] = (count + 1, lsum + l, ssum + s)
  maxCount = max(maxCount, count + 1)
  numPixels += 1

def setSmoothVec():
  global numBuckets
  global smoothVec
  global gaussParam
  if smoothVec is not None: return
  # Uncomment these lines to set up the identity vector.
  #smoothVec = [0 for i in xrange(numBuckets)]
  #smoothVec[0] = 1
  #return
  smoothVec = []
  s = 0.0
  # Lower weights give smoother end results.
  weight = gaussParam # float(numBuckets) * 0.05
  for i in xrange(numBuckets):
    x = float(i) / numBuckets
    wx = weight * x
    y = math.exp(-wx * wx)
    x -= 1
    wx = weight * x
    y += math.exp(-wx * wx)
    smoothVec.append(y)
    s += y
  for i in xrange(numBuckets):
    smoothVec[i] /= s

def drawLine(ctx, a, b, c, d):
  ctx.move_to(a, b)
  ctx.line_to(c, d)
  ctx.stroke()

def drawRadialLine(ctx, idx, lineLen):
  global numBuckets
  global histInnerRadius
  margin = histInnerRadius
  startX = float(width) / 2
  startY = float(height) / 2
  radius = (min(startX, startY) - margin) * lineLen
  angle = float(idx) / numBuckets * 2.0 * math.pi
  delta = 1.0 * math.pi / numBuckets
  numLines = 10
  for i in xrange(-numLines, numLines):
    a = angle + i * delta / numLines
    fromX = startX + margin * math.cos(a)
    fromY = startY + margin * math.sin(a)
    toX = startX + (radius + margin) * math.cos(a)
    toY = startY + (radius + margin) * math.sin(a)
    drawLine(ctx, fromX, fromY, toX, toY)

def pathPieSlice(ctx, start, end):
  global histInnerRadius
  midX = float(width) / 2
  midY = float(height) / 2
  a1 = 2.0 * math.pi * start
  a2 = 2.0 * math.pi * end
  ctx.move_to(midX, midY)
  ctx.arc(midX, midY, histInnerRadius, a1, a2)
  ctx.close_path()

# Accepts start, end in the range [0, 1].
def drawPieSlice(ctx, start, end):
  pathPieSlice(ctx, start, end)
  ctx.fill()
  pathPieSlice(ctx, start, end)
  ctx.save()
  ctx.set_line_width(0.5)
  ctx.stroke()
  ctx.restore()

# primary functions (called from main)
# ====================================

def makeParser():
  usage = "Usage: %s [options] <input png> <output png>" % sys.argv[0]
  parser = OptionParser(usage=usage)
  parser.add_option("-s", "--size", action="store", type="string",
                    dest="size", default="600x600",
                    help="set the output size (default 600x600)")
  parser.add_option("-m", "--mode", action="store", type="string",
                    dest="mode", default="hist",
                    help="one of {hist,pie,both} (default is hist)")
  parser.add_option("-b", "--drawBkg", action="store_true",
                    dest="drawBkg", default=False,
                    help="draw a white background")
  return parser

def setupParameters(options, args):
  global width
  global height
  global outputMode
  global numBuckets
  global gaussParam
  global lineLenExp
  global lineWidth
  global inputFilename
  global outputFilename
  global histInnerRadius

  if len(args) != 3:
    parser.print_help()
    sys.exit(2)

  [width, height] = map(int, options.size.split('x'))

  outputMode = options.mode

  factor = 2 if outputMode == 'hist' else 4
  numBuckets = factor * (width - 2 * margin) # TODO set good numBuckets for pies

  gaussParam = max(float(numBuckets) * 0.2, 150)  # Lower = smoother histogram.
  if outputMode == 'both': gaussParam /= 2
  lineLenExp = 0.5 if outputMode == 'hist' else 0.4  # In [0,1].  Lower = more even histogram.

  lineWidth = float(width - 2 * margin) / numBuckets * 1. # 0.4

  inputFilename = args[1]
  outputFilename = args[2]

  radiusFactor = 0.2 if outputMode == 'both' else 0.45
  histInnerRadius = min(width, height) * radiusFactor

def readImage():
  global pixPts
  print "Reading in image data."
  reader = png.Reader(inputFilename)
  imgData = reader.asRGBA8()
  pixels = imgData[2]
  pixPts = []
  rowCount = 0
  for row in pixels:
    rowCount += 1
    row = map(lambda x: x / 255.0, row)
    numElts = len(row)
    for i in xrange(0, numElts, 4):
      pixPts.append(row[i:i+3])
      if outputMode != 'pie':
        (h, l, s) = colorsys.rgb_to_hls(*tuple(row[i:i+3]))
        addHLSPixel(h, l, s)

def smoothHist():
  global colorBuckets
  global numBuckets
  global smoothVec
  global maxCount
  global outputMode
  if outputMode == 'pie': return
  print "Smoothing the histogram"
  maxCount = 0
  setSmoothVec()
  newVals = []
  for i in xrange(numBuckets):
    count = 0.0
    lsum = 0.0
    ssum = 0.0
    for j in xrange(numBuckets):
      k = (i + j) % numBuckets
      (c, l, s) = colorBuckets[k]
      count += smoothVec[j] * c
      lsum += smoothVec[j] * l
      ssum += smoothVec[j] * s
    newVals.append((count, lsum, ssum))
    maxCount = max(maxCount, count)
  colorBuckets = newVals

def makeCairoContext(drawBkg):
  surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
  ctx = cairo.Context(surface)
  ctx.set_line_width(lineWidth)
  if drawBkg:
    ctx.set_source_rgb(1, 1, 1)
    ctx.rectangle(0, 0, width, height)
    ctx.fill()
  return (ctx, surface)

def drawHist(ctx):
  print "Drawing the histogram."
  minCount = 0
  for i in xrange(numBuckets):
    (count, lsum, ssum) = colorBuckets[i]
    if count == 0: continue
    hue = float(i + 0.5) / numBuckets
    lightness = float(lsum) / count
    saturation = float(ssum) / count
    lineLen = float(count + minCount) / (maxCount + minCount)
    (r, g, b) = colorsys.hls_to_rgb(hue, lightness, saturation)
    ctx.set_source_rgb(r, g, b)
    x = margin + i * float(width - 2 * margin) / numBuckets
    # Normalizer; basically smooth out tall peaks a bit.
    lineLen = math.pow(lineLen, lineLenExp);
    if outputMode == 'both':
      drawRadialLine(ctx, i, lineLen)
    elif outputMode == 'hist':
      barHeight = height - 2 * margin
      bottom = height - margin
      drawLine(ctx, x, bottom, x, bottom - lineLen * barHeight)

def drawPie(ctx):
  k = 5
  (centers, nums) = kmeans(pixPts, k)
  total = sum(nums)
  s = 0.0
  print "Drawing the color pie."
  k = len(centers)  # May be < initial k.
  for i in xrange(k):
    c = centers[i]
    ctx.set_source_rgb(c[0], c[1], c[2])
    size = float(nums[i]) / total
    drawPieSlice(ctx, s, s + size)
    s += size

# main
# ====

if __name__ == '__main__':

  parser = makeParser()
  (options, args) = parser.parse_args(sys.argv)
  setupParameters(options, args)

  readImage()
  smoothHist()
  ctx, surface = makeCairoContext(options.drawBkg)
  if outputMode != 'pie': drawHist(ctx)
  if outputMode != 'hist': drawPie(ctx)

  surface.write_to_png(outputFilename)
