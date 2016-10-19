from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import math as m
import CovarianceEllipse as ell 
import KalmanFilter as kfilter

#size of our window
win_width, win_height = 1000, 600

# Initialize our Kalman Filter to track movement
# of the mouse
# x, y are the first mouse coordinates
# The kalman filter is global!
def initializeKalmanFilter():
   A = np.array([[1, 0], [0, 1]])
   B = np.array([[1, 0], [0, 1]])
   Q = 20*np.array([[1, 0], [0, 1]])
   H = np.array([[1, 0], [0, 1]])
   R = 10*np.array([[1, 0], [0, 1]])
   return kfilter.KalmanFilter(A, B, Q, H, R)

def mouse(x, y):
   refreshProgram(x, win_height - y)

# Points on outline of ellipse with center center
# Width and height are 1/2 the lengths of each primary axis
# Theta is how to rotate the ellipse counter-clockwise about
# the origin
def ellipsePoints(center = [0, 0], width = 1, height = 1, theta = 0, precision = 20):
   stretch = np.array([[width, 0], [0, height]])
   rotate = np.array([[m.cos(theta), -1*m.sin(theta)], [m.sin(theta), m.cos(theta)]])
   transform = np.dot(rotate, stretch)

   points = []
   for angle in np.linspace(0, 2*np.pi, precision):
      point = np.array([m.cos(angle), m.sin(angle)])
      new_point = np.dot(transform, point)
      points.append(new_point + center)
  
   return points 

#Draw the ellipse
def drawEllipse(x, y, width = 1, height = 1, theta = 0, precision = 30):
   glBegin(GL_LINE_LOOP) #start drawing

   points = ellipsePoints([x, y], width, height, theta, precision)

   for point in points:
      glVertex2f(point[0], point[1])

   glEnd() #done drawing


# x, y are mouse coordinates
def refreshProgram(x = -1, y = -1):
   global kalman_filter

   #clear window, reset appearance
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   glLoadIdentity()
   refresh2D()
   glColor3f(0.0, 0.0, 1.0)

   # draw the new prediction ellipse if our mouse
   # is on the screen
   if x is not -1 and y is not -1:

      # We have an observation of where the mouse is
      obs = np.array([x, y])
      if kalman_filter.step == -1:
         cov_init = np.array([[20, 0], [0, 40]])
         kalman_filter.initializeState(obs, cov_init)
      else:
         kalman_filter.updateState(obs)

      #draw prediction ellipse
      state = kalman_filter.state
      cov = kalman_filter.cov 
      stretch_x, stretch_y, theta = ell.getEllipseParameters(cov)
      drawEllipse(state[0], state[1], stretch_x, stretch_y, theta, 30)

   #Not sure, just have to do this
   glutSwapBuffers() 

# Makes our window have 2d appearance
def refresh2D():
   glViewport(0, 0, win_width, win_height)
   glMatrixMode(GL_PROJECTION)
   glLoadIdentity()
   glOrtho(0.0, win_width, 0.0, win_height, 0.0, 1.0)
   glMatrixMode(GL_MODELVIEW)
   glLoadIdentity()

#Initialize the entire window
def startKalmanGraphic():
   #Initialize display stuff
   glutInit()
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
   glutInitWindowSize(win_width, win_height) #set window size
   glutInitWindowPosition(0, 0) #set window location
   window = glutCreateWindow("Kalman Filter") #make window

   #Assign when functions run
   glutPassiveMotionFunc(mouse) #run mouse when mouse moves
   #glutDisplayFunc(draw) #function called to draw
   #glutIdleFunc(draw) #function to call all the time

   refreshProgram()
   glutMainLoop() #start!

global kalman_filter
kalman_filter = initializeKalmanFilter()
startKalmanGraphic()

