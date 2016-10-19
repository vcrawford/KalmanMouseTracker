import numpy.linalg as la
import numpy as np

# Emulates a Kalman filter, which tracks the state
# of a system.
class KalmanFilter:

   # - Let x_{k} be the believed state of the system at step k 
   # - x_{k}' the predicted state of the kth step 
   #   without looking at observations
   # - z_{k} the observation at step k
   # - A is the prediction matrix that takes x_{k-1} to x_{k}'
   # - B is the control transformation
   # - Q is the covariance for the prediction noise
   # - H is the transformation from state vector to observed vector
   # - R is the covariance for the observation noise
   # - state_init is the state we would like to start with
   # - cov_init is the covariance we'd like to start with
   def __init__(self, A, B, Q, H, R):
      self.A = A
      self.B = B
      self.Q = Q
      self.H = H
      self.R = R

      #Not the first time step until initialized
      self.step = -1

   def initializeState(self, state_init, cov_init):
      self.step = 0
      #What the state of our system is
      self.state = state_init;
      self.cov = cov_init;

   # Update state to next step
   # obs is the observation value we got
   def updateState(self, obs): 
      
      pred = self.makePrediction()
      P = self.covariancePrediction()
      K = self.kalmanGain(P)
      self.newState(pred, obs, K, P)

   # Make prediction of next state
   # Return prediction
   def makePrediction(self):
      return np.dot(self.A, self.state)

   # What is the covariance of the prediction for the next state
   # So this is the covariance of the Gaussian distribution with the
   # real value at its mean that our prediction is drawn from
   # (tells us about its certainty)
   def covariancePrediction(self):
      return np.dot(self.A, np.dot(self.cov, np.transpose(self.A))) + self.Q


   # Compute the Kalman Gain
   # P is the predicted covariance
   def kalmanGain(self, P):
      H_t = np.transpose(self.H)
      a = np.dot(P, H_t)
      b = np.dot(self.H, np.dot(P, H_t))
      return np.dot(a, la.inv(b + self.R))


   # Compute the new state of the system using Kalman gain
   # x is the predicted state
   # obs is the observed state
   # K is the Kalman gain
   def newState(self, x, obs, K, P):
      self.state = x + np.dot(K, (obs - np.dot(self.H, x)))
      self.cov = np.dot((np.array([[1, 0], [0, 1]]) - np.dot(K, self.H)), P)
      self.step = self.step + 1

