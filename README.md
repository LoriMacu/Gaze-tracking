# Gaze-tracking
Simple gaze tracker based on an inference method. 
The gaze tracker works with a calibration first step, and then inferes the gaze via GPR and smoothing through a Kalman Filter. The code will perform well only if the head is kept fixed during the calibration and gaze estimation step.
