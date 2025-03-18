import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

################################################################################
# 0) SCREEN SIZE (for centering windows) - Adjust to your laptop's resolution
################################################################################
SCREEN_W = 1440
SCREEN_H = 720

################################################################################
# (A) KALMAN FILTER FOR 2D GAZE
################################################################################
class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        # [x, y, vx, vy]
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        # Tune to your environment
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros((4,1), np.float32)

    def predict(self):
        return self.kf.predict()

    def correct(self, x_meas, y_meas):
        measurement = np.array([[np.float32(x_meas)], [np.float32(y_meas)]])
        corrected = self.kf.correct(measurement)
        return corrected[0, 0], corrected[1, 0]

################################################################################
# (B) EYE + HEAD POSE FEATURES
################################################################################
def get_eye_features(face_landmarks, frame_width, frame_height):
    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * frame_width, lm.y * frame_height])

    # Right eye
    iris_right = get_point(473)
    r_left_corner = get_point(33)
    r_right_corner = get_point(133)
    r_top = get_point(159)
    r_bottom = get_point(145)

    norm_r_x = (iris_right[0] - r_left_corner[0]) / (r_right_corner[0] - r_left_corner[0] + 1e-6)
    norm_r_y = (iris_right[1] - r_top[1])        / (r_bottom[1] - r_top[1] + 1e-6)

    # Left eye
    iris_left = get_point(468)
    l_left_corner = get_point(263)
    l_right_corner = get_point(362)
    l_top = get_point(386)
    l_bottom = get_point(374)

    norm_l_x = (iris_left[0] - l_left_corner[0])  / (l_right_corner[0] - l_left_corner[0] + 1e-6)
    norm_l_y = (iris_left[1] - l_top[1])         / (l_bottom[1] - l_top[1] + 1e-6)

    return (norm_r_x, norm_r_y), (norm_l_x, norm_l_y)

def get_head_pose_solvePnP(face_landmarks, frame):
    focal_length_px = frame.shape[1]
    face_2d = []
    face_3d = []

    # Using 6 key landmarks
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in {1, 33, 263, 61, 291, 199}:
            x_px = int(lm.x * frame.shape[1])
            y_px = int(lm.y * frame.shape[0])
            face_2d.append([x_px, y_px])
            face_3d.append([x_px, y_px, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    cam_matrix = np.array([
        [focal_length_px, 0,               frame.shape[1]/2],
        [0,               focal_length_px, frame.shape[0]/2],
        [0,               0,               1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs)
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x_angle = angles[0] * 360
    y_angle = angles[1] * 360
    z_angle = angles[2] * 360
    return x_angle, y_angle, z_angle

def get_feature_vector(face_landmarks, frame, frame_width, frame_height):
    (rnx, rny), (lnx, lny) = get_eye_features(face_landmarks, frame_width, frame_height)
    x_angle, y_angle, z_angle = get_head_pose_solvePnP(face_landmarks, frame)
    return [rnx, rny, lnx, lny, x_angle, y_angle, z_angle]

################################################################################
# (C) MOVING-DOT CALIBRATION WITH TWO PASSES, DIFFERENT ORDERS
################################################################################
def moving_dot_calibration_phase(cap, face_mesh,
                                 frame_width, frame_height,
                                 stim_width, stim_height,
                                 duration_sec=30, fps=30):
    """
    We do TWO passes of the moving dot with different waypoint orders.
    The total calibration time is 'duration_sec', 
    meaning each pass effectively takes half that time (if they have same length).
    """

    cv2.namedWindow("Calibration Stimulus", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Stimulus", stim_width, stim_height)

    # Center window
    center_x = (SCREEN_W - stim_width) // 2
    center_y = (SCREEN_H - stim_height) // 2
    cv2.moveWindow("Calibration Stimulus", center_x, center_y)

    print("=== Moving Dot Calibration (Two Passes) ===")
    print(f"Will run ~{duration_sec}s total. Press 'c' to start or 'q' to exit early.")

    # Wait for 'c'
    while True:
        key = cv2.waitKey(50) & 0xFF
        temp_screen = np.zeros((stim_height, stim_width, 3), dtype=np.uint8)
        cv2.putText(temp_screen, "Press 'c' to start calibration",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        cv2.imshow("Calibration Stimulus", temp_screen)
        if key == ord('c'):
            break
        elif key == ord('q'):
            cv2.destroyWindow("Calibration Stimulus")
            return None, None

    total_frames = int(duration_sec * fps)

    # --- PASS 1 WAYPOINTS ---
    pass1 = [
        (0.5, 0.5),  # center
        (0.05, 0.05),# top-left
        (0.95, 0.05),# top-right
        (0.95, 0.95),# bottom-right
        (0.05, 0.95),# bottom-left
        (0.5, 0.5)   # back center
    ]
    # --- PASS 2 WAYPOINTS (different order) ---
    pass2 = [
        (0.5, 0.5),  # center
        (0.05, 0.95),# bottom-left
        (0.95, 0.95),# bottom-right
        (0.95, 0.05),# top-right
        (0.05, 0.05),# top-left
        (0.5, 0.5)   # back center
    ]

    # Combine them into one big list
    # This means we have pass1+pass2, each pass covers half the total time.
    waypoints = pass1 + pass2[1:]  
    # pass2[1:] to avoid repeating (0.5,0.5) consecutively

    segments = len(waypoints) - 1  # total segments
    frames_per_segment = total_frames // segments

    X = []
    Y = []

    def interpolate(sx, sy, ex, ey, alpha):
        return (sx + (ex - sx)*alpha, sy + (ey - sy)*alpha)

    current_segment = 0
    frame_count = 0

    while frame_count < total_frames:
        if current_segment >= segments:
            break

        start_nx, start_ny = waypoints[current_segment]
        end_nx, end_ny = waypoints[current_segment + 1]

        seg_frame_index = frame_count - (current_segment * frames_per_segment)
        alpha = seg_frame_index / frames_per_segment

        dot_nx, dot_ny = interpolate(start_nx, start_ny, end_nx, end_ny, alpha)
        dot_x = int(dot_nx * stim_width)
        dot_y = int(dot_ny * stim_height)

        # Draw
        stim_img = np.zeros((stim_height, stim_width, 3), dtype=np.uint8)
        cv2.circle(stim_img, (dot_x, dot_y), 20, (0,0,255), -1)
        cv2.imshow("Calibration Stimulus", stim_img)

        ret, frame = cap.read()
        if not ret:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_count += 1
            continue

        # FaceMesh
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            try:
                feat = get_feature_vector(results.multi_face_landmarks[0],
                                          frame,
                                          frame_width,
                                          frame_height)
                X.append(feat)
                Y.append([dot_nx, dot_ny])  # normalized
            except Exception as e:
                print("Feature extraction error:", e)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1
        if seg_frame_index >= frames_per_segment:
            current_segment += 1

    cv2.destroyWindow("Calibration Stimulus")
    if len(X) == 0:
        return None, None
    return np.array(X), np.array(Y)

################################################################################
# (D) REAL-TIME TEST (WITH KALMAN), CENTERED
################################################################################
def real_time_test(cap, face_mesh,
                   frame_width, frame_height,
                   stim_width, stim_height,
                   model,
                   use_kalman=True):
    import csv
    cv2.namedWindow("Gaze Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gaze Estimation", stim_width, stim_height)

    # Center window
    center_x = (SCREEN_W - stim_width) // 2
    center_y = (SCREEN_H - stim_height) // 2
    cv2.moveWindow("Gaze Estimation", center_x, center_y)

    kf = KalmanFilter2D() if use_kalman else None

    gaze_positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        raw_x, raw_y = -100, -100

        if results.multi_face_landmarks:
            try:
                feat = get_feature_vector(results.multi_face_landmarks[0],
                                          frame, frame_width, frame_height)
                feat = np.array([feat], dtype=np.float32)
                pred = model.predict(feat)[0]
                nx, ny = pred[0], pred[1]
                raw_x = nx * stim_width
                raw_y = ny * stim_height
            except:
                pass

        if kf:
            kf.predict()
            if raw_x >= 0 and raw_y >= 0:
                filtered_x, filtered_y = kf.correct(raw_x, raw_y)
            else:
                predicted = kf.kf.statePre
                filtered_x, filtered_y = predicted[0,0], predicted[1,0]
            gaze_x, gaze_y = filtered_x, filtered_y
        else:
            gaze_x, gaze_y = raw_x, raw_y

        gaze_positions.append((gaze_x, gaze_y))

        # Draw
        stim_img = np.zeros((stim_height, stim_width, 3), dtype=np.uint8)
        if gaze_x >= 0 and gaze_y >= 0:
            cv2.circle(stim_img, (int(gaze_x), int(gaze_y)), 20, (0,255,0), -1)
        cv2.imshow("Gaze Estimation", stim_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Gaze Estimation")

    # Save CSV
    with open("gaze_positions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        for gx, gy in gaze_positions:
            writer.writerow([gx, gy])
    print("Saved gaze_positions.csv")

    # Plot
    xs = [p[0] for p in gaze_positions if p[0] >= 0]
    ys = [p[1] for p in gaze_positions if p[1] >= 0]
    if len(xs) == 0:
        print("No valid gaze points.")
        return

    plt.figure(figsize=(10,6))
    plt.scatter(xs, ys, alpha=0.5, s=80, edgecolors='none')
    plt.title("Real-Time Gaze Predictions (Kalman)" if use_kalman else "Real-Time Gaze Predictions")
    plt.xlim([0, stim_width])
    plt.ylim([stim_height, 0])
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True)
    plt.savefig("gaze_points.png", bbox_inches='tight')
    print("Saved gaze_points.png.")
    plt.show()

################################################################################
# (E) MAIN
################################################################################
def main():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stim_w = 1280
    stim_h = 720

    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        # 1) Calibration with TWO passes
        X, Y = moving_dot_calibration_phase(
            cap, face_mesh,
            frame_width, frame_height,
            stim_width=stim_w, stim_height=stim_h,
            duration_sec=30,  # total calibration time
            fps=30
        )
        if X is None or len(X) == 0:
            print("No calibration data collected. Exiting.")
            cap.release()
            return

        # 2) Train GPR
        kernel = RBF() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, Y)

        # 3) Real-time test
        real_time_test(
            cap, face_mesh,
            frame_width, frame_height,
            stim_w, stim_h,
            model,
            use_kalman=True
        )

    cap.release()


if __name__ == "__main__":
    main()
