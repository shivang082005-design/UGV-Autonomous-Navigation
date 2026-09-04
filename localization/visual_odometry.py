import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path


class VisualOdometry:
    """
    Visual Odometry using ORB features and Essential Matrix recovery.
    
    Pipeline:
    1. Read consecutive frames from video
    2. Detect and match ORB features
    3. Estimate camera motion via Essential Matrix
    4. Recover pose (R, t) using cv2.recoverPose()
    5. Accumulate relative transformations to compute absolute trajectory
    """
    
    def __init__(self, focal_length=500, principal_point=None):
        """
        Initialize Visual Odometry parameters.
        
        Args:
            focal_length: Camera focal length (default 500 for normalized camera)
            principal_point: (cx, cy) principal point; if None, use image center
        """
        self.focal_length = focal_length
        self.principal_point = principal_point
        
        # Camera intrinsic matrix (assumes normalized camera if not specified)
        self.K = None
        self.dist_coeffs = np.zeros(4)
        
        # ORB detector with max 5000 features
        self.orb = cv2.ORB_create(nfeatures=5000)
        
        # BFMatcher for ORB (Hamming distance)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Trajectory storage: list of (x, y, theta)
        self.trajectory = [(0, 0, 0)]
        
        # Current pose in world frame
        self.R_global = np.eye(3)  # Rotation matrix
        self.t_global = np.zeros((3, 1))  # Translation vector
        
        # Previous frame for feature matching
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        
        print("[VisualOdometry] Initialized with focal_length={}, principal_point={}".format(
            focal_length, principal_point))
    
    def set_camera_intrinsics(self, frame_shape):
        """
        Set camera intrinsic matrix based on frame dimensions.
        
        Args:
            frame_shape: (height, width) of video frames
        """
        height, width = frame_shape[:2]
        if self.principal_point is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = self.principal_point
        
        self.K = np.array([
            [self.focal_length, 0, cx],
            [0, self.focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print("[VisualOdometry] Camera intrinsics K:\n{}".format(self.K))
    
    def extract_features(self, frame):
        """
        Extract ORB features from a frame.
        
        Args:
            frame: Grayscale image frame
            
        Returns:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Numpy array of ORB descriptors
        """
        kp, desc = self.orb.detectAndCompute(frame, None)
        return kp, desc
    
    def match_features(self, desc1, desc2):
        """
        Match features between two frames using BFMatcher.
        Apply Lowe's ratio test for robust matching.
        
        Args:
            desc1: Descriptors from first frame
            desc2: Descriptors from second frame
            
        Returns:
            good_matches: List of good cv2.DMatch objects
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # knnMatch returns k nearest neighbors for each descriptor
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # If first match is significantly closer, keep it
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def get_matched_points(self, kp1, kp2, matches):
        """
        Extract (x, y) coordinates from matched keypoints.
        
        Args:
            kp1: Keypoints from first frame
            kp2: Keypoints from second frame
            matches: List of DMatch objects
            
        Returns:
            pts1, pts2: Numpy arrays of shape (N, 2)
        """
        if len(matches) < 4:
            return None, None
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return pts1, pts2
    
    def estimate_motion(self, pts1, pts2):
        """
        Estimate camera motion using Essential Matrix.
        
        Args:
            pts1: Points in first frame (N, 2)
            pts2: Points in second frame (N, 2)
            
        Returns:
            R: Rotation matrix (3, 3) or None on failure
            t: Translation vector (3, 1) or None on failure
        """
        if pts1 is None or pts2 is None or len(pts1) < 5:
            return None, None
        
        # Find Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            print("[WARNING] Could not compute Essential Matrix")
            return None, None
        
        # Recover pose from Essential Matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t
    
    def update_pose(self, R, t):
        """
        Update global pose by accumulating relative transformations.
        
        Args:
            R: Relative rotation (3, 3)
            t: Relative translation (3, 1)
        """
        # T = [R | t; 0 | 1] represents transformation from current to next frame
        # Update global pose: T_global = T_global * T_relative
        self.R_global = self.R_global @ R
        self.t_global = self.t_global + self.R_global @ t
    
    def compute_theta(self):
        """
        Compute yaw angle (theta) from rotation matrix.
        Assumes camera optical axis is Z, and we extract rotation around Z.
        
        Returns:
            theta: Yaw angle in radians
        """
        # For 3x3 rotation matrix, yaw can be approximated from R[1,0] and R[0,0]
        # theta = atan2(R[1,0], R[0,0])
        # But for camera motion, we use atan2(R[0,1], R[0,0])
        theta = np.arctan2(self.R_global[0, 1], self.R_global[0, 0])
        return theta
    
    def process_frame(self, frame, frame_id):
        """
        Process a single frame and update trajectory.
        
        Args:
            frame: Current video frame (BGR)
            frame_id: Frame number
            
        Returns:
            success: Boolean indicating if processing was successful
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize camera intrinsics on first frame
        if self.K is None:
            self.set_camera_intrinsics(gray.shape)
        
        # Extract ORB features
        kp, desc = self.extract_features(gray)
        
        if frame_id == 0:
            # First frame: store features and skip motion estimation
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_desc = desc
            print(f"[Frame {frame_id}] Features extracted: {len(kp)}")
            return True
        
        # Match features with previous frame
        matches = self.match_features(self.prev_desc, desc)
        print(f"[Frame {frame_id}] Features: {len(kp)}, Matches: {len(matches)}")
        
        if len(matches) < 5:
            print(f"[Frame {frame_id}] WARNING: Insufficient matches, skipping this frame")
            return False
        
        # Get matched point coordinates
        pts1, pts2 = self.get_matched_points(self.prev_kp, kp, matches)
        
        if pts1 is None:
            return False
        
        # Estimate camera motion
        R, t = self.estimate_motion(pts1, pts2)
        
        if R is None or t is None:
            print(f"[Frame {frame_id}] WARNING: Could not estimate motion")
            return False
        
        # Update global pose
        self.update_pose(R, t)
        
        # Compute current pose
        x = self.t_global[0, 0]
        y = self.t_global[1, 0]
        theta = self.compute_theta()
        
        self.trajectory.append((x, y, theta))
        
        print(f"[Frame {frame_id}] Pose: x={x:.4f}, y={y:.4f}, theta={theta:.4f}")
        
        # Update previous frame
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_desc = desc
        
        return True
    
    def process_video(self, video_path, max_frames=None):
        """
        Process video file and compute trajectory.
        
        Args:
            video_path: Path to input video file
            max_frames: Maximum number of frames to process (None = all)
        """
        print(f"\n[VisualOdometry] Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[VisualOdometry] Video: {frame_count} frames @ {fps:.1f} FPS")
        
        frame_id = 0
        success_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_id >= max_frames:
                break
            
            if self.process_frame(frame, frame_id):
                success_count += 1
            
            frame_id += 1
        
        cap.release()
        print(f"\n[VisualOdometry] Processing complete: {frame_id} frames, {success_count} successful")
        print(f"[VisualOdometry] Trajectory length: {len(self.trajectory)} poses")
    
    def save_trajectory_csv(self, output_path):
        """
        Save trajectory as CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'x', 'y', 'theta_rad', 'theta_deg'])
            for i, (x, y, theta) in enumerate(self.trajectory):
                writer.writerow([i, x, y, theta, np.degrees(theta)])
        
        print(f"[VisualOdometry] Trajectory saved to {output_path}")
    
    def plot_trajectory(self, output_path):
        """
        Plot trajectory in 2D and save as image.
        
        Args:
            output_path: Path to output PNG file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract coordinates
        xs = [p[0] for p in self.trajectory]
        ys = [p[1] for p in self.trajectory]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectory
        ax.plot(xs, ys, 'b-', linewidth=2, label='Trajectory')
        ax.scatter([xs[0]], [ys[0]], color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], color='red', s=100, marker='x', label='End', zorder=5)
        
        # Plot some direction arrows
        step = max(1, len(xs) // 10)
        for i in range(0, len(xs) - step, step):
            dx = xs[i + step] - xs[i]
            dy = ys[i + step] - ys[i]
            if np.sqrt(dx**2 + dy**2) > 0.01:
                ax.arrow(xs[i], ys[i], dx * 0.5, dy * 0.5, 
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('Visual Odometry Trajectory', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[VisualOdometry] Trajectory plot saved to {output_path}")
        plt.close()


def main():
    """
    Main entry point for Visual Odometry.
    Expects video file as command-line argument or default test video.
    """
    import sys
    
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Usage: python visual_odometry.py <video_path> [max_frames]")
        print("\nExample: python visual_odometry.py outdoor_video.mp4 300")
        return
    
    # Optional max frames parameter
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Create Visual Odometry instance
    vo = VisualOdometry(focal_length=500)
    
    # Process video
    try:
        vo.process_video(video_path, max_frames=max_frames)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    vo.save_trajectory_csv(output_dir / "trajectory.csv")
    vo.plot_trajectory(output_dir / "trajectory.png")
    
    print("\n" + "="*60)
    print("Visual Odometry Complete!")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  - trajectory.csv: Pose data (x, y, theta)")
    print(f"  - trajectory.png: Trajectory plot")
    print("="*60)


if __name__ == "__main__":
    main()