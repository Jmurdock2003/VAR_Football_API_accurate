import numpy as np  # array operations
import cv2  # OpenCV for image processing
from sklearn.cluster import KMeans  # clustering for team shirt colours

class TeamAssigner:
    """
    Assigns each detected person track to a team based on shirt colour clustering.
    Handles ball and referee with fixed colours.
    """
    def __init__(self):
        # Stored RGB centre colours for each team after initialization
        self.team_colors = None
        # Mapping from track ID to assigned team ID
        self.id_to_team = {}
        # Flag indicating if team colours have been initialized
        self.initialised = False
        # Colours for referee and ball overlays
        self.ref_color = (0, 255, 255)  # yellow for referees
        self.ball_color = (0, 255, 0)    # green for ball
        # Frame counter to periodically re-evaluate assignments
        self.frame_counter = 0

    def reset(self):
        """
        Force a re-initialization of team colours and assignments.
        Call this if tracking drifts or teams swap jerseys.
        """
        print("[RESET] Forcing team reset")
        self.team_colors = None
        self.id_to_team.clear()
        self.initialised = False
        self.frame_counter = 0

    def extract_shirt_colour(self, frame, bbox):
        """
        Sample the top half of a player's bounding box, resize to a small patch,
        and average the pixel RGB values to estimate shirt colour.
        Returns a NumPy array [R, G, B], or [0,0,0] on failure.
        """
        h_frame, w_frame = frame.shape[:2]
        # Convert bbox to integer and clamp to frame boundaries
        x1, y1, x2, y2 = [int(max(0, min(val, lim)))
                         for val, lim in zip(bbox, [w_frame, h_frame, w_frame, h_frame])]
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3, dtype=float)

        # Focus on the upper half of the box (shirt area)
        shirt_region = frame[y1:y1 + (y2 - y1)//2, x1:x2]
        if shirt_region.size == 0 or shirt_region.shape[0] < 3 or shirt_region.shape[1] < 3:
            return np.zeros(3, dtype=float)

        # Resize to fixed small patch and average colours
        try:
            patch = cv2.resize(shirt_region, (10, 10)).reshape(-1, 3)
            return np.mean(patch, axis=0)
        except Exception as e:
            print(f"Shirt colour extraction failed: {e}")
            return np.zeros(3, dtype=float)

    def initialise_teams(self, frame, tracks):
        """
        Cluster player shirt colours into two groups via KMeans.
        Must be called when at least two valid shirt samples exist.
        Returns True on success, False otherwise.
        """
        # Gather shirt colour samples from tracked players and goalkeepers
        samples = []
        for t in tracks:
            if t.get('cls') in ['1', '2']:  # classes for keeper and player
                colour = self.extract_shirt_colour(frame, t.get('bbox', []))
                samples.append(colour)

        # Require at least two samples to cluster
        if len(samples) < 2:
            print("⚠ Not enough shirts to initialize teams.")
            return False

        # Perform 2-means clustering
        try:
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(samples)
            centres = kmeans.cluster_centers_
            # Assign each cluster centre to a team ID (1 and 2)
            self.team_colors = {1: centres[0], 2: centres[1]}
            self.initialised = True
            print(f"[INIT] Team colours set: T1={centres[0]}, T2={centres[1]}")
            return True
        except Exception as e:
            print(f"Team initialization error: {e}")
            return False

    def assign(self, frame, full_tracks):
        """
        Assign team IDs and overlay colours to each track:
        - Ball cls '0' gets green
        - Referee cls '3' gets yellow
        - Players cls '1' or '2' are assigned based on nearest shirt cluster
        Returns the same list with 'team' and 'color' keys added.
        """
        self.frame_counter += 1
        # Filter out only player/keeper tracks for potential clustering
        candidates = [t for t in full_tracks if t.get('cls') in ['1', '2']]

        # Initialize team clusters on first call
        if not self.initialised:
            success = self.initialise_teams(frame, candidates)
            if not success:
                # If initialization fails, set all teams to None (grey)
                for t in full_tracks:
                    t['team'] = None
                    t['color'] = (128, 128, 128)
                return full_tracks

        # Assign each track based on its class
        for t in full_tracks:
            cls = t.get('cls')
            tid = t.get('id')
            # Ball
            if cls == '0':
                t['team'] = None
                t['color'] = self.ball_color
                continue
            # Referee
            if cls == '3':
                t['team'] = None
                t['color'] = self.ref_color
                # Remove any previous assignment for this ID
                self.id_to_team.pop(tid, None)
                continue

            # Players and keepers
            if cls in ['1', '2']:
                reassign = (tid not in self.id_to_team) or (self.frame_counter % 10 == 0)
                # Optionally force reassign on flag
                if reassign:
                    shirt = self.extract_shirt_colour(frame, t.get('bbox', []))
                    # Compute distances to each team centre
                    dists = {
                        team: np.linalg.norm(shirt - centre)
                        for team, centre in self.team_colors.items()
                    }
                    # Assign to the nearest cluster
                    assigned = min(dists, key=dists.get)
                    self.id_to_team[tid] = assigned
                    print(f"[ASSIGN] Track {tid} → Team {assigned}")
                # Set output team and color from the mapping
                team_id = self.id_to_team.get(tid)
                t['team'] = team_id
                colour = self.team_colors.get(team_id, np.array([128,128,128]))
                t['color'] = tuple(int(c) for c in colour)
                continue

            # Default fallback for unrecognized classes
            t['team'] = None
            t['color'] = (128, 128, 128)

        return full_tracks
