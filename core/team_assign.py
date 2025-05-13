import numpy as np
import cv2
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = None
        self.id_to_team = {}
        self.initialised = False

        self.ref_color  = (0, 255, 255)  # Yellow
        self.ball_color = (0, 255, 0)    # Green

    def reset(self):
        print("[RESET] Forcing team reset")
        self.team_colors = None
        self.id_to_team = {}
        self.initialised = False

    def extract_shirt_colour(self, frame, bbox):
        h_frame, w_frame, _ = frame.shape
        x1, y1, x2, y2 = [int(i) for i in bbox]
        x1 = max(0, min(x1, w_frame - 1))
        x2 = max(0, min(x2, w_frame - 1))
        y1 = max(0, min(y1, h_frame - 1))
        y2 = max(0, min(y2, h_frame - 1))

        h = y2 - y1
        if h <= 0 or x2 <= x1:
            return np.array([0, 0, 0])

        shirt_region = frame[y1:y1 + h // 2, x1:x2]
        if shirt_region.size == 0:
            return np.array([0, 0, 0])

        resized = cv2.resize(shirt_region, (10, 10)).reshape(-1, 3)
        return np.mean(resized, axis=0)

    def initialise_teams(self, frame, tracks):
        shirt_colours = []
        for t in tracks:
            if t['cls'] in ['1', '2']:
                shirt = self.extract_shirt_colour(frame, t['bbox'])
                shirt_colours.append(shirt)

        if len(shirt_colours) < 2:
            print("⚠ Not enough player/keeper data to cluster team colours.")
            return False

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(shirt_colours)
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1]
        }
        self.initialised = True
        print(f"[INIT] Team colours locked → T1: {self.team_colors[1]}, T2: {self.team_colors[2]}")
        return True

    def assign(self, frame, full_tracks):
        team_candidates = [t for t in full_tracks if t['cls'] in ['1', '2']]

        if not self.initialised:
            success = self.initialise_teams(frame, team_candidates)
            if not success:
                for t in full_tracks:
                    t['team'] = None
                    t['color'] = (128, 128, 128)
                return full_tracks

        for t in full_tracks:
            cls = t['cls']
            tid = t['id']

            if cls == '0':  # Ball
                t['team'] = None
                t['color'] = self.ball_color
                continue
            elif cls == '3':  # Referee
                t['team'] = None
                t['color'] = self.ref_color
                if tid in self.id_to_team:
                    del self.id_to_team[tid]
                continue
            elif cls in ['1', '2']:
                if tid in self.id_to_team:
                    team_id = self.id_to_team[tid]
                else:
                    shirt = self.extract_shirt_colour(frame, t['bbox'])
                    dists = [np.linalg.norm(shirt - self.team_colors[team]) for team in self.team_colors]
                    team_id = int(np.argmin(dists)) + 1
                    self.id_to_team[tid] = team_id
                    print(f"[LOCK] Track ID {tid} assigned to Team {team_id}")

                t['team'] = self.id_to_team[tid]
                t['color'] = tuple(int(c) for c in self.team_colors[t['team']])
            else:
                t['team'] = None
                t['color'] = (128, 128, 128)

        return full_tracks
