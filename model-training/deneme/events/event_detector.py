# events/event_detector.py
import yaml
from collections import deque
from .utils_event import center_from_bbox, speed, is_close
import logging
logger = logging.getLogger(__name__)

class EventDetector:
    def __init__(self, config=None):
        # config: dict with thresholds
        if config is None:
            config = {}
        self.shot_speed = config.get('ball_speed_shot_threshold', 12.0)  # px/frame
        self.pass_speed = config.get('ball_speed_pass_threshold', 4.0)
        self.player_run_speed = config.get('player_run_speed_threshold', 6.0)
        self.player_close_threshold = config.get('player_close_threshold', 60)
        self.possession_frames = config.get('possession_frames', 3)

        self.prev_ball_center = None
        self.ball_speed_history = deque(maxlen=5)
        self.player_prev_centers = {}  # track_id -> center
        self.player_speed_history = {}  # track_id -> deque
        self.possession = None  # current possessor track_id
        self.possession_count = 0
        self.events = []

    def process_frame(self, tracks):
        """
        tracks: list of dicts with {'track_id','class','bbox','center'}
        """
        # separate ball & players
        ball = None
        players = []
        for t in tracks:
            if t['class'] == 'ball':
                ball = t
            elif t['class'] == 'player':
                players.append(t)

        # compute ball center and speed
        ball_center = ball['center'] if ball else None
        b_speed = 0.0
        if ball_center is not None:
            b_speed = speed(self.prev_ball_center, ball_center, dt=1.0)
            self.ball_speed_history.append(b_speed)

        # shot detection
        if b_speed >= self.shot_speed:
            # register shot event
            self.events.append({
                "type": "SHOT",
                "frame": None,
                "ball_speed": b_speed,
                "ball_center": ball_center
            })
            logger.debug(f"SHOT detected speed={b_speed}")

        # pass detection: ball speed between pass thresholds and possession change
        if b_speed >= self.pass_speed and b_speed < self.shot_speed:
            # find nearest player to ball center
            nearest = None
            nearest_dist = float('inf')
            for p in players:
                d = is_close(p['center'], ball_center, float('inf')) and 0 or ((p['center'][0]-ball_center[0])**2 + (p['center'][1]-ball_center[1])**2)**0.5
                if d < nearest_dist:
                    nearest_dist = d
                    nearest = p
            if nearest and nearest_dist < self.player_close_threshold:
                # consider a pass to 'nearest'
                self.events.append({
                    "type": "PASS",
                    "frame": None,
                    "to_track": nearest['track_id'],
                    "ball_speed": b_speed,
                    "ball_center": ball_center
                })
                logger.debug(f"PASS detected to {nearest['track_id']} speed={b_speed}")

        # player run detection
        for p in players:
            tid = p['track_id']
            prev = self.player_prev_centers.get(tid)
            sp = speed(prev, p['center'])
            hist = self.player_speed_history.get(tid)
            if hist is None:
                hist = deque(maxlen=5)
                self.player_speed_history[tid] = hist
            hist.append(sp)
            if sp >= self.player_run_speed:
                self.events.append({
                    "type": "RUN",
                    "frame": None,
                    "player": tid,
                    "speed": sp,
                    "center": p['center']
                })
                logger.debug(f"RUN detected for {tid} speed={sp}")
            self.player_prev_centers[tid] = p['center']

        # possession detection (who is nearest to ball and within threshold)
        if ball_center is not None:
            possessor = None
            min_d = float('inf')
            for p in players:
                d = ((p['center'][0]-ball_center[0])**2 + (p['center'][1]-ball_center[1])**2)**0.5
                if d < min_d:
                    min_d = d
                    possessor = p['track_id']
            if min_d <= self.player_close_threshold:
                if self.possession == possessor:
                    self.possession_count += 1
                else:
                    # possession changed
                    self.possession = possessor
                    self.possession_count = 1
                    self.events.append({
                        "type": "POSSESSION_CHANGE",
                        "frame": None,
                        "player": possessor
                    })
            else:
                # ball free
                if self.possession is not None:
                    self.events.append({
                        "type": "POSSESSION_LOST",
                        "frame": None,
                        "player": self.possession
                    })
                self.possession = None
                self.possession_count = 0

        self.prev_ball_center = ball_center

    def get_events(self):
        # return and clear events
        ev = list(self.events)
        self.events.clear()
        return ev
