import os
import pickle

import cv2
import numpy as np
import pandas as pd

from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tracker = None

    def _ensure_tracking_backend(self):
        if self.model is not None and self.tracker is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required when track stubs are unavailable. "
                "Install it or run with --read-from-stub and a valid stub file."
            ) from exc

        try:
            import supervision as sv
        except ImportError as exc:
            raise ImportError(
                "supervision is required when generating fresh tracks. "
                "Install it or run with --read-from-stub and a valid stub file."
            ) from exc

        self.model = YOLO(self.model_path)
        self.tracker = sv.ByteTrack()
        self.supervision = sv

    def interpolate_ball_positions(self,ball_positions):
        ball_positions =[x.get(1,{}).get("bbox",[np.nan] * 4) for x in ball_positions]
        df_ball_positions=pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate().bfill()

        ball_positions= [{1:{"bbox":x}}for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        self._ensure_tracking_backend()
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections+=detections_batch
           
        return detections

    def get_object_tracks(self, frames,read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks=pickle.load(f)
            return tracks

        detections=self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
    }

        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv={v:k for k,v in cls_names.items()}
            

            #Convert to supervision dectection  format
            detection_supervision= self.supervision.Detections.from_ultralytics(detection)

            #Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                 detection_supervision.class_id[object_ind]=cls_names_inv["player"]
            
            #Track Objects
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id =frame_detection[3]
                track_id=frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id]={"bbox":bbox}
                    
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id]={"bbox":bbox}

            ball_candidates = []
            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    width = max(1.0, bbox[2] - bbox[0])
                    height = max(1.0, bbox[3] - bbox[1])
                    area = width * height
                    ball_candidates.append((area, bbox))

            if ball_candidates:
                _, best_ball_bbox = min(ball_candidates, key=lambda item: item[0])
                tracks["ball"][frame_num][1]={"bbox":best_ball_bbox}

        if stub_path is not None:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir:
                os.makedirs(stub_dir, exist_ok=True)
            with open(stub_path,"wb") as f:
                pickle.dump(tracks,f)

        return tracks
        
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2=int(bbox[3])

        x_center,_=get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width=40
        rectangle_height=20
        x1_rect=x_center - rectangle_width//2
        x2_rect=x_center + rectangle_width//2
        y1_rect=(y2-rectangle_height//2)+15
        y2_rect=(y2+rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame,(int(x1_rect),int(y1_rect)),(int(x2_rect),int(y2_rect)),color,cv2.FILLED)
            
            x1_text=x1_rect+12
            if track_id>99:
                x1_text-=10

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2)

        
        return frame

    def draw_triangle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points=np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)
        
        return frame  

    def draw_ball_trail(self, frame, ball_tracks, frame_num, trail_length=15):
        start_frame = max(0, frame_num - trail_length + 1)
        trail_points = []

        for index in range(start_frame, frame_num + 1):
            ball_track = ball_tracks[index].get(1)
            if ball_track is None:
                continue
            trail_points.append(get_center_of_bbox(ball_track["bbox"]))

        for point_index, point in enumerate(trail_points):
            radius = max(2, 5 - (len(trail_points) - point_index - 1) // 3)
            cv2.circle(frame, point, radius, (0, 255, 0), -1)

        return frame

    def _project_to_minimap(self, normalized_point, top_left, size):
        x = int(top_left[0] + normalized_point[0] * size[0])
        y = int(top_left[1] + normalized_point[1] * size[1])
        return x, y

    def draw_minimap(self, frame, frame_num, tracks, match_summary):
        frame_height, frame_width = frame.shape[:2]
        map_width = min(360, max(220, frame_width // 4))
        map_height = int(map_width * 0.62)
        margin = 20
        top_left = (margin, margin)
        bottom_right = (top_left[0] + map_width, top_left[1] + map_height)

        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (32, 95, 45), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, top_left, bottom_right, (245, 245, 245), 2)

        mid_x = top_left[0] + map_width // 2
        cv2.line(frame, (mid_x, top_left[1]), (mid_x, bottom_right[1]), (245, 245, 245), 2)
        cv2.circle(frame, (mid_x, top_left[1] + map_height // 2), max(18, map_width // 10), (245, 245, 245), 2)

        box_depth = max(28, map_width // 7)
        box_half_height = max(30, map_height // 5)
        center_y = top_left[1] + map_height // 2
        cv2.rectangle(frame, (top_left[0], center_y - box_half_height), (top_left[0] + box_depth, center_y + box_half_height), (245, 245, 245), 2)
        cv2.rectangle(frame, (bottom_right[0] - box_depth, center_y - box_half_height), bottom_right, (245, 245, 245), 2)
        cv2.putText(frame, "Mini Map", (top_left[0] + 10, top_left[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)

        current_players = tracks["players"][frame_num]
        for _, player in current_players.items():
            team_id = int(player.get("team", 0))
            bbox = player["bbox"]
            normalized_point = (
                min(max(((bbox[0] + bbox[2]) / 2) / max(frame_width, 1), 0.0), 1.0),
                min(max(bbox[3] / max(frame_height, 1), 0.0), 1.0),
            )
            point = self._project_to_minimap(normalized_point, top_left, (map_width, map_height))
            color = tuple(int(value) for value in player.get("team_color", (0, 0, 255)))
            cv2.circle(frame, point, 5, color, -1)

        ball_track = tracks["ball"][frame_num].get(1)
        if ball_track is not None:
            bbox = ball_track["bbox"]
            normalized_point = (
                min(max(((bbox[0] + bbox[2]) / 2) / max(frame_width, 1), 0.0), 1.0),
                min(max(bbox[3] / max(frame_height, 1), 0.0), 1.0),
            )
            point = self._project_to_minimap(normalized_point, top_left, (map_width, map_height))
            cv2.circle(frame, point, 4, (0, 255, 0), -1)
            cv2.circle(frame, point, 7, (245, 245, 245), 1)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        frame_height, frame_width = frame.shape[:2]
        overlay=frame.copy()

        panel_width = min(520, frame_width - 40)
        panel_height = 110
        x1 = max(20, frame_width - panel_width - 20)
        y1 = max(20, frame_height - panel_height - 20)
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),-1)
        alpha = 0.4
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame=team_ball_control[:frame_num+1]
        #Get the number of time each time each team had the ball
        team_1_num_frames= team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames= team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        tracked_frames = team_1_num_frames+team_2_num_frames
        if tracked_frames == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_1= team_1_num_frames/tracked_frames
            team_2= team_2_num_frames/tracked_frames

        cv2.putText(frame,f"Team 1 Ball Control: {team_1*100:.2f}%",(x1 + 20,y1 + 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        cv2.putText(frame,f"Team 2 Ball Control: {team_2*100:.2f}%",(x1 + 20,y1 + 80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

        return frame

    def draw_possession_timeline(self, frame, frame_num, team_ball_control):
        frame_height, frame_width = frame.shape[:2]
        bar_width = min(760, frame_width - 80)
        bar_height = 18
        x1 = (frame_width - bar_width) // 2
        y1 = frame_height - 36
        x2 = x1 + bar_width

        cv2.rectangle(frame, (x1, y1), (x2, y1 + bar_height), (225, 225, 225), 2)

        total_frames = max(len(team_ball_control), 1)
        color_map = {0: (180, 180, 180), 1: (255, 120, 80), 2: (80, 170, 255)}

        for index in range(frame_num + 1):
            start_x = x1 + int(index / total_frames * bar_width)
            end_x = x1 + int((index + 1) / total_frames * bar_width)
            team_id = int(team_ball_control[index])
            cv2.rectangle(frame, (start_x, y1), (max(start_x + 1, end_x), y1 + bar_height), color_map.get(team_id, (180, 180, 180)), -1)

        marker_x = x1 + int(((frame_num + 1) / total_frames) * bar_width)
        cv2.line(frame, (marker_x, y1 - 6), (marker_x, y1 + bar_height + 6), (20, 20, 20), 2)
        cv2.putText(frame, "Possession Timeline", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2)

        return frame

    def draw_leaderboard(self, frame, match_summary):
        frame_height, frame_width = frame.shape[:2]
        panel_width = min(330, frame_width // 3)
        panel_height = 150
        x1 = frame_width - panel_width - 20
        y1 = 20
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "Leaders", (x1 + 14, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2)
        for index, player in enumerate(match_summary.get("top_movers", [])[:3]):
            y = y1 + 54 + index * 28
            text = f"P{player['player_id']}  Dist {player['distance_px']:.0f}px  Max {player['max_speed_px_per_sec']:.0f}"
            cv2.putText(frame, text, (x1 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1)

        return frame

    def draw_annotation(self,video_frame,tracks,team_ball_control,match_summary):

        output_video_frames=[]
        for frame_num,frame in enumerate(video_frame):
            frame=frame.copy()

            player_dict=tracks["players"][frame_num]
            ball_dict=tracks["ball"][frame_num]
            referee_dict=tracks["referees"][frame_num]

            #Draw Players
            for track_id,player in player_dict.items():
                color=player.get("team_color",(0,0,255))
                frame=self.draw_ellipse(frame,player["bbox"],color,track_id)

                if player.get("has_ball",False):
                    frame=self.draw_triangle(frame,player["bbox"],(0,0,255))


            #Draw Referee
            for _,referee in referee_dict.items():
                frame=self.draw_ellipse(frame,referee["bbox"],(0,255,255))

            #Draw Ball  
            for track_id,ball in ball_dict.items():
                frame=self.draw_triangle(frame,ball["bbox"],(0,255,0))

            frame = self.draw_ball_trail(frame, tracks["ball"], frame_num)
            frame = self.draw_minimap(frame, frame_num, tracks, match_summary)
            frame = self.draw_leaderboard(frame, match_summary)

            #Draw Team Control
            frame = self.draw_team_ball_control(frame,frame_num,team_ball_control)
            frame = self.draw_possession_timeline(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)
        
        return output_video_frames
    

