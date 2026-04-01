from utils import read_video, save_video, build_match_summary, export_match_summary, export_player_heatmaps, export_team_formations
from trackers import Tracker
import numpy as np
import argparse
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner


def parse_args():
    parser = argparse.ArgumentParser(description="Football video analysis pipeline")
    parser.add_argument("--input-video", default="input_videos/08fd33_4.mp4")
    parser.add_argument("--model-path", default="models/best.pt")
    parser.add_argument("--stub-path", default="stubs/track_stubs.pkl")
    parser.add_argument("--output-video", default="output_videos/output_video.avi")
    parser.add_argument("--summary-json", default="output_videos/match_summary.json")
    parser.add_argument("--player-stats-csv", default="output_videos/player_stats.csv")
    parser.add_argument("--heatmap-dir", default="output_videos/heatmaps")
    parser.add_argument("--formations-image", default="output_videos/team_formations.png")
    parser.add_argument("--read-from-stub", dest="read_from_stub", action="store_true")
    parser.add_argument("--no-read-from-stub", dest="read_from_stub", action="store_false")
    parser.set_defaults(read_from_stub=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Read video
    video_frames, video_fps = read_video(args.input_video)

    # Initialize tracker
    tracker=Tracker(args.model_path)

    tracks=tracker.get_object_tracks(video_frames,
                                     read_from_stub=args.read_from_stub,
                                     stub_path=args.stub_path)

    #Interpolate Ball Positions
    tracks["ball"]=tracker.interpolate_ball_positions(tracks["ball"])


    # Assign Player Teams
    teams_assigner = TeamAssigner()
    teams_assigner.assign_team_color(video_frames,tracks["players"])

    for frame_num,player_track in enumerate(tracks["players"]):
        for player_id,track in player_track.items():
            team = teams_assigner.get_player_team(video_frames[frame_num],track["bbox"],player_id)


            tracks["players"][frame_num][player_id]["team"]=team
            tracks["players"][frame_num][player_id]["team_color"]=teams_assigner.team_colors[team]

    #Assign Ball Aquisition
    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num,player_track in enumerate (tracks["players"]):
        ball_track = tracks["ball"][frame_num].get(1)
        assigned_player = -1

        if ball_track is not None:
            assigned_player=player_assigner.assign_ball_to_players(player_track,ball_track["bbox"])

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"]=True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        elif team_ball_control:
            team_ball_control.append(team_ball_control[-1])
        else:
            team_ball_control.append(0)
    team_ball_control=np.array(team_ball_control)


    #Draw Output
    ## Draw object tracks
    match_summary = build_match_summary(tracks, team_ball_control, video_fps, video_frames[0].shape[:2])
    output_video_frames=tracker.draw_annotation(video_frames,tracks,team_ball_control,match_summary)

    #save video
    save_video(output_video_frames,args.output_video, fps=video_fps)

    export_match_summary(match_summary, args.summary_json, args.player_stats_csv)
    export_player_heatmaps(tracks, video_frames[0].shape[:2], args.heatmap_dir)
    export_team_formations(match_summary, args.formations_image)


if __name__ == '__main__':
    main()
