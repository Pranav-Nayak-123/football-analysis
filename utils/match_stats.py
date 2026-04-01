import csv
import json
from pathlib import Path

from .bbox_utils import get_center_of_bbox, measure_distance


def _get_player_anchor(bbox):
    x_center, _ = get_center_of_bbox(bbox)
    return x_center, int(bbox[3])


def _normalize_point(point, frame_width, frame_height):
    if frame_width <= 1 or frame_height <= 1:
        return [0.0, 0.0]

    normalized_x = min(max(point[0] / (frame_width - 1), 0.0), 1.0)
    normalized_y = min(max(point[1] / (frame_height - 1), 0.0), 1.0)
    return [round(normalized_x, 4), round(normalized_y, 4)]


def _build_possession_segments(team_ball_control, fps):
    if not team_ball_control:
        return []

    segments = []
    current_team = int(team_ball_control[0])
    start_frame = 0

    for frame_index, team_id in enumerate(team_ball_control[1:], start=1):
        if int(team_id) != current_team:
            end_frame = frame_index - 1
            segments.append(
                {
                    "team": current_team,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_seconds": round((end_frame - start_frame + 1) / fps, 2),
                }
            )
            current_team = int(team_id)
            start_frame = frame_index

    segments.append(
        {
            "team": current_team,
            "start_frame": start_frame,
            "end_frame": len(team_ball_control) - 1,
            "duration_seconds": round((len(team_ball_control) - start_frame) / fps, 2),
        }
    )
    return segments


def build_match_summary(tracks, team_ball_control, fps, frame_size):
    frame_height, frame_width = frame_size
    player_stats = {}

    for frame_players in tracks["players"]:
        current_positions = {}
        for player_id, player in frame_players.items():
            anchor = _get_player_anchor(player["bbox"])
            stats = player_stats.setdefault(
                player_id,
                {
                    "team": int(player.get("team", 0)),
                    "frames_tracked": 0,
                    "distance_px": 0.0,
                    "max_speed_px_per_sec": 0.0,
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "last_position": None,
                },
            )

            stats["frames_tracked"] += 1
            if stats["team"] == 0:
                stats["team"] = int(player.get("team", 0))

            stats["sum_x"] += anchor[0]
            stats["sum_y"] += anchor[1]

            if stats["last_position"] is not None:
                step_distance = measure_distance(stats["last_position"], anchor)
                stats["distance_px"] += step_distance
                stats["max_speed_px_per_sec"] = max(stats["max_speed_px_per_sec"], step_distance * fps)

            current_positions[player_id] = anchor

        for player_id, anchor in current_positions.items():
            player_stats[player_id]["last_position"] = anchor

    possession_counts = {0: 0, 1: 0, 2: 0}
    for team_id in team_ball_control.tolist():
        possession_counts[int(team_id)] = possession_counts.get(int(team_id), 0) + 1

    tracked_possession_frames = possession_counts[1] + possession_counts[2]
    possession_summary = {
        "unassigned_frames": possession_counts[0],
        "team_1_frames": possession_counts[1],
        "team_2_frames": possession_counts[2],
        "team_1_pct": (possession_counts[1] / tracked_possession_frames * 100) if tracked_possession_frames else 0.0,
        "team_2_pct": (possession_counts[2] / tracked_possession_frames * 100) if tracked_possession_frames else 0.0,
    }

    player_rows = []
    for player_id in sorted(player_stats):
        stats = player_stats[player_id]
        average_position = (
            stats["sum_x"] / stats["frames_tracked"],
            stats["sum_y"] / stats["frames_tracked"],
        )
        player_rows.append(
            {
                "player_id": int(player_id),
                "team": stats["team"],
                "frames_tracked": stats["frames_tracked"],
                "distance_px": round(stats["distance_px"], 2),
                "max_speed_px_per_sec": round(stats["max_speed_px_per_sec"], 2),
                "avg_position_norm": _normalize_point(average_position, frame_width, frame_height),
            }
        )

    top_movers = sorted(player_rows, key=lambda row: row["distance_px"], reverse=True)[:5]
    top_speeds = sorted(player_rows, key=lambda row: row["max_speed_px_per_sec"], reverse=True)[:5]
    possession_segments = _build_possession_segments(team_ball_control.tolist(), fps)

    team_shapes = {1: [], 2: []}
    for player in player_rows:
        if player["team"] in team_shapes:
            team_shapes[player["team"]].append(
                {
                    "player_id": player["player_id"],
                    "avg_position_norm": player["avg_position_norm"],
                    "frames_tracked": player["frames_tracked"],
                }
            )

    for team_id in team_shapes:
        team_shapes[team_id] = sorted(
            team_shapes[team_id],
            key=lambda row: row["frames_tracked"],
            reverse=True,
        )[:11]

    return {
        "total_frames": len(tracks["players"]),
        "fps": round(fps, 2),
        "duration_seconds": round(len(tracks["players"]) / fps, 2) if fps else 0.0,
        "frame_size": {"width": frame_width, "height": frame_height},
        "possession": possession_summary,
        "possession_segments": possession_segments,
        "top_movers": top_movers,
        "top_speeds": top_speeds,
        "team_shapes": team_shapes,
        "players": player_rows,
    }


def export_match_summary(summary, json_output_path, csv_output_path):
    json_path = Path(json_output_path)
    csv_path = Path(csv_output_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["player_id", "team", "frames_tracked", "distance_px", "max_speed_px_per_sec", "avg_position_norm"],
        )
        writer.writeheader()
        rows = []
        for player in summary["players"]:
            row = dict(player)
            row["avg_position_norm"] = f"{player['avg_position_norm'][0]},{player['avg_position_norm'][1]}"
            rows.append(row)
        writer.writerows(rows)
