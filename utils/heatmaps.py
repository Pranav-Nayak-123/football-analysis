from pathlib import Path

import cv2
import numpy as np


def _create_pitch_canvas(width=720, height=420):
    canvas = np.full((height, width, 3), (35, 120, 55), dtype=np.uint8)
    line_color = (235, 235, 235)
    margin = 24

    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), line_color, 2)
    cv2.line(canvas, (width // 2, margin), (width // 2, height - margin), line_color, 2)
    cv2.circle(canvas, (width // 2, height // 2), 45, line_color, 2)
    cv2.rectangle(canvas, (margin, height // 2 - 70), (margin + 80, height // 2 + 70), line_color, 2)
    cv2.rectangle(canvas, (width - margin - 80, height // 2 - 70), (width - margin, height // 2 + 70), line_color, 2)

    return canvas


def _normalized_to_pitch(point, width, height, margin=24):
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin
    x = int(margin + point[0] * usable_width)
    y = int(margin + point[1] * usable_height)
    return x, y


def export_player_heatmaps(tracks, frame_size, output_dir, pitch_size=(720, 420)):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    player_points = {}
    team_points = {1: [], 2: []}
    frame_height, frame_width = frame_size
    pitch_width, pitch_height = pitch_size

    for frame_players in tracks["players"]:
        for player_id, player in frame_players.items():
            x = ((player["bbox"][0] + player["bbox"][2]) / 2) / max(frame_width, 1)
            y = player["bbox"][3] / max(frame_height, 1)
            point = (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))
            player_points.setdefault(
                player_id,
                {"team": int(player.get("team", 0)), "points": []},
            )["points"].append(point)

            team_id = int(player.get("team", 0))
            if team_id in team_points:
                team_points[team_id].append(point)

    def render_heatmap(points, title, output_file):
        pitch = _create_pitch_canvas(pitch_width, pitch_height)
        heat = np.zeros((pitch_height, pitch_width), dtype=np.float32)

        for point in points:
            x, y = _normalized_to_pitch(point, pitch_width, pitch_height)
            cv2.circle(heat, (x, y), 18, 1.0, -1)

        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=18, sigmaY=18)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_uint8 = np.uint8(255 * heat)
        colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(pitch, 0.55, colored, 0.45, 0)
        cv2.putText(blended, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
        cv2.imwrite(str(output_file), blended)

    for player_id, info in player_points.items():
        if not info["points"]:
            continue
        render_heatmap(
            info["points"],
            f"Player {player_id} Heatmap",
            output_path / f"player_{player_id}_heatmap.png",
        )

    for team_id, points in team_points.items():
        if not points:
            continue
        render_heatmap(points, f"Team {team_id} Heatmap", output_path / f"team_{team_id}_heatmap.png")


def export_team_formations(match_summary, output_path, pitch_size=(720, 420)):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pitch_width, pitch_height = pitch_size
    pitch = _create_pitch_canvas(pitch_width, pitch_height)
    colors = {1: (255, 120, 80), 2: (80, 170, 255)}

    cv2.putText(pitch, "Average Team Formations", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)

    for team_id, players in match_summary.get("team_shapes", {}).items():
        color = colors.get(int(team_id), (245, 245, 245))
        for player in players:
            x, y = _normalized_to_pitch(player["avg_position_norm"], pitch_width, pitch_height)
            cv2.circle(pitch, (x, y), 13, color, 2)
            cv2.putText(
                pitch,
                str(player["player_id"]),
                (x - 9, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    cv2.putText(pitch, "Team 1", (24, pitch_height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[1], 2)
    cv2.putText(pitch, "Team 2", (140, pitch_height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[2], 2)
    cv2.imwrite(str(output_file), pitch)
