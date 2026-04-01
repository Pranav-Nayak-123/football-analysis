# Football Analysis

A computer-vision football analytics project that tracks players, estimates team possession, assigns teams by kit color, and renders an annotated match video with extra analytics overlays.

## Features

- Player, referee, and ball tracking
- Team-color assignment
- Ball possession estimation
- Annotated output video with:
  - live mini-map
  - ball trail
  - possession timeline
  - top-movers leaderboard
- Match summary export as JSON and CSV
- Player and team heatmap image export
- Separate team formations image export

## Project Structure

- `main.py`: main pipeline entry point
- `trackers/`: object tracking and video overlays
- `team_assigner/`: jersey-color clustering and team assignment
- `player_ball_assigner/`: ball-to-player possession logic
- `utils/`: video helpers, analytics summaries, and heatmap exports

## Setup

```bash
pip install -r requirements.txt
```

## Run

Use cached tracks:

```bash
python main.py --read-from-stub
```

Run fresh detection:

```bash
python main.py --no-read-from-stub
```

## Outputs

The pipeline writes generated results to `output_videos/`, including:

- `output_video.avi`
- `match_summary.json`
- `player_stats.csv`
- `team_formations.png`
- `heatmaps/`

## Notes

- Large local assets such as models, training data, videos, and generated outputs are ignored in Git for a cleaner public repository.
- To run the full pipeline without stubs, install the dependencies in `requirements.txt` and provide the required model/video assets locally.
