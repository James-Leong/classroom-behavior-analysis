# Frontend Resources

This directory contains the frontend application and its static resources for the Classroom Behavior Analysis visualization.

## Directory Structure

- `assets/`: Static assets
  - `frames/`: Keyframe images (extracted from video)
  - `clips/`: Video clips (short segments)
  - `overlays/`: Pre-rendered overlay images
  - `indexes/`: Index files for large JSON data
- `index.html`: Entry point for the visualization application
- `app.js`: Main application logic (or build output)
- `styles.css`: Styles

## Usage

This frontend is designed to run statically or with a simple HTTP server. It consumes the JSON outputs from the backend pipeline (located in `../outputs/` or similar, usually copied or symlinked here or fetched via API).

For large file handling and "Simulation Mode", ensure `debug_trace.json` and other artifacts are available.
