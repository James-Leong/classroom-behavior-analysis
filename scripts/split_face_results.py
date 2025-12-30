import json
import os
import sys

print("Script started", flush=True)


def split_face_results(input_path, output_dir, chunk_duration_sec=60):
    print(f"Loading {input_path}...", flush=True)
    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading json: {e}", flush=True)
        return

    meta = data.get("meta", {})
    frames = data.get("frames", [])
    tracklets = data.get("tracklets", [])

    # Handle frames whether it's list or dict
    frames_list = []
    if isinstance(frames, dict):
        # Convert dict to list of (key, value) or just values with timestamps
        for k, v in frames.items():
            # Ensure timestamp is present
            if "timestamp" not in v:
                try:
                    v["timestamp"] = float(k)
                except:
                    pass
            frames_list.append(v)
    elif isinstance(frames, list):
        frames_list = frames
    else:
        print(f"Unknown frames type: {type(frames)}", flush=True)
        return

    print(f"Loaded {len(frames_list)} frames. Processing...", flush=True)

    fps = meta.get("fps", 30)
    duration = meta.get("duration", 0)

    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Remove heavy fields from tracklets in manifest to save space
    for t in tracklets:
        if "bbox_history" in t:
            del t["bbox_history"]
        if "representative_frames" in t:
            del t["representative_frames"]

    manifest = {"meta": meta, "tracklets": tracklets, "chunks": []}

    chunk_frames = {}

    for frame_data in frames_list:
        timestamp = frame_data.get("timestamp")
        if timestamp is None:
            # Try to infer from frame_index if available
            frame_idx = frame_data.get("frame_index")
            if frame_idx is not None and fps > 0:
                timestamp = frame_idx / fps
            else:
                continue

        chunk_idx = int(timestamp // chunk_duration_sec)

        if chunk_idx not in chunk_frames:
            chunk_frames[chunk_idx] = []

        chunk_frames[chunk_idx].append(frame_data)

    # Save chunks
    # We will save chunks as a Map (Record<string, FrameData>) keyed by timestamp (rounded) or frame index?
    # To make frontend lookup easy O(1), let's save as Record<string, FrameData> where key is "timestamp" string or frame index.
    # The frontend usually queries by timestamp.

    for chunk_idx, frames_subset in chunk_frames.items():
        chunk_filename = f"face_chunk_{chunk_idx}.json"
        chunk_path = os.path.join(chunks_dir, chunk_filename)

        # Convert list back to dict for O(1) lookup in frontend
        # Key = timestamp string (to 2 decimal places?) or frame index?
        # Let's use stringified timestamp for now, as that's what `types/index.ts` hinted (Record<string, ...>)
        frames_dict = {}
        for f in frames_subset:
            ts = f.get("timestamp")
            frame_idx = f.get("frame_index")
            if frame_idx is None and ts is not None and fps > 0:
                frame_idx = int(round(ts * fps))

            if frame_idx is not None:
                frames_dict[str(frame_idx)] = f
            elif ts is not None:
                # Fallback to timestamp if no frame index
                frames_dict[str(ts)] = f

        with open(chunk_path, "w") as f:
            json.dump({"frames": frames_dict}, f)

        manifest["chunks"].append(
            {
                "chunk_id": chunk_idx,
                "start_time": chunk_idx * chunk_duration_sec,
                "end_time": (chunk_idx + 1) * chunk_duration_sec,
                "file": f"chunks/{chunk_filename}",
            }
        )

        print(f"Saved chunk {chunk_idx} ({len(frames_subset)} frames)", flush=True)

    manifest_path = os.path.join(output_dir, "face_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"Saved manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    INPUT_FILE = "fronts/public/data/outputs/face_results_1h.json"
    OUTPUT_DIR = "fronts/public/data/outputs"

    print(f"Checking {INPUT_FILE}...", flush=True)
    if os.path.exists(INPUT_FILE):
        split_face_results(INPUT_FILE, OUTPUT_DIR)
    else:
        print(f"Input file not found: {INPUT_FILE}", flush=True)
