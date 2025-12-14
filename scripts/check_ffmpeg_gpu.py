#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode, out.strip()
    except Exception as e:
        return 999, f"ERROR: {e}"


def main() -> int:
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg: {ffmpeg or 'NOT FOUND'}")
    if not ffmpeg:
        print("Install ffmpeg first (e.g. apt/yum/pacman) if you want transcoding.")
        return 2

    code, out = _run(["ffmpeg", "-hide_banner", "-version"], timeout=10)
    print("\n== ffmpeg -version ==")
    print(out.splitlines()[0] if out else f"(exit={code})")

    code, enc = _run(["ffmpeg", "-hide_banner", "-encoders"], timeout=15)
    has_nvenc = "h264_nvenc" in enc
    has_qsv = "h264_qsv" in enc
    has_vaapi = "h264_vaapi" in enc

    print("\n== encoder availability ==")
    print(f"h264_nvenc: {'YES' if has_nvenc else 'NO'}")
    print(f"h264_qsv  : {'YES' if has_qsv else 'NO'}")
    print(f"h264_vaapi: {'YES' if has_vaapi else 'NO'}")

    code, out = _run(["nvidia-smi"], timeout=10) if shutil.which("nvidia-smi") else (127, "")
    print("\n== nvidia-smi ==")
    if code == 0:
        print("OK (GPU driver present)")
        print("\n".join(out.splitlines()[:8]))
    else:
        print("NOT FOUND / NOT WORKING (this is OK on non-NVIDIA machines)")

    if has_nvenc:
        print("\n== quick NVENC smoke test (1s synthetic) ==")
        # Writes nothing to disk; fails fast if NVENC isn't actually usable.
        code, out = _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=1280x720:rate=30",
                "-t",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            timeout=20,
        )
        if code == 0:
            print("NVENC encode OK")
        else:
            print("NVENC encode FAILED")
            print(out)

    print("\nDone.")
    print("If you want pytest to force transcoding, set:")
    print("  PYTEST_FFMPEG_CODEC=h264_nvenc  (or libx264)")
    print("To disable even when available:")
    print("  PYTEST_FFMPEG_CODEC=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
