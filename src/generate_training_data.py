"""Generate labeled synthetic training data for the AI detector."""

import numpy as np
import cv2
import json
import os
from pathlib import Path

# Import from sibling modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from sim import Simulator, CameraState, CircularMotion, SinusoidalMotion
from detect import Detection


def generate_dataset(output_dir, num_frames=5000,
                     seeds=None, scenarios=None):
    """Generate labeled frames across scenarios and seeds.

    Args:
        output_dir: Directory to save frames and labels
        num_frames: Total frames to generate
        seeds: Random seeds for reproducibility
        scenarios: Scenario names ("clean", "hard")
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 101]
    if scenarios is None:
        scenarios = ["clean", "hard"]

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    labels = []
    frame_count = 0
    frames_per_seed = num_frames // (len(seeds) * len(scenarios))

    for scenario in scenarios:
        for seed in seeds:
            sim = _create_simulator(scenario, seed)

            for i in range(frames_per_seed):
                frame, gt = sim.step(frame_count)

                # Save frame
                fname = "frame_{:06d}.jpg".format(frame_count)
                cv2.imwrite(os.path.join(frames_dir, fname), frame)

                # Save label (bounding box from ground truth)
                if gt.target_visible:
                    # Create bbox around beacon (approximate from Gaussian sigma)
                    sigma = 4.0  # default beacon sigma
                    x1 = max(0, int(gt.target_pixel_x - 3 * sigma))
                    y1 = max(0, int(gt.target_pixel_y - 3 * sigma))
                    x2 = min(frame.shape[1], int(gt.target_pixel_x + 3 * sigma))
                    y2 = min(frame.shape[0], int(gt.target_pixel_y + 3 * sigma))

                    labels.append({
                        "image": fname,
                        "bbox": [x1, y1, x2, y2],
                        "center": [gt.target_pixel_x, gt.target_pixel_y],
                        "visible": True,
                        "scenario": scenario,
                        "seed": seed,
                    })
                else:
                    labels.append({
                        "image": fname,
                        "bbox": [],
                        "center": [],
                        "visible": False,
                        "scenario": scenario,
                        "seed": seed,
                    })

                frame_count += 1

    # Save labels
    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    # Save metadata
    meta = {
        "total_frames": frame_count,
        "num_positive": sum(1 for l in labels if l["visible"]),
        "num_negative": sum(1 for l in labels if not l["visible"]),
        "seeds": seeds,
        "scenarios": scenarios,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Generated {} frames ({} positive, {} negative)".format(
        frame_count, meta["num_positive"], meta["num_negative"]))
    print("Saved to {}".format(output_dir))
    return meta


def _create_simulator(scenario, seed):
    """Create a simulator for a given scenario."""
    from config import Config
    config = Config(random_seed=seed)

    cam = CameraState(
        width=config.width, height=config.height,
        h_fov_deg=config.h_fov_deg, v_fov_deg=config.v_fov_deg,
    )

    if scenario == "clean":
        from sim import CircularMotion
        motion = CircularMotion(
            center_az_deg=config.clean_center_az,
            center_el_deg=config.clean_center_el,
            radius_deg=config.clean_radius,
            angular_speed_deg_s=config.clean_speed,
        )
    else:
        from sim import SinusoidalMotion
        motion = SinusoidalMotion(
            az_amp_deg=config.hard_az_amp,
            az_freq_hz=config.hard_az_freq,
            el_amp_deg=config.hard_el_amp,
            el_freq_hz=config.hard_el_freq,
        )

    return Simulator(
        cam=cam, motion=motion,
        beacon_sigma_px=config.beacon_sigma_px,
        beacon_peak=config.beacon_peak,
        fps=config.fps,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training_data")
    parser.add_argument("--frames", type=int, default=5000)
    args = parser.parse_args()
    generate_dataset(args.output, args.frames)
