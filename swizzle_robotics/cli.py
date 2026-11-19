from __future__ import annotations
import argparse
import sys
from .hardware import HardwareDetector
from .models import get_policy, get_world_model, MODEL_REGISTRY
from .servoing import VisualServoingLoop
from .core.camera import LiveCamera, MockCamera
from .server import run_server

def main():
    parser = argparse.ArgumentParser(
        prog="swizzle-robotics",
        description="Swizzle Robotics: 80.5 Hz Visual Servoing, Latent World Models & Edge Accelerators."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: detect
    subparsers.add_parser("detect", help="Detect local GPU hardware and recommend models")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run closed-loop visual servoing")
    run_parser.add_argument("--model", type=str, default="lerobot/smolvla-450m", help="VLA model identifier")
    run_parser.add_argument("--steps", type=int, default=30, help="Number of control steps to execute")
    run_parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    run_parser.add_argument("--mock", action="store_true", help="Force synthetic mock camera")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch Swizzle Robotics REST API server")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")

    # Command: arena
    arena_parser = subparsers.add_parser("arena", help="Run closed-loop MuJoCo manipulation arena with CBF + MCAP recording")
    arena_parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    arena_parser.add_argument("--steps", type=int, default=40, help="Max steps per episode")
    arena_parser.add_argument("--record-dir", type=str, default="", help="Optional output folder for .mcap and LeRobot v2.1 episodes")

    # Command: arena-server
    arena_srv_parser = subparsers.add_parser("arena-server", help="Launch Real-Life Robot Arena FastAPI + WebSocket tuning backend")
    arena_srv_parser.add_argument("--port", type=int, default=8787, help="Port to listen on (default: 8787)")
    arena_srv_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")

    args = parser.parse_args()

    if args.command == "detect" or not args.command:
        profile = HardwareDetector.detect()
        print("=" * 60)
        print("SWIZZLE ROBOTICS HARDWARE & MODEL PROFILE")
        print("=" * 60)
        print(f"Device Type:           {profile.device_type.upper()}")
        print(f"Device Name:           {profile.device_name}")
        print(f"Detected VRAM:         {profile.vram_gb:.2f} GB")
        print(f"Recommended VLA:       {profile.recommended_vla}")
        print(f"Recommended World Mod: {profile.recommended_world_model}")
        print(f"Quantization Target:   {profile.recommended_quantization}")
        print(f"Needs Colab Tunnel:    {'YES (7B requires tunnel)' if profile.needs_colab_tunnel_for_7b else 'NO (Fully local)'}")
        print("=" * 60)

    elif args.command == "serve":
        run_server(port=args.port, host=args.host)

    elif args.command == "run":
        print(f"Initializing Swizzle Visual Servoing with policy '{args.model}'...")
        policy = get_policy(args.model)
        camera = MockCamera() if args.mock else LiveCamera(camera_id=args.camera)
        loop = VisualServoingLoop(policy=policy, camera=camera, target_hz=80.5)

        print(f"Executing {args.steps} closed-loop servoing steps:")
        print(f"{'Step':<6} | {'Hz':<8} | {'Latency':<10} | {'Patch Reuse':<12} | {'SpecAct':<8} | {'7-DoF Actuation [dx, dy, dz, ...]':<30}")
        print("-" * 85)

        for s in range(args.steps):
            telemetry = loop.step("Pick up target object")
            act_str = f"[{', '.join(f'{x:.2f}' for x in telemetry.action_7dof[:3])}, ...]"
            print(f"{telemetry.step:<6} | {telemetry.effective_hz:<8.1f} | {telemetry.dt_ms:<7.2f} ms | {telemetry.patch_hit_ratio * 100:<10.1f}% | {telemetry.spec_act_rate * 100:<6.0f}% | {act_str}")

    elif args.command == "arena":
        import numpy as np
        from pathlib import Path
        from .envs import ArenaTaskConfig, SwizzleManipulationArena
        from .core.hierarchical import HierarchicalController
        from .core.safety import SafetyConfig

        cfg = ArenaTaskConfig(max_steps=args.steps)
        arena = SwizzleManipulationArena(cfg, prefer_mujoco=True)
        print(f"Swizzle Manipulation Arena backend={arena.backend_mode} | episodes={args.episodes}")
        for ep in range(args.episodes):
            obs = arena.reset(seed=42 + ep)
            ctrl = HierarchicalController(
                policy=get_policy("swizzle/edge-flow-vla"),
                safety_config=SafetyConfig(z_bounds=(0.02, 0.70), max_linear_step=0.22, max_accel_step=0.14),
                episode_id=f"ep_{ep:06d}",
            )
            plan = ctrl.plan_s2(
                "pick up red cube while avoiding pillar then place on green tray",
                known_objects={"pillar": (*cfg.obstacle_pos, cfg.obstacle_radius)},
            )
            collisions = 0
            for _ in range(args.steps):
                target = np.array(obs.cube_pos if not obs.grasped else obs.goal_pos)
                err = (target - np.array(obs.eef_pose[:3])) * 4.0
                grip = 1.0 if float(np.linalg.norm(target - np.array(obs.eef_pose[:3]))) < 0.09 or obs.grasped else -1.0
                sec = ctrl.safety_shield.project([float(err[0]), float(err[1]), float(err[2]), 0.0, 0.0, 0.0, grip], obs.eef_pose)
                obs, _, done, info = arena.step(sec.safe_action)
                ctrl.recorder.record_step(obs.frame, obs.eef_pose, sec, prompt=plan.subtasks[0])
                if info["obstacle_collision"] or info["table_penetration_mm"] > 0.0:
                    collisions += 1
                if done:
                    break
            print(f"Episode {ep}: steps={len(ctrl.recorder.steps)}, collisions={collisions}, goal_dist={info['cube_to_goal_dist']}, success={info['success']}")
            if args.record_dir:
                out_root = Path(args.record_dir)
                ctrl.recorder.export_mcap(out_root / f"ep_{ep:06d}.mcap")
                ctrl.recorder.export_lerobot_v2(out_root / "lerobot_dataset", episode_index=ep)
        arena.close()

    elif args.command == "arena-server":
        from .arena_backend import run_arena_server
        run_arena_server(port=args.port, host=args.host)

if __name__ == "__main__":
    main()
