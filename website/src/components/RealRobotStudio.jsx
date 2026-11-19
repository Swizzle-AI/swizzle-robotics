import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertOctagon,
  Check,
  Copy,
  Download,
  Pause,
  Play,
  Plus,
  Radio,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Trash2,
  Zap,
} from 'lucide-react';

const DEFAULT_PARAMS = {
  cbf_enabled: true,
  cbf_alpha: 0.65,
  max_linear_step: 0.22,
  max_angular_step: 0.35,
  max_accel_step: 0.14,
  eef_radius: 0.045,
  x_bounds: [-0.55, 0.55],
  y_bounds: [-0.55, 0.55],
  z_bounds: [0.03, 0.7],
  watchdog_timeout_steps: 5,
  policy_model: 'swizzle/edge-flow-vla',
  target_hz: 60.0,
  decay_lambda: 0.06,
  rtc_substeps: 4,
  patch_delta_threshold: 0.025,
  action_gain: 1.0,
  instruction: 'pick up target cube while avoiding obstacle pillar then place on goal tray',
  obstacles: [{ name: 'pillar', center: [0.0, 0.0, 0.24], radius: 0.065 }],
};

const ADAPTER_PRESETS = [
  {
    preset_id: 'so100',
    name: 'LeRobot SO-ARM100',
    adapter: 'serial_so100',
    default_endpoint: 'serial://COM4@1000000',
    dof: 6,
    recommended_cbf_alpha: 0.55,
    recommended_max_linear_step: 0.16,
  },
  {
    preset_id: 'franka_fr3',
    name: 'Franka FR3 (FCI / Zenoh)',
    adapter: 'franka_fci',
    default_endpoint: 'tcp://172.16.0.2:7447',
    dof: 7,
    recommended_cbf_alpha: 0.65,
    recommended_max_linear_step: 0.22,
  },
  {
    preset_id: 'ur5e',
    name: 'Universal Robots UR5e',
    adapter: 'ur_rtde',
    default_endpoint: 'rtde://192.168.1.102:30004',
    dof: 7,
    recommended_cbf_alpha: 0.6,
    recommended_max_linear_step: 0.2,
  },
  {
    preset_id: 'unitree_g1',
    name: 'Unitree G1 (Zenoh S2->S1->S0)',
    adapter: 'ros2_zenoh',
    default_endpoint: 'tcp/192.168.123.164:7447',
    dof: 14,
    recommended_cbf_alpha: 0.72,
    recommended_max_linear_step: 0.25,
  },
];

const POLICY_OPTIONS = [
  'swizzle/edge-flow-vla',
  'lerobot/smolvla-450m',
  'physical-intelligence/pi0-fast',
  'nvidia/gr00t-manipulator-v1',
  'openvla/openvla-7b-swizzle',
];

function SliderField({ label, value, min, max, step, unit = '', hint, onChange }) {
  return (
    <div className="rounded-lg border border-white/[0.07] bg-obsidian/70 p-2.5">
      <div className="flex items-center justify-between gap-2 mb-1">
        <span className="font-mono text-[11px] text-slate-200">{label}</span>
        <span className="font-mono text-[11px] font-semibold text-limebrand">
          {typeof value === 'number' ? value.toFixed(step < 0.01 ? 3 : step < 1 ? 2 : 0) : value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-white/10 rounded-lg appearance-none cursor-pointer accent-[#C9FF3B]"
      />
      {hint && <div className="font-mono text-[10px] text-slate-500 mt-1">{hint}</div>}
    </div>
  );
}

export default function RealRobotStudio() {
  const [backendStatus, setBackendStatus] = useState('connecting'); // 'connected' | 'offline' | 'connecting'
  const [backendBaseUrl, setBackendBaseUrl] = useState('');
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [robots, setRobots] = useState([
    {
      robot_id: 'local-mujoco-hil',
      name: 'Swizzle Workcell Twin (MuJoCo HIL)',
      adapter: 'mujoco_hil',
      endpoint: 'local://kinematic-twin',
      dof: 7,
      connected: true,
      live_ws: false,
      ticks_served: 0,
    },
  ]);
  const [activeRobotId, setActiveRobotId] = useState('local-mujoco-hil');
  const [estopActive, setEstopActive] = useState(false);
  const [running, setRunning] = useState(true);
  const [tuneTab, setTuneTab] = useState('safety'); // 'safety' | 'rtc' | 'obstacles'
  const [showConnectForm, setShowConnectForm] = useState(false);

  // Connect Robot Form state
  const [newRobotId, setNewRobotId] = useState('franka-cell-01');
  const [newRobotName, setNewRobotName] = useState('Franka FR3 Cell 01');
  const [newRobotAdapter, setNewRobotAdapter] = useState('franka_fci');
  const [newRobotEndpoint, setNewRobotEndpoint] = useState('tcp://172.16.0.2:7447');
  const [newRobotDof, setNewRobotDof] = useState(7);

  // New obstacle inputs
  const [obsName, setObsName] = useState('fixture_01');
  const [obsX, setObsX] = useState('0.14');
  const [obsY, setObsY] = useState('-0.12');
  const [obsRadius, setObsRadius] = useState('0.07');

  // Telemetry & oscilloscope history
  const [telemetry, setTelemetry] = useState({
    pose: [-0.26, -0.18, 0.24, 0, 0, 0],
    target: [0.22, 0.14, 0.14],
    raw_action: [0.25, 0.18, 0, 0, 0, 0, -1],
    safe_action: [0.22, 0.15, 0, 0, 0, 0, -1],
    min_barrier_margin: 0.084,
    cbf_intervened: false,
    clamped_axes: [],
    intervention_reasons: [],
    solver_latency_ms: 0.125,
    patch_hit_ratio: 0.964,
    effective_hz: 60.0,
    collisions: 0,
    interventions_count: 0,
    step_index: 0,
    receipt_digest: 'sha256:9f38b1e4d0c7...',
    backend_mode: 'standalone-preview',
    obstacles: DEFAULT_PARAMS.obstacles,
  });
  const [events, setEvents] = useState([
    {
      ts: Date.now() / 1000,
      step: 0,
      kind: 'INFO',
      message: 'Connect backend via `swizzle-robotics arena-server --port 8787` or test parameters live below.',
    },
  ]);
  const [exportBanner, setExportBanner] = useState(null);
  const [copiedCmd, setCopiedCmd] = useState(false);

  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const trailRef = useRef([]);
  const marginHistoryRef = useRef([]);
  const offlineStateRef = useRef({
    pose: [-0.26, -0.18, 0.24],
    target: [0.22, 0.14, 0.14],
    vel: [0, 0, 0],
    step: 0,
    impulse: [0, 0, 0],
  });

  const apiBase = useMemo(() => backendBaseUrl.replace(/\/$/, ''), [backendBaseUrl]);

  // Connect to /ws/studio if backend is running; otherwise run realistic offline twin seamlessly
  useEffect(() => {
    let closed = false;
    let reconnectTimer = null;

    const connectWs = () => {
      if (closed) return;
      const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = apiBase
        ? `${apiBase.replace(/^http/, 'ws')}/ws/studio`
        : `${wsProto}//${window.location.host}/ws/studio`;

      try {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!closed) setBackendStatus('connected');
        };

        ws.onmessage = (evt) => {
          try {
            const msg = JSON.parse(evt.data);
            if (msg.type === 'state_sync' && msg.snapshot) {
              const s = msg.snapshot;
              if (s.params) setParams(s.params);
              if (s.robots) setRobots(s.robots);
              if (s.active_robot_id) setActiveRobotId(s.active_robot_id);
              setEstopActive(Boolean(s.estop_active));
              setRunning(Boolean(s.running));
              if (s.events) setEvents(s.events);
              if (s.last_telemetry && Object.keys(s.last_telemetry).length > 0) {
                setTelemetry(s.last_telemetry);
              }
            } else if (msg.type === 'telemetry') {
              setTelemetry(msg);
              setEstopActive(Boolean(msg.estop_active));
              trailRef.current.push(msg.pose.slice(0, 2));
              if (trailRef.current.length > 70) trailRef.current.shift();
              marginHistoryRef.current.push({
                margin: Number(msg.min_barrier_margin ?? 0.05),
                rawMag: Math.hypot(msg.raw_action?.[0] ?? 0, msg.raw_action?.[1] ?? 0),
                safeMag: Math.hypot(msg.safe_action?.[0] ?? 0, msg.safe_action?.[1] ?? 0),
              });
              if (marginHistoryRef.current.length > 56) marginHistoryRef.current.shift();
            }
          } catch {
            // ignore malformed frame
          }
        };

        ws.onerror = () => {
          if (!closed) setBackendStatus('offline');
        };

        ws.onclose = () => {
          if (wsRef.current === ws) wsRef.current = null;
          if (!closed) {
            setBackendStatus('offline');
            reconnectTimer = setTimeout(connectWs, 3000);
          }
        };
      } catch {
        setBackendStatus('offline');
      }
    };

    connectWs();
    return () => {
      closed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [apiBase]);

  // Fallback local loop when FastAPI backend is not started yet so sliders remain 100% responsive
  useEffect(() => {
    if (backendStatus === 'connected') return undefined;
    const id = setInterval(() => {
      if (!running || estopActive) return;
      const st = offlineStateRef.current;
      st.step += 1;

      const phase = st.step * 0.04;
      const autoTarget =
        st.step % 180 < 90
          ? [0.22 * Math.cos(phase * 0.35), 0.16 * Math.sin(phase * 0.45), 0.18]
          : [-0.22, 0.16, 0.18];
      const tgt = st.target_manual || autoTarget;

      let rawDx = (tgt[0] - st.pose[0]) * 3.0 * params.action_gain + st.impulse[0];
      let rawDy = (tgt[1] - st.pose[1]) * 3.0 * params.action_gain + st.impulse[1];
      let rawDz = (tgt[2] - st.pose[2]) * 2.5 * params.action_gain + st.impulse[2];
      st.impulse = [0, 0, 0];

      // Compute nearest obstacle margin + discrete CBF half-space clamp in preview
      let minMargin = Math.min(
        st.pose[0] - params.x_bounds[0],
        params.x_bounds[1] - st.pose[0],
        st.pose[1] - params.y_bounds[0],
        params.y_bounds[1] - st.pose[1],
        st.pose[2] - params.z_bounds[0]
      );
      let intervened = false;
      let clampedAxes = [];
      let safeDx = rawDx;
      let safeDy = rawDy;
      let safeDz = rawDz;

      const linNorm = Math.hypot(safeDx, safeDy, safeDz);
      if (params.cbf_enabled && linNorm > params.max_linear_step) {
        const sc = params.max_linear_step / linNorm;
        safeDx *= sc;
        safeDy *= sc;
        safeDz *= sc;
        intervened = true;
        clampedAxes.push('vel');
      }

      for (const obs of params.obstacles) {
        const dx = st.pose[0] - obs.center[0];
        const dy = st.pose[1] - obs.center[1];
        const dist = Math.hypot(dx, dy);
        const safeR = obs.radius + params.eef_radius;
        const margin = dist - safeR;
        minMargin = Math.min(minMargin, margin);
        if (params.cbf_enabled && dist > 1e-4) {
          const nx = dx / dist;
          const ny = dy / dist;
          const vDotN = safeDx * nx + safeDy * ny;
          const minDot = -params.cbf_alpha * margin * 12.0;
          if (vDotN < minDot) {
            const corr = minDot - vDotN;
            safeDx += corr * nx;
            safeDy += corr * ny;
            intervened = true;
            clampedAxes.push(obs.name);
          }
        }
      }

      const stepGain = 0.032;
      st.pose[0] += safeDx * stepGain;
      st.pose[1] += safeDy * stepGain;
      st.pose[2] = Math.max(params.cbf_enabled ? params.z_bounds[0] : 0.0, st.pose[2] + safeDz * stepGain);

      trailRef.current.push([st.pose[0], st.pose[1]]);
      if (trailRef.current.length > 70) trailRef.current.shift();

      const rawMag = Math.hypot(rawDx, rawDy);
      const safeMag = Math.hypot(safeDx, safeDy);
      marginHistoryRef.current.push({ margin: minMargin, rawMag, safeMag });
      if (marginHistoryRef.current.length > 56) marginHistoryRef.current.shift();

      setTelemetry((prev) => ({
        ...prev,
        pose: [...st.pose, 0, 0, 0],
        target: tgt,
        raw_action: [rawDx, rawDy, rawDz, 0, 0, 0, -1],
        safe_action: [safeDx, safeDy, safeDz, 0, 0, 0, -1],
        min_barrier_margin: Number(minMargin.toFixed(4)),
        cbf_intervened: intervened,
        clamped_axes: clampedAxes,
        solver_latency_ms: params.cbf_enabled ? 0.125 : 0.0,
        patch_hit_ratio: Math.min(0.99, 0.88 + params.patch_delta_threshold * 1.4),
        effective_hz: params.target_hz,
        collisions: minMargin < -0.005 ? prev.collisions + 1 : prev.collisions,
        interventions_count: intervened ? prev.interventions_count + 1 : prev.interventions_count,
        step_index: st.step,
        backend_mode: 'browser-sandbox-preview',
        obstacles: params.obstacles,
      }));
    }, 33);
    return () => clearInterval(id);
  }, [backendStatus, estopActive, params, running]);

  // Draw live top-down XY workspace viewport
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const scale = 330; // px per meter
    const cx = w / 2;
    const cy = h / 2;
    const toPx = (x, y) => [cx + x * scale, cy - y * scale];

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#07090D';
    ctx.fillRect(0, 0, w, h);

    // 10cm precision grid
    ctx.strokeStyle = 'rgba(255,255,255,0.045)';
    ctx.lineWidth = 1;
    for (let m = -0.6; m <= 0.61; m += 0.1) {
      const [gx] = toPx(m, 0);
      const [, gy] = toPx(0, m);
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(w, gy);
      ctx.stroke();
    }

    // Tunable Geofence box
    const [x0, y1] = toPx(params.x_bounds[0], params.y_bounds[1]);
    const [x1, y0] = toPx(params.x_bounds[1], params.y_bounds[0]);
    ctx.strokeStyle = params.cbf_enabled ? 'rgba(56, 189, 248, 0.45)' : 'rgba(239, 68, 68, 0.35)';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);
    ctx.setLineDash([]);

    // Trail
    if (trailRef.current.length > 1) {
      ctx.beginPath();
      trailRef.current.forEach(([tx, ty], i) => {
        const [px, py] = toPx(tx, ty);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = 'rgba(201, 255, 59, 0.35)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Obstacle Spheres + CBF Buffer Margin Ring
    const obstacles = telemetry.obstacles?.length ? telemetry.obstacles : params.obstacles;
    for (const obs of obstacles) {
      const [ox, oy] = toPx(obs.center[0], obs.center[1]);
      const rPx = obs.radius * scale;
      const rSafePx = (obs.radius + params.eef_radius) * scale;

      ctx.beginPath();
      ctx.arc(ox, oy, rSafePx, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(ox, oy, rPx, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(239, 68, 68, 0.22)';
      ctx.fill();
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 1.8;
      ctx.stroke();

      ctx.fillStyle = '#FCA5A5';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(obs.name.toUpperCase(), ox - 20, oy - rPx - 6);
    }

    // Target Crosshair
    if (telemetry.target) {
      const [tx, ty] = toPx(telemetry.target[0], telemetry.target[1]);
      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(tx, ty, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tx - 13, ty);
      ctx.lineTo(tx + 13, ty);
      ctx.moveTo(tx, ty - 13);
      ctx.lineTo(tx, ty + 13);
      ctx.stroke();
    }

    // EEF Pose + Raw vs Safe Vectors
    if (telemetry.pose) {
      const [ex, ey] = toPx(telemetry.pose[0], telemetry.pose[1]);
      const raw = telemetry.raw_action || [0, 0];
      const safe = telemetry.safe_action || [0, 0];

      // Raw nominal arrow (amber dashed)
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex + raw[0] * 140, ey - raw[1] * 140);
      ctx.strokeStyle = '#F59E0B';
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);

      // CBF-safe arrow (lime solid)
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex + safe[0] * 140, ey - safe[1] * 140);
      ctx.strokeStyle = estopActive ? '#EF4444' : '#C9FF3B';
      ctx.lineWidth = 2.6;
      ctx.stroke();

      // EEF Body
      ctx.beginPath();
      ctx.arc(ex, ey, Math.max(8, params.eef_radius * scale), 0, Math.PI * 2);
      ctx.fillStyle = estopActive
        ? 'rgba(239,68,68,0.28)'
        : telemetry.cbf_intervened
          ? 'rgba(245,158,11,0.22)'
          : 'rgba(201,255,59,0.20)';
      ctx.fill();
      ctx.strokeStyle = estopActive ? '#EF4444' : '#C9FF3B';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(ex, ey, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.fill();
    }
  }, [estopActive, params, telemetry]);

  // Unified Param Patch Helper (REST / WS + local state)
  const patchParams = async (updates) => {
    setParams((prev) => ({ ...prev, ...updates }));
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'patch_params', params: updates }));
      return;
    }
    if (backendStatus === 'connected') {
      try {
        await fetch(`${apiBase}/api/arena/params`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updates),
        });
      } catch {
        // ignore network hiccup
      }
    }
  };

  const sendControl = async (command, extra = {}) => {
    if (command === 'estop') {
      setEstopActive(true);
      setRunning(false);
    } else if (command === 'clear_estop') {
      setEstopActive(false);
    } else if (command === 'start') {
      if (!estopActive) setRunning(true);
    } else if (command === 'pause') {
      setRunning(false);
    } else if (command === 'reset') {
      trailRef.current = [];
      offlineStateRef.current = {
        pose: [-0.26, -0.18, 0.24],
        target: [0.22, 0.14, 0.14],
        vel: [0, 0, 0],
        step: 0,
        impulse: [0, 0, 0],
        target_manual: null,
      };
      setTelemetry((prev) => ({ ...prev, collisions: 0, interventions_count: 0, step_index: 0 }));
    } else if (command === 'nudge') {
      offlineStateRef.current.impulse = extra.delta || [0.65, 0.0, 0.0];
    } else if (command === 'set_target' && extra.target) {
      offlineStateRef.current.target_manual = extra.target;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'control', payload: { command, ...extra } }));
      return;
    }
    if (backendStatus === 'connected') {
      await fetch(`${apiBase}/api/arena/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, ...extra }),
      });
    }
  };

  const handleCanvasClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * e.currentTarget.width;
    const py = ((e.clientY - rect.top) / rect.height) * e.currentTarget.height;
    const x = Number(((px - e.currentTarget.width / 2) / 330).toFixed(3));
    const y = Number(((e.currentTarget.height / 2 - py) / 330).toFixed(3));
    sendControl('set_target', { target: [x, y, telemetry.pose?.[2] ?? 0.18] });
  };

  const handleConnectRobot = async (e) => {
    e.preventDefault();
    const payload = {
      robot_id: newRobotId.trim() || 'robot-01',
      name: newRobotName.trim() || newRobotId,
      adapter: newRobotAdapter,
      endpoint: newRobotEndpoint.trim(),
      dof: Number(newRobotDof) || 7,
      hardware_preset: newRobotAdapter,
    };
    if (backendStatus === 'connected') {
      const res = await fetch(`${apiBase}/api/arena/robots/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        setShowConnectForm(false);
        return;
      }
    }
    setRobots((prev) => [
      ...prev.filter((r) => r.robot_id !== payload.robot_id),
      { ...payload, connected: true, live_ws: false, ticks_served: 0 },
    ]);
    setActiveRobotId(payload.robot_id);
    setShowConnectForm(false);
  };

  const handleSelectRobot = async (robotId) => {
    setActiveRobotId(robotId);
    if (backendStatus === 'connected') {
      await fetch(`${apiBase}/api/arena/robots/${robotId}/select`, { method: 'POST' });
    }
  };

  const handleAddObstacle = async (e) => {
    e.preventDefault();
    const obs = {
      name: obsName.trim() || `zone_${params.obstacles.length + 1}`,
      center: [parseFloat(obsX) || 0, parseFloat(obsY) || 0, 0.22],
      radius: Math.max(0.02, Math.min(0.35, parseFloat(obsRadius) || 0.07)),
    };
    const nextList = [...params.obstacles.filter((o) => o.name !== obs.name), obs];
    await patchParams({ obstacles: nextList });
  };

  const handleRemoveObstacle = async (name) => {
    const nextList = params.obstacles.filter((o) => o.name !== name);
    await patchParams({ obstacles: nextList });
  };

  const handleExportProfile = async () => {
    if (backendStatus === 'connected') {
      const res = await fetch(`${apiBase}/api/arena/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episode_label: activeRobotId }),
      });
      if (res.ok) {
        const data = await res.json();
        setExportBanner({
          mcapBytes: data.mcap_bytes,
          path: data.profile_path,
          profile: data.deploy_profile,
        });
        const blob = new Blob([JSON.stringify(data.deploy_profile, null, 2)], {
          type: 'application/json',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `swizzle_deploy_profile_${activeRobotId}.json`;
        a.click();
        URL.revokeObjectURL(url);
        return;
      }
    }
    const localProfile = {
      schema: 'swizzle.robotics.deploy_profile.v1',
      active_robot_id: activeRobotId,
      tuned_params: params,
    };
    const blob = new Blob([JSON.stringify(localProfile, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `swizzle_deploy_profile_${activeRobotId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportBanner({ mcapBytes: 0, path: 'Downloaded swizzle_deploy_profile.json' });
  };

  const handleCopyBridgeCmd = () => {
    navigator.clipboard?.writeText(
      `python examples/connect_real_robot.py --robot-id ${activeRobotId} --url ws://127.0.0.1:8787`
    );
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 1800);
  };

  // Oscilloscope SVG path
  const marginPath = useMemo(() => {
    const pts = marginHistoryRef.current;
    if (pts.length < 2) return '';
    return pts
      .map((p, idx) => {
        const x = (idx / (pts.length - 1)) * 280;
        const clamped = Math.max(-0.08, Math.min(0.2, p.margin));
        const y = 52 - ((clamped + 0.08) / 0.28) * 44;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(' ');
  }, [telemetry.step_index]);

  return (
    <section id="studio" className="max-w-[1360px] mx-auto px-5 sm:px-8 py-14 sm:py-18 scroll-mt-20">
      {/* Header Bar */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 mb-6">
        <div>
          <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-limebrand mb-2">
            <Radio className="w-3.5 h-3.5" />
            <span>REAL-LIFE HARDWARE ARENA & PARAMETER TUNER</span>
          </div>
          <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Connect Your Robot. Tune Parameters Live Before Deployment.
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 mt-1.5 max-w-2xl">
            Stream proprioception over <code className="text-slate-200">/ws/robot/&#123;id&#125;</code> or{' '}
            Eclipse Zenoh, hot-tune CBF barrier gain &amp; RTC chunk smoothness, test disturbances safely,
            and export binary <code className="text-slate-200">.mcap</code> + deployment JSON.
          </p>
        </div>

        {/* Live Backend Status & URL */}
        <div className="flex flex-wrap items-center gap-2.5">
          <div
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border font-mono text-[11px] ${
              backendStatus === 'connected'
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'border-amber-500/35 bg-amber-500/10 text-amber-300'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus === 'connected' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span>
              {backendStatus === 'connected'
                ? 'LIVE PYTHON ARENA CONNECTED'
                : 'BROWSER TWIN (START SERVER FOR REAL ROBOTS)'}
            </span>
          </div>
          <input
            type="text"
            value={backendBaseUrl}
            onChange={(e) => setBackendBaseUrl(e.target.value)}
            placeholder="http://127.0.0.1:8787 (auto)"
            aria-label="Arena backend URL"
            className="w-48 px-2.5 py-1.5 rounded-lg bg-panel border border-white/[0.12] font-mono text-[11px] text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-limebrand"
          />
        </div>
      </div>

      {/* Main 12-Col Studio Cockpit */}
      <div className="rounded-2xl border border-white/[0.12] bg-panel/95 overflow-hidden shadow-2xl">
        {/* Top Robot Fleet Bar */}
        <div className="px-4 sm:px-5 py-3 border-b border-white/[0.085] bg-surface/90 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {robots.map((bot) => {
              const selected = bot.robot_id === activeRobotId;
              return (
                <button
                  key={bot.robot_id}
                  type="button"
                  onClick={() => handleSelectRobot(bot.robot_id)}
                  className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border font-mono text-[11px] transition-all ${
                    selected
                      ? 'border-limebrand/60 bg-limebrand/10 text-white'
                      : 'border-white/[0.08] bg-obsidian/60 text-slate-400 hover:text-white'
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      bot.live_ws || bot.connected ? 'bg-emerald-400' : 'bg-slate-500'
                    }`}
                  />
                  <span>{bot.name}</span>
                  <span className="text-[10px] text-slate-500 uppercase">{bot.adapter}</span>
                </button>
              );
            })}
            <button
              type="button"
              onClick={() => setShowConnectForm((v) => !v)}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-cyanbrand/40 bg-cyanbrand/10 hover:bg-cyanbrand/20 text-cyanbrand font-mono text-[11px]"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Connect Real Robot</span>
            </button>
          </div>

          {/* Hard E-STOP & Export */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleExportProfile}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-white/[0.12] bg-white/[0.04] hover:bg-white/[0.09] font-mono text-[11px] text-slate-200"
            >
              <Download className="w-3.5 h-3.5 text-cyanbrand" />
              <span>Export Profile + .MCAP</span>
            </button>
            <button
              type="button"
              onClick={() => sendControl(estopActive ? 'clear_estop' : 'estop')}
              className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-mono text-[11px] font-bold tracking-wider transition-all ${
                estopActive
                  ? 'bg-amber-400 text-obsidian hover:bg-amber-300'
                  : 'bg-red-600 hover:bg-red-500 text-white shadow-[0_0_20px_rgba(239,68,68,0.4)]'
              }`}
            >
              <AlertOctagon className="w-3.5 h-3.5" />
              <span>{estopActive ? 'CLEAR E-STOP' : 'HARD E-STOP'}</span>
            </button>
          </div>
        </div>

        {/* Connect Real Robot Modal/Drawer */}
        {showConnectForm && (
          <form
            onSubmit={handleConnectRobot}
            className="p-4 sm:p-5 border-b border-white/[0.085] bg-obsidian/90 grid grid-cols-1 md:grid-cols-12 gap-3 items-end"
          >
            <div className="md:col-span-12 flex flex-wrap items-center gap-2 pb-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                Hardware Presets:
              </span>
              {ADAPTER_PRESETS.map((preset) => (
                <button
                  key={preset.preset_id}
                  type="button"
                  onClick={() => {
                    setNewRobotId(preset.preset_id + '-cell-01');
                    setNewRobotName(preset.name);
                    setNewRobotAdapter(preset.adapter);
                    setNewRobotEndpoint(preset.default_endpoint);
                    setNewRobotDof(preset.dof);
                    patchParams({
                      cbf_alpha: preset.recommended_cbf_alpha,
                      max_linear_step: preset.recommended_max_linear_step,
                    });
                  }}
                  className="px-2.5 py-1 rounded bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.08] font-mono text-[10px] text-slate-200"
                >
                  {preset.name}
                </button>
              ))}
            </div>

            <div className="md:col-span-2">
              <label className="block font-mono text-[10px] text-slate-400 mb-1">Robot ID</label>
              <input
                value={newRobotId}
                onChange={(e) => setNewRobotId(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded bg-panel border border-white/[0.12] font-mono text-xs text-white"
              />
            </div>
            <div className="md:col-span-3">
              <label className="block font-mono text-[10px] text-slate-400 mb-1">Display Name</label>
              <input
                value={newRobotName}
                onChange={(e) => setNewRobotName(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded bg-panel border border-white/[0.12] font-mono text-xs text-white"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block font-mono text-[10px] text-slate-400 mb-1">Adapter</label>
              <select
                value={newRobotAdapter}
                onChange={(e) => setNewRobotAdapter(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded bg-panel border border-white/[0.12] font-mono text-xs text-white"
              >
                <option value="websocket_sdk">websocket_sdk</option>
                <option value="franka_fci">franka_fci</option>
                <option value="ur_rtde">ur_rtde</option>
                <option value="serial_so100">serial_so100</option>
                <option value="ros2_zenoh">ros2_zenoh</option>
              </select>
            </div>
            <div className="md:col-span-3">
              <label className="block font-mono text-[10px] text-slate-400 mb-1">Endpoint / Port</label>
              <input
                value={newRobotEndpoint}
                onChange={(e) => setNewRobotEndpoint(e.target.value)}
                className="w-full px-2.5 py-1.5 rounded bg-panel border border-white/[0.12] font-mono text-xs text-white"
              />
            </div>
            <div className="md:col-span-2 flex gap-2">
              <button
                type="submit"
                className="flex-1 px-3 py-1.5 rounded bg-limebrand text-obsidian font-mono text-xs font-semibold hover:bg-[#d8ff66]"
              >
                Register Cell
              </button>
            </div>
          </form>
        )}

        {/* Export feedback strip */}
        {exportBanner && (
          <div className="px-5 py-2 border-b border-emerald-500/30 bg-emerald-500/10 flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-emerald-300">
            <span>
              Exported deployment profile ({exportBanner.mcapBytes.toLocaleString()} bytes .mcap + LeRobot v2.1)
            </span>
            <button
              type="button"
              onClick={() => setExportBanner(null)}
              className="text-slate-300 hover:text-white"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12">
          {/* Left 7 Columns: Live Workspace + Oscilloscope + HIL Test Controls */}
          <div className="lg:col-span-7 p-4 sm:p-5 border-b lg:border-b-0 lg:border-r border-white/[0.085] flex flex-col gap-4">
            <div className="relative rounded-xl border border-white/[0.085] overflow-hidden bg-obsidian">
              <canvas
                ref={canvasRef}
                width={660}
                height={360}
                onClick={handleCanvasClick}
                className="w-full h-auto block cursor-crosshair"
                title="Click anywhere in workspace to command manual Cartesian setpoint"
              />
              <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap items-center gap-2">
                <span className="px-2 py-0.5 rounded bg-obsidian/85 border border-white/10 font-mono text-[10px] text-slate-300">
                  CLICK WORKSPACE TO SET TARGET
                </span>
                <span className="px-2 py-0.5 rounded bg-obsidian/85 border border-white/10 font-mono text-[10px] text-cyanbrand">
                  POSE [{telemetry.pose?.slice(0, 3).map((v) => Number(v).toFixed(2)).join(', ')}]
                </span>
              </div>
              <div className="pointer-events-none absolute bottom-3 left-3 flex items-center gap-4 font-mono text-[10px]">
                <span className="inline-flex items-center gap-1.5 text-amber-300">
                  <span className="w-3 h-0.5 bg-amber-400 inline-block" /> Nominal VLA
                </span>
                <span className="inline-flex items-center gap-1.5 text-limebrand">
                  <span className="w-3 h-0.5 bg-limebrand inline-block" /> CBF-Safe Command
                </span>
              </div>
              {estopActive && (
                <div className="absolute inset-0 bg-red-950/55 backdrop-blur-[2px] flex flex-col items-center justify-center p-4 text-center">
                  <AlertOctagon className="w-10 h-10 text-red-400 mb-2" />
                  <div className="font-display text-xl font-bold text-white tracking-wide">
                    HARDWARE E-STOP LATCHED
                  </div>
                  <div className="font-mono text-xs text-red-200 mt-1">
                    All 7-DoF motor commands zeroed ([0, 0, 0, 0, 0, 0, 0]). Click CLEAR E-STOP to unlock.
                  </div>
                </div>
              )}
            </div>

            {/* HIL Controls + Live Bridge Snippet */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => sendControl(running ? 'pause' : 'start')}
                  disabled={estopActive}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.11] disabled:opacity-40 border border-white/[0.1] font-mono text-xs text-white"
                >
                  {running ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  <span>{running ? 'Pause Loop' : 'Resume Loop'}</span>
                </button>
                <button
                  type="button"
                  onClick={() => sendControl('reset')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.09] border border-white/[0.08] font-mono text-xs text-slate-300"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Reset</span>
                </button>
                <button
                  type="button"
                  onClick={() => sendControl('nudge', { delta: [0.85, 0.25, 0.0] })}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/35 font-mono text-xs text-amber-300"
                >
                  <Zap className="w-3.5 h-3.5" />
                  <span>Inject Disturbance</span>
                </button>
              </div>

              <button
                type="button"
                onClick={handleCopyBridgeCmd}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-obsidian border border-white/[0.1] font-mono text-[11px] text-slate-300 hover:text-white"
              >
                {copiedCmd ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-slate-400" />}
                <span className="truncate max-w-[230px]">connect_real_robot.py --robot-id {activeRobotId}</span>
              </button>
            </div>

            {/* 4-Metric Telemetry Strip + SVG Margin Oscilloscope */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <div className="rounded-lg border border-white/[0.08] bg-obsidian/70 p-2.5">
                <div className="font-mono text-[10px] text-slate-500 uppercase">Barrier Margin h(x)</div>
                <div
                  className={`font-mono text-sm font-bold mt-0.5 ${
                    (telemetry.min_barrier_margin ?? 0) < 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}
                >
                  {(telemetry.min_barrier_margin ?? 0) >= 0 ? '+' : ''}
                  {Number(telemetry.min_barrier_margin ?? 0).toFixed(3)} m
                </div>
              </div>
              <div className="rounded-lg border border-white/[0.08] bg-obsidian/70 p-2.5">
                <div className="font-mono text-[10px] text-slate-500 uppercase">CBF Solve / Shield</div>
                <div className="font-mono text-sm font-bold text-white mt-0.5">
                  {Number(telemetry.solver_latency_ms ?? 0).toFixed(3)} ms{' '}
                  <span className={telemetry.cbf_intervened ? 'text-amber-300 text-[11px]' : 'text-emerald-400 text-[11px]'}>
                    {telemetry.cbf_intervened ? 'CLAMPED' : 'PASS'}
                  </span>
                </div>
              </div>
              <div className="rounded-lg border border-white/[0.08] bg-obsidian/70 p-2.5">
                <div className="font-mono text-[10px] text-slate-500 uppercase">PatchCache Reuse</div>
                <div className="font-mono text-sm font-bold text-cyanbrand mt-0.5">
                  {(Number(telemetry.patch_hit_ratio ?? 0) * 100).toFixed(1)}%
                </div>
              </div>
              <div className="rounded-lg border border-white/[0.08] bg-obsidian/70 p-2.5">
                <div className="font-mono text-[10px] text-slate-500 uppercase">Collisions / Steps</div>
                <div className="font-mono text-sm font-bold text-white mt-0.5">
                  <span className={telemetry.collisions > 0 ? 'text-red-400' : 'text-emerald-400'}>
                    {telemetry.collisions}
                  </span>{' '}
                  / {telemetry.step_index}
                </div>
              </div>
            </div>

            {/* Real-Time Oscilloscope */}
            <div className="rounded-lg border border-white/[0.08] bg-obsidian/80 p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-400">
                  <Activity className="w-3.5 h-3.5 text-limebrand" />
                  Live CBF Safety Margin Oscilloscope (0.0m Breach Line)
                </span>
                <span className="font-mono text-[10px] text-slate-500">
                  {params.policy_model} · {telemetry.effective_hz} Hz
                </span>
              </div>
              <svg viewBox="0 0 280 56" className="w-full h-14 overflow-visible">
                {/* Zero breach line (y = 52 - (0.08/0.28)*44 = 39.4) */}
                <line x1="0" y1="39.4" x2="280" y2="39.4" stroke="rgba(239,68,68,0.45)" strokeDasharray="3 3" strokeWidth="1" />
                {marginPath && <path d={marginPath} fill="none" stroke="#C9FF3B" strokeWidth="1.8" />}
              </svg>
            </div>
          </div>

          {/* Right 5 Columns: Parameter Fine-Tuning Console */}
          <div className="lg:col-span-5 p-4 sm:p-5 flex flex-col gap-3">
            {/* Console Tabs */}
            <div className="flex items-center justify-between gap-2 pb-2 border-b border-white/[0.08]">
              <div className="flex items-center gap-1.5 font-mono text-xs text-white font-semibold">
                <Sliders className="w-3.5 h-3.5 text-limebrand" />
                <span>LIVE PARAMETER TUNER</span>
              </div>
              <div className="flex items-center gap-1 bg-obsidian p-1 rounded-lg border border-white/[0.08]">
                {[
                  { id: 'safety', label: 'CBF Safety' },
                  { id: 'rtc', label: 'RTC & Vision' },
                  { id: 'obstacles', label: 'Zones & Log' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setTuneTab(tab.id)}
                    className={`px-2.5 py-1 rounded font-mono text-[10px] transition-colors ${
                      tuneTab === tab.id ? 'bg-limebrand text-obsidian font-semibold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {tuneTab === 'safety' && (
              <div className="space-y-2.5">
                <button
                  type="button"
                  onClick={() => patchParams({ cbf_enabled: !params.cbf_enabled })}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg border font-mono text-xs transition-colors ${
                    params.cbf_enabled
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                      : 'border-red-500/40 bg-red-500/10 text-red-300'
                  }`}
                >
                  <span className="inline-flex items-center gap-2">
                    {params.cbf_enabled ? <ShieldCheck className="w-4 h-4" /> : <ShieldAlert className="w-4 h-4" />}
                    <span>SWIZZLE-CBF HARDWARE SHIELD</span>
                  </span>
                  <span className="font-bold">{params.cbf_enabled ? 'ARMED' : 'BYPASSED'}</span>
                </button>

                <SliderField
                  label="cbf_alpha (Barrier Class-K Gain)"
                  value={params.cbf_alpha}
                  min={0.05}
                  max={1.0}
                  step={0.05}
                  hint="Lower = gentler slowdown near obstacles; Higher = tighter workspace agility."
                  onChange={(val) => patchParams({ cbf_alpha: val })}
                />
                <SliderField
                  label="max_linear_step (Velocity Cap)"
                  value={params.max_linear_step}
                  min={0.04}
                  max={0.45}
                  step={0.01}
                  unit=" m/s"
                  hint="Hard Cartesian velocity limit enforced on raw VLA proposals."
                  onChange={(val) => patchParams({ max_linear_step: val })}
                />
                <SliderField
                  label="max_accel_step (Jerk Slew Limit)"
                  value={params.max_accel_step}
                  min={0.02}
                  max={0.4}
                  step={0.01}
                  unit=" m/s²"
                  hint="Prevents chunk-transition mechanical spikes on gears/actuators."
                  onChange={(val) => patchParams({ max_accel_step: val })}
                />
                <div className="grid grid-cols-2 gap-2.5">
                  <SliderField
                    label="eef_radius"
                    value={params.eef_radius}
                    min={0.02}
                    max={0.14}
                    step={0.005}
                    unit=" m"
                    onChange={(val) => patchParams({ eef_radius: val })}
                  />
                  <SliderField
                    label="z_min Table Floor"
                    value={params.z_bounds[0]}
                    min={0.0}
                    max={0.25}
                    step={0.01}
                    unit=" m"
                    onChange={(val) => patchParams({ z_bounds: [val, params.z_bounds[1]] })}
                  />
                </div>
                <SliderField
                  label="watchdog_timeout_steps"
                  value={params.watchdog_timeout_steps}
                  min={1}
                  max={15}
                  step={1}
                  unit=" ticks"
                  hint="Automatic deadman velocity decay window if telemetry drops."
                  onChange={(val) => patchParams({ watchdog_timeout_steps: Math.round(val) })}
                />
              </div>
            )}

            {tuneTab === 'rtc' && (
              <div className="space-y-2.5">
                <div className="rounded-lg border border-white/[0.07] bg-obsidian/70 p-2.5">
                  <label className="block font-mono text-[11px] text-slate-300 mb-1.5">Active Policy Model</label>
                  <select
                    value={params.policy_model}
                    onChange={(e) => patchParams({ policy_model: e.target.value })}
                    className="w-full px-2.5 py-1.5 rounded bg-panel border border-white/[0.12] font-mono text-xs text-limebrand"
                  >
                    {POLICY_OPTIONS.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>

                <SliderField
                  label="target_hz (Control Frequency)"
                  value={params.target_hz}
                  min={15}
                  max={120}
                  step={5}
                  unit=" Hz"
                  onChange={(val) => patchParams({ target_hz: val })}
                />
                <SliderField
                  label="decay_lambda (RTC Chunk Blend)"
                  value={params.decay_lambda}
                  min={0.0}
                  max={0.4}
                  step={0.01}
                  hint="Temporal action ensembler decay weight across overlapping flow chunks."
                  onChange={(val) => patchParams({ decay_lambda: val })}
                />
                <div className="grid grid-cols-2 gap-2.5">
                  <SliderField
                    label="rtc_substeps"
                    value={params.rtc_substeps}
                    min={1}
                    max={8}
                    step={1}
                    onChange={(val) => patchParams({ rtc_substeps: Math.round(val) })}
                  />
                  <SliderField
                    label="action_gain"
                    value={params.action_gain}
                    min={0.2}
                    max={1.8}
                    step={0.05}
                    unit="x"
                    onChange={(val) => patchParams({ action_gain: val })}
                  />
                </div>
                <SliderField
                  label="patch_delta_threshold (ViT PatchCache)"
                  value={params.patch_delta_threshold}
                  min={0.005}
                  max={0.1}
                  step={0.005}
                  hint="Higher skips more static camera patches for faster Jetson/laptop latency."
                  onChange={(val) => patchParams({ patch_delta_threshold: val })}
                />
                <div className="rounded-lg border border-white/[0.07] bg-obsidian/70 p-2.5">
                  <label className="block font-mono text-[11px] text-slate-300 mb-1">Task Instruction (S2 Prompt)</label>
                  <input
                    value={params.instruction}
                    onChange={(e) => patchParams({ instruction: e.target.value })}
                    className="w-full px-2.5 py-1.5 rounded bg-panel border border-white/[0.12] font-mono text-[11px] text-white"
                  />
                </div>
              </div>
            )}

            {tuneTab === 'obstacles' && (
              <div className="space-y-3">
                <form onSubmit={handleAddObstacle} className="rounded-lg border border-white/[0.08] bg-obsidian/70 p-2.5 space-y-2">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-cyanbrand">
                    Add Keep-Out Sphere (Human / Fixture)
                  </div>
                  <div className="grid grid-cols-4 gap-1.5">
                    <input
                      value={obsName}
                      onChange={(e) => setObsName(e.target.value)}
                      placeholder="name"
                      className="px-2 py-1 rounded bg-panel border border-white/10 font-mono text-[11px] text-white"
                    />
                    <input
                      value={obsX}
                      onChange={(e) => setObsX(e.target.value)}
                      placeholder="x"
                      className="px-2 py-1 rounded bg-panel border border-white/10 font-mono text-[11px] text-white"
                    />
                    <input
                      value={obsY}
                      onChange={(e) => setObsY(e.target.value)}
                      placeholder="y"
                      className="px-2 py-1 rounded bg-panel border border-white/10 font-mono text-[11px] text-white"
                    />
                    <input
                      value={obsRadius}
                      onChange={(e) => setObsRadius(e.target.value)}
                      placeholder="r (m)"
                      className="px-2 py-1 rounded bg-panel border border-white/10 font-mono text-[11px] text-white"
                    />
                  </div>
                  <button
                    type="submit"
                    className="w-full py-1.5 rounded bg-cyanbrand/15 hover:bg-cyanbrand/25 border border-cyanbrand/35 font-mono text-[11px] text-cyanbrand"
                  >
                    + Inject Keep-Out Zone Live
                  </button>
                </form>

                <div className="space-y-1.5 max-h-36 overflow-y-auto">
                  {params.obstacles.map((o) => (
                    <div
                      key={o.name}
                      className="flex items-center justify-between px-2.5 py-1.5 rounded bg-obsidian/80 border border-white/[0.07] font-mono text-[11px]"
                    >
                      <span className="text-slate-200">
                        {o.name}{' '}
                        <span className="text-slate-500">
                          ({o.center[0]}, {o.center[1]}) r={o.radius}m
                        </span>
                      </span>
                      <button
                        type="button"
                        onClick={() => handleRemoveObstacle(o.name)}
                        aria-label={`Remove obstacle ${o.name}`}
                        className="text-red-400 hover:text-red-300"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>

                <div className="rounded-lg border border-white/[0.08] bg-obsidian/80 p-2.5">
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400 mb-1.5">
                    Live Audit Stream
                  </div>
                  <div className="space-y-1 max-h-32 overflow-y-auto font-mono text-[10px]">
                    {events.slice(0, 8).map((ev, i) => (
                      <div key={i} className="text-slate-300 leading-snug">
                        <span className="text-limebrand">[{ev.kind}]</span> {ev.message}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
