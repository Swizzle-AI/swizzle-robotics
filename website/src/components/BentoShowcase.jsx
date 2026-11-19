import React, { useState } from 'react';
import {
  Shield,
  Activity,
  FileCode2,
  HardDrive,
  Check,
  Copy,
  Radio,
  Cpu,
} from 'lucide-react';

const CODE_TABS = [
  {
    id: 'python',
    label: 'Python S2->S1->S0',
    code: `from swizzle_robotics import HierarchicalController, SafetyConfig, SwizzleManipulationArena, get_policy

arena = SwizzleManipulationArena(prefer_mujoco=True)
obs = arena.reset(seed=42)

ctrl = HierarchicalController(
    policy=get_policy("swizzle/edge-flow-vla"),
    safety_config=SafetyConfig(z_bounds=(0.02, 0.70), max_accel_step=0.14),
)
# S2 queries swizzle-rt JSON Schema & injects keep-out barriers into CBF
plan = ctrl.plan_s2(
    "pick up cube while avoiding pillar then place on tray",
    known_objects={"pillar": (0.12, -0.02, 0.09, 0.045)},
)

tick = ctrl.step_s1_s0(obs.frame, trace_id="00-4bf92f3577b34da6...")
assert tick.receipt.verify()  # Cryptographic audit pass`,
  },
  {
    id: 'cli',
    label: 'Arena CLI + MCAP',
    code: `# Install Swizzle Robotics + MuJoCo arena extras
pip install "swizzle-robotics[sim]"

# Detect GPU silicon & quantize edge VLA tier
swizzle-robotics detect

# Run closed-loop manipulation arena & export binary .mcap + LeRobot v2.1
swizzle-robotics arena --episodes 3 --steps 40 --record-dir ./recordings

# Verified output:
# Episode 0: steps=30, collisions=0, rate=357.4 Hz, mcap_bytes=27612`,
  },
  {
    id: 'zenoh',
    label: 'Eclipse Zenoh Bridge',
    code: `from swizzle_robotics import ZenohBridgeConfig, ZenohRoboticsBridge, ControlBarrierShield

bridge = ZenohRoboticsBridge(
    ZenohBridgeConfig(locator="tcp/192.168.1.40:7447", queue_capacity=4),
    use_live_zenoh=True,
)
shield = ControlBarrierShield()

# Drop-oldest bounded QoS avoids DDS multicast storms over robot Wi-Fi
sec = shield.project(raw_action, current_pose, heartbeat_ok=bridge.heartbeat_healthy())
bridge.publish("rt/cmd_vel", {"action": sec.safe_action}, traceparent="00-...")`,
  },
];

const MCAP_PACKETS = [
  { ch: '/swizzle/s2_plan', payload: '{"subtasks":["pick red cube","place tray"],"avoid_zones":["pillar"]}' },
  { ch: '/action/raw_vla', payload: '{"dx":0.214,"dy":-0.088,"dz":-0.150,"gripper":1.0}' },
  { ch: '/cbf/safety', payload: '{"intervened":true,"margin":0.011,"reason":"cbf_obstacle_pillar","ms":0.125}' },
  { ch: '/action/cbf_safe', payload: '{"dx":0.118,"dy":0.064,"dz":-0.045,"gripper":1.0}' },
  { ch: '/swizzle/receipt', payload: '{"digest":"sha256:8e1b...c40f","dagger_flagged":true,"verified":true}' },
];

export default function BentoShowcase() {
  const [activeTab, setActiveTab] = useState('python');
  const [copied, setCopied] = useState(false);
  const [verifiedTick, setVerifiedTick] = useState(true);

  const activeSnippet = CODE_TABS.find((t) => t.id === activeTab)?.code || '';

  const copyCode = () => {
    navigator.clipboard.writeText(activeSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <section id="architecture" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
      <div className="mb-12 max-w-3xl">
        <div className="font-mono text-xs uppercase tracking-[0.18em] text-[#C9FF3B] mb-3">
          CORE RUNTIME ARCHITECTURE // S2 -&gt; S1 -&gt; S0
        </div>
        <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-white">
          Engineered for the Missing Layer in Robotics AI.
        </h2>
        <p className="mt-4 text-slate-400 text-base sm:text-lg leading-relaxed">
          Frontier foundation labs train larger vision-language-action policies, yet robots still fail on chunk boundary
          jerk, workspace crashes, and fragmented telemetry. Swizzle ships deterministic runtime primitives between your
          weights and motors.
        </p>
      </div>

      {/* Asymmetric Enterprise Bento Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Card 1: Swizzle-CBF (Large 7 cols) */}
        <div className="lg:col-span-7 rounded-2xl border hairline bg-[#0B0D11] p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-4 mb-5">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#C9FF3B]/10 border border-[#C9FF3B]/30 font-mono text-[11px] text-[#C9FF3B]">
                <Shield className="w-3.5 h-3.5" />
                PILLAR 01 · SWIZZLE-CBF SAFETY SHIELD
              </div>
              <span className="font-mono text-xs text-slate-500">0.125 ms / tick</span>
            </div>
            <h3 className="font-display text-2xl font-bold text-white">
              Mathematical Safety Invariance Before Every Actuator Pulse.
            </h3>
            <p className="mt-2.5 text-sm text-slate-400 leading-relaxed">
              Every raw VLA command passes through a discrete Control Barrier Function projector before touching joints:
              workspace geofencing, spherical keep-out zones, Euclidean velocity &amp; slew-rate clamping, and smooth
              deadman velocity decay when network tunnels stall.
            </p>
          </div>

          <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="rounded-xl border hairline bg-[#11151C] p-4">
              <div className="font-mono text-[10px] text-slate-500 uppercase">Adversarial Stress</div>
              <div className="font-display text-2xl font-bold text-[#C9FF3B] mt-1">0 / 25</div>
              <div className="font-mono text-[11px] text-slate-400 mt-1">vs 25/25 unshielded collisions</div>
            </div>
            <div className="rounded-xl border hairline bg-[#11151C] p-4">
              <div className="font-mono text-[10px] text-slate-500 uppercase">Median Solve</div>
              <div className="font-display text-2xl font-bold text-sky-300 mt-1">0.125 ms</div>
              <div className="font-mono text-[11px] text-slate-400 mt-1">Pure NumPy active half-spaces</div>
            </div>
            <div className="rounded-xl border hairline bg-[#11151C] p-4">
              <div className="font-mono text-[10px] text-slate-500 uppercase">Network Deadman</div>
              <div className="font-display text-2xl font-bold text-emerald-400 mt-1">Soft Halt</div>
              <div className="font-mono text-[11px] text-slate-400 mt-1">Auto-decays velocity over N ticks</div>
            </div>
          </div>
        </div>

        {/* Card 2: Swizzle-RTC Waveform (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl border hairline bg-[#0B0D11] p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-5">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-sky-500/10 border border-sky-500/30 font-mono text-[11px] text-sky-300">
                <Activity className="w-3.5 h-3.5" />
                PILLAR 02 · SWIZZLE-RTC
              </div>
              <span className="font-mono text-xs text-slate-500">1.55x smoother</span>
            </div>
            <h3 className="font-display text-xl font-bold text-white">
              Real-Time Flow-Matching Prefix Inpainting.
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Freezes the committed delay prefix (<code className="font-mono text-slate-200">w = 1.0</code>) while GPU
              inference runs, then smoothly blends overlapping chunks via an exponential schedule paired with C2 quintic
              Hermite splines.
            </p>
          </div>

          {/* Crisp SVG Comparison Plot */}
          <div className="mt-6 rounded-xl border hairline bg-[#07090D] p-4">
            <div className="flex items-center justify-between font-mono text-[10px] text-slate-400 mb-2">
              <span className="text-rose-400">--- Naive Chunk Switch (0.800 spike)</span>
              <span className="text-[#C9FF3B]">— Swizzle-RTC (0.515 smooth)</span>
            </div>
            <svg viewBox="0 0 320 96" className="w-full h-24">
              <line x1="0" y1="48" x2="320" y2="48" stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
              <rect x="0" y="6" width="76" height="84" fill="rgba(56,189,248,0.07)" />
              <text x="8" y="18" fill="#38BDF8" fontSize="8" fontFamily="JetBrains Mono">FROZEN PREFIX</text>
              {/* Jagged naive step */}
              <polyline
                fill="none"
                stroke="#FB7185"
                strokeWidth="1.75"
                strokeDasharray="4 3"
                points="10,24 85,24 105,82 220,82 310,74"
              />
              {/* Smooth RTC spline */}
              <path
                d="M 10 24 L 80 24 C 125 24, 150 78, 220 80 L 310 74"
                fill="none"
                stroke="#C9FF3B"
                strokeWidth="2.5"
              />
            </svg>
          </div>
        </div>

        {/* Card 3: Code & SDK First (5 cols) */}
        <div className="lg:col-span-5 rounded-2xl border hairline bg-[#0B0D11] p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-4">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-white/5 border hairline font-mono text-[11px] text-slate-200">
                <FileCode2 className="w-3.5 h-3.5 text-[#C9FF3B]" />
                DEVELOPER SDK &amp; ZENOH
              </div>
              <button
                onClick={copyCode}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-white/5 hover:bg-white/10 font-mono text-[11px] text-slate-300 border hairline transition"
              >
                {copied ? <Check className="w-3 h-3 text-[#C9FF3B]" /> : <Copy className="w-3 h-3" />}
                {copied ? 'COPIED' : 'COPY'}
              </button>
            </div>

            <div className="flex flex-wrap gap-1.5 mb-3">
              {CODE_TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`px-2.5 py-1 rounded font-mono text-[11px] transition ${
                    activeTab === t.id
                      ? 'bg-[#C9FF3B] text-[#050608] font-semibold'
                      : 'bg-white/5 text-slate-400 hover:text-white'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <pre className="rounded-xl border hairline bg-[#07090D] p-4 font-mono text-[11.5px] leading-relaxed text-slate-300 overflow-x-auto">
            <code>{activeSnippet}</code>
          </pre>
        </div>

        {/* Card 4: Foxglove MCAP + LeRobot v2.1 + Receipts (7 cols) */}
        <div className="lg:col-span-7 rounded-2xl border hairline bg-[#0B0D11] p-6 sm:p-7 flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-5">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 font-mono text-[11px] text-emerald-300">
                <HardDrive className="w-3.5 h-3.5" />
                PILLAR 03 · EMBODIED BLACK-BOX FLIGHT RECORDER
              </div>
              <div className="flex items-center gap-2 font-mono text-[11px]">
                <span className="px-2 py-0.5 rounded bg-white/5 text-slate-300 border hairline">\x89MCAP0</span>
                <span className="px-2 py-0.5 rounded bg-white/5 text-slate-300 border hairline">LeRobot v2.1</span>
              </div>
            </div>
            <h3 className="font-display text-2xl font-bold text-white">
              Zero-Dependency Binary MCAP, Auto-DAgger &amp; Cryptographic Action Receipts.
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Streams Foxglove-compatible binary <code className="font-mono text-slate-200">.mcap</code> logs alongside
              Hugging Face <code className="font-mono text-slate-200">LeRobot v2.1</code> datasets, automatically flags
              episodes where <code className="font-mono text-slate-200">Swizzle-CBF</code> intervened for DAgger
              retraining, and seals every command with a tamper-evident hash.
            </p>
          </div>

          <div className="mt-5 rounded-xl border hairline bg-[#07090D] divide-y divide-white/5 overflow-hidden">
            {MCAP_PACKETS.map((pkt) => (
              <div key={pkt.ch} className="px-3.5 py-2 flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-4 font-mono text-[11px]">
                <span className="text-sky-300 shrink-0 w-40">{pkt.ch}</span>
                <span className="text-slate-400 truncate">{pkt.payload}</span>
              </div>
            ))}
            <div className="px-3.5 py-2.5 bg-[#0D1219] flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
              <span className="text-slate-400">
                Receipt Chain: <span className="text-emerald-400">observation -&gt; model_id -&gt; cbf_safe_action</span>
              </span>
              <button
                onClick={() => setVerifiedTick((v) => !v)}
                className={`px-2.5 py-1 rounded border text-[11px] font-semibold transition ${
                  verifiedTick
                    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                    : 'bg-rose-500/15 text-rose-400 border-rose-500/40'
                }`}
              >
                {verifiedTick ? 'VERIFY RECEIPT: VALID (CLICK TO TAMPER)' : 'TAMPER DETECTED: DIGEST MISMATCH (RESTORE)'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Capabilities Strip */}
      <div className="mt-5 grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="rounded-xl border hairline bg-[#0B0D11] p-5 flex items-start gap-3.5">
          <Cpu className="w-5 h-5 text-[#C9FF3B] shrink-0 mt-0.5" />
          <div>
            <div className="font-display font-semibold text-white">PyTorch Sparse PatchCache + Flow</div>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Runs neural ViT projection and MLP strictly on changed image tokens (<span className="text-slate-200 font-mono">0.720 ms vs 1.364 ms</span>, <span className="text-[#C9FF3B] font-mono">1.89x</span>) with SafeTensors weights.
            </p>
          </div>
        </div>
        <div className="rounded-xl border hairline bg-[#0B0D11] p-5 flex items-start gap-3.5">
          <Radio className="w-5 h-5 text-sky-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-display font-semibold text-white">Eclipse Zenoh / rmw_zenoh Native</div>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Drop-oldest bounded ring queues on <span className="text-slate-200 font-mono">rt/camera</span> &amp; <span className="text-slate-200 font-mono">rt/cmd_vel</span> eliminate ROS 2 DDS Wi-Fi storms.
            </p>
          </div>
        </div>
        <div className="rounded-xl border hairline bg-[#0B0D11] p-5 flex items-start gap-3.5">
          <Shield className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-display font-semibold text-white">MuJoCo MJCF Manipulation Arena</div>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              Instant tabletop pick-and-place physics with automatic fallback to deterministic 3D rigid-body simulation.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
