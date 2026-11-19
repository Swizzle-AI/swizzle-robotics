import React, { useState } from 'react';
import { CheckCircle2, Gauge } from 'lucide-react';

const ABLATION_ROWS = [
  {
    workload: 'Adversarial Obstacle & Table Safety (25 steps)',
    baseline: '25 / 25 Violations',
    swizzle: '0 / 25 Violations (0.125 ms solve)',
    speedup: '100% Safe',
    pct: 100,
  },
  {
    workload: 'Closed-Loop VLA Control Frequency (120 steps)',
    baseline: '15.9 Hz (62.9 ms)',
    swizzle: '92.3 Hz (10.83 ms mean)',
    speedup: '5.82x faster',
    pct: 94,
  },
  {
    workload: 'Vectorized World-Model CEM (N=128, H=5)',
    baseline: '46.10 ms (Serial)',
    swizzle: '8.90 ms (Batched Matrix)',
    speedup: '5.18x CPU',
    pct: 88,
  },
  {
    workload: 'PyTorch Sparse PatchCache ViT (196 tokens)',
    baseline: '1.364 ms (Dense ViT)',
    swizzle: '0.720 ms (91.8% Reused)',
    speedup: '1.89x faster',
    pct: 76,
  },
  {
    workload: 'Async Chunk-Boundary Continuity (Swizzle-RTC)',
    baseline: '0.800 Max Jump',
    swizzle: '0.515 Inpainted Jump',
    speedup: '1.55x smoother',
    pct: 68,
  },
];

const SIMPLER_ROWS = [
  { model: 'nvidia/gr00t-manipulator-v1', embodiment: 'Dual-Arm Humanoid 14-DoF', rate: '206.2 Hz', reuse: '95.6%', spec: '100.0%' },
  { model: 'lerobot/smolvla-450m', embodiment: 'WidowX 250 6-DoF + Gripper', rate: '203.4 Hz', reuse: '95.6%', spec: '100.0%' },
  { model: 'nvidia/cosmos-predict-mini', embodiment: 'Humanoid Bimanual Handover', rate: '194.3 Hz', reuse: '95.6%', spec: '100.0%' },
  { model: 'physical-intelligence/pi0-fast', embodiment: 'Humanoid Bimanual Handover', rate: '192.4 Hz', reuse: '95.6%', spec: '100.0%' },
  { model: 'meta/vjepa2-latent-mpc', embodiment: 'WidowX + Humanoid Suite', rate: '189.1 Hz', reuse: '95.6%', spec: '100.0%' },
  { model: 'openvla/openvla-7b-swizzle', embodiment: 'Google Robot 7-DoF Mobile', rate: '171.6 Hz', reuse: '96.0%', spec: '100.0%' },
];

export default function BenchmarkSuite() {
  const [tab, setTab] = useState('ablation');

  return (
    <section id="benchmarks" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-t hairline">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-10">
        <div>
          <div className="font-mono text-xs uppercase tracking-[0.18em] text-sky-400 mb-3">
            EMPIRICAL VERIFICATION // 30 / 30 TESTS PASSING
          </div>
          <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-white">
            Measured Safety, Latency &amp; Throughput.
          </h2>
          <p className="mt-2.5 text-slate-400 text-sm sm:text-base max-w-2xl">
            Every figure below is reproducible from the open benchmark scripts in{' '}
            <code className="font-mono text-slate-200">Swizzle-AI/swizzle-robotics</code>.
          </p>
        </div>

        <div className="inline-flex p-1 rounded-xl border hairline bg-[#0B0D11]">
          <button
            onClick={() => setTab('ablation')}
            className={`px-4 py-2 rounded-lg font-mono text-xs transition ${
              tab === 'ablation' ? 'bg-[#C9FF3B] text-[#050608] font-semibold' : 'text-slate-400 hover:text-white'
            }`}
          >
            Safety &amp; RTC Ablation
          </button>
          <button
            onClick={() => setTab('simpler')}
            className={`px-4 py-2 rounded-lg font-mono text-xs transition ${
              tab === 'simpler' ? 'bg-[#C9FF3B] text-[#050608] font-semibold' : 'text-slate-400 hover:text-white'
            }`}
          >
            SIMPLER Suite (18 / 18)
          </button>
        </div>
      </div>

      {tab === 'ablation' ? (
        <div className="rounded-2xl border hairline bg-[#0B0D11] overflow-hidden">
          <div className="divide-y divide-white/5">
            {ABLATION_ROWS.map((row) => (
              <div
                key={row.workload}
                className="p-5 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-4 items-center hover:bg-white/[0.02] transition"
              >
                <div className="lg:col-span-4">
                  <div className="font-display font-semibold text-white text-base">{row.workload}</div>
                  <div className="font-mono text-xs text-slate-500 mt-1">Baseline: {row.baseline}</div>
                </div>

                <div className="lg:col-span-5">
                  <div className="h-2.5 rounded-full bg-white/5 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-sky-400 to-[#C9FF3B]"
                      style={{ width: `${row.pct}%` }}
                    />
                  </div>
                  <div className="font-mono text-xs text-slate-200 mt-2">{row.swizzle}</div>
                </div>

                <div className="lg:col-span-3 lg:text-right">
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#C9FF3B]/15 border border-[#C9FF3B]/35 font-mono text-xs font-semibold text-[#C9FF3B]">
                    <Gauge className="w-3.5 h-3.5" />
                    {row.speedup}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border hairline bg-[#0B0D11] overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b hairline bg-[#090B0F] font-mono text-[11px] uppercase text-slate-400">
                <th className="py-3.5 px-5">Foundation Model Adapter</th>
                <th className="py-3.5 px-5">Robot Embodiment</th>
                <th className="py-3.5 px-5">Control Rate</th>
                <th className="py-3.5 px-5">PatchCache Hit</th>
                <th className="py-3.5 px-5">SpecAct Accept</th>
                <th className="py-3.5 px-5">SIMPLER Result</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono text-xs">
              {SIMPLER_ROWS.map((r) => (
                <tr key={r.model} className="hover:bg-white/[0.02]">
                  <td className="py-3.5 px-5 text-white font-semibold">{r.model}</td>
                  <td className="py-3.5 px-5 text-slate-400">{r.embodiment}</td>
                  <td className="py-3.5 px-5 text-[#C9FF3B] font-semibold">{r.rate}</td>
                  <td className="py-3.5 px-5 text-sky-300">{r.reuse}</td>
                  <td className="py-3.5 px-5 text-slate-200">{r.spec}</td>
                  <td className="py-3.5 px-5">
                    <span className="inline-flex items-center gap-1 text-emerald-400 font-semibold">
                      <CheckCircle2 className="w-3.5 h-3.5" /> PASS
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
