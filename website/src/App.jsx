import React, { useState } from 'react';
import {
  ArrowUpRight,
  Check,
  Copy,
  Cpu,
  ExternalLink,
  Layers,
  ShieldCheck,
  Terminal,
  Zap,
} from 'lucide-react';
import ArenaSimulator from './components/ArenaSimulator';
import BentoShowcase from './components/BentoShowcase';
import BenchmarkSuite from './components/BenchmarkSuite';
import RealRobotStudio from './components/RealRobotStudio';

const TELEMETRY_STRIPS = [
  {
    label: 'CLOSED-LOOP VLA',
    value: '92.3 Hz',
    delta: '5.82x vs 15.9 Hz baseline',
    accent: 'text-limebrand',
  },
  {
    label: 'CBF COLLISION RATE',
    value: '0 / 25',
    delta: '0.125 ms quadratic safety filter',
    accent: 'text-emerald-400',
  },
  {
    label: 'PATCHCACHE VIT REUSE',
    value: '97.15%',
    delta: '0.720 ms vs 1.364 ms dense ViT',
    accent: 'text-cyanbrand',
  },
  {
    label: 'WORLD-MODEL MPC',
    value: '5.18x',
    delta: '8.90 ms vs 46.10 ms CEM baseline',
    accent: 'text-limebrand',
  },
  {
    label: 'HIERARCHICAL S2->S1->S0',
    value: '73.3 Hz',
    delta: '13.64 ms/tick (5 Hz / 30 Hz / 250 Hz)',
    accent: 'text-[#F4F6FA]',
  },
];

const PIPELINE_STEPS = [
  {
    step: '01',
    title: 'Install Edge Package',
    cmd: 'pip install swizzle-robotics[sim]',
    desc: 'Zero-bloat runtime with NumPy/PyTorch sparse ViT hooks, CBF QP filter, and MCAP encoder.',
  },
  {
    step: '02',
    title: 'Verify Safety & RTC Ablations',
    cmd: 'python scripts/bench_safety_rtc_ablation.py',
    desc: 'Reproduce 0/25 CBF safety shields, 1.55x smoother RTC chunks, and 5.18x world-model MPC.',
  },
  {
    step: '03',
    title: 'Stream to Foxglove & LeRobot',
    cmd: 'python scripts/bench_mujoco_closed_loop.py --export-mcap out.mcap',
    desc: 'Emit binary \\x89MCAP0 flight logs alongside LeRobot v2.1 Parquet + JSONL SHA-256 receipts.',
  },
];

export default function App() {
  const [copiedPip, setCopiedPip] = useState(false);

  const handleCopyPip = () => {
    navigator.clipboard?.writeText('pip install swizzle-robotics[sim]');
    setCopiedPip(true);
    setTimeout(() => setCopiedPip(false), 1800);
  };

  return (
    <div className="min-h-screen bg-obsidian text-[#F4F6FA] relative overflow-x-clip">
      {/* Subtle architectural background grid & ambient spotlights */}
      <div className="pointer-events-none fixed inset-0 tech-grid opacity-70" />
      <div
        className="pointer-events-none fixed -top-40 left-1/2 -translate-x-1/2 w-[920px] h-[420px] rounded-full blur-[150px] opacity-20"
        style={{
          background:
            'radial-gradient(circle, rgba(201,255,59,0.28) 0%, rgba(56,189,248,0.12) 45%, transparent 75%)',
        }}
      />

      {/* Sticky Navigation */}
      <header className="sticky top-0 z-50 border-b border-white/[0.085] bg-obsidian/80 backdrop-blur-xl">
        <div className="max-w-[1360px] mx-auto px-5 sm:px-8 h-16 flex items-center justify-between gap-4">
          <a href="#" className="flex items-center gap-3 group">
            <img
              src="/swizzle_avatar.svg"
              alt="Swizzle Robotics"
              className="w-8 h-8 rounded-lg border border-white/15 bg-panel p-0.5 transition-transform group-hover:scale-105"
            />
            <div className="flex items-baseline gap-2">
              <span className="font-display font-semibold tracking-tight text-sm sm:text-base text-white">
                SWIZZLE ROBOTICS
              </span>
              <span className="hidden sm:inline-block font-mono text-[10px] px-1.5 py-0.5 rounded bg-white/[0.06] border border-white/[0.08] text-limebrand">
                RUNTIME
              </span>
            </div>
          </a>

          <nav className="hidden md:flex items-center gap-7 text-xs font-mono uppercase tracking-[0.12em] text-slate-400">
            <a href="#studio" className="text-limebrand hover:text-white transition-colors">
              Real Studio
            </a>
            <a href="#simulator" className="hover:text-white transition-colors">
              Arena
            </a>
            <a href="#architecture" className="hover:text-white transition-colors">
              Architecture
            </a>
            <a href="#benchmarks" className="hover:text-white transition-colors">
              Benchmarks
            </a>
            <a href="#quickstart" className="hover:text-white transition-colors">
              Deploy
            </a>
          </nav>

          <div className="flex items-center gap-2.5">
            <a
              href="https://github.com/Swizzle-AI/swizzle-robotics"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-white/[0.12] bg-white/[0.03] hover:bg-white/[0.07] text-xs font-mono text-slate-200 transition-colors"
            >
              <span>GitHub</span>
              <ExternalLink className="w-3.5 h-3.5 text-slate-400" />
            </a>
            <a
              href="#studio"
              className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-limebrand hover:bg-[#d8ff66] text-obsidian font-mono text-xs font-semibold transition-colors"
            >
              <span>Connect Robot</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </header>

      <main className="relative z-10">
        {/* Hero Split Section */}
        <section className="max-w-[1360px] mx-auto px-5 sm:px-8 pt-10 pb-12 lg:pt-16 lg:pb-16">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 items-center">
            {/* Left Editorial / Value Prop */}
            <div className="lg:col-span-5 flex flex-col items-start">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/[0.12] bg-panel/90 mb-6">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="font-mono text-[11px] tracking-[0.08em] text-slate-300 uppercase">
                  30 / 30 TESTS PASSING · DETERMINISTIC EDGE STACK
                </span>
              </div>

              <h1 className="font-display text-4xl sm:text-5xl xl:text-[54px] font-bold tracking-[-0.03em] leading-[1.04] text-white mb-5">
                Deterministic Physical AI{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-limebrand via-[#e6ff99] to-cyanbrand">
                  at 92.3 Hz.
                </span>
              </h1>

              <p className="text-sm sm:text-base text-slate-300/90 leading-relaxed mb-7 max-w-xl">
                Enterprise VLAs fail in production when unconstrained flow chunks jerk hardware or
                dense ViT encoders stall on Jetson Orin. Swizzle unifies{' '}
                <span className="text-white font-medium">sub-ms CBF safety shields</span>,{' '}
                <span className="text-white font-medium">Real-Time Chunking (RTC)</span>, and{' '}
                <span className="text-white font-medium">sparse 14x14 PatchCache</span> behind a
                single lock-free flight recorder.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 mb-7">
                <a
                  href="#studio"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-limebrand hover:bg-[#d8ff66] text-obsidian font-mono text-xs font-semibold tracking-wide transition-all glow-lime"
                >
                  <Layers className="w-4 h-4" />
                  <span>LAUNCH REAL ROBOT STUDIO</span>
                </a>
                <a
                  href="https://github.com/Swizzle-AI/swizzle-robotics"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-white/[0.14] bg-panel hover:bg-surface text-white font-mono text-xs transition-colors"
                >
                  <span>VIEW BENCHMARK CODE</span>
                  <ArrowUpRight className="w-4 h-4 text-slate-400" />
                </a>
              </div>

              {/* Copyable pip install bar */}
              <div className="w-full max-w-md rounded-xl border border-white/[0.1] bg-panel/95 p-1.5 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2.5 pl-3 overflow-x-auto">
                  <Terminal className="w-4 h-4 text-limebrand shrink-0" />
                  <code className="font-mono text-xs text-slate-200 whitespace-nowrap">
                    pip install swizzle-robotics[sim]
                  </code>
                </div>
                <button
                  onClick={handleCopyPip}
                  type="button"
                  aria-label="Copy pip install command"
                  className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.08] font-mono text-[11px] text-slate-200 transition-colors"
                >
                  {copiedPip ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-slate-400" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>

              {/* 3 quick pill specs */}
              <div className="mt-6 grid grid-cols-3 gap-3 w-full max-w-md pt-5 border-t border-white/[0.08]">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                    Safety Filter
                  </div>
                  <div className="font-mono text-sm font-semibold text-white mt-0.5">
                    0.125 ms QP
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                    World MPC
                  </div>
                  <div className="font-mono text-sm font-semibold text-white mt-0.5">
                    8.90 ms CEM
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-slate-400">
                    Telemetry
                  </div>
                  <div className="font-mono text-sm font-semibold text-white mt-0.5">
                    MCAP + LeRobot
                  </div>
                </div>
              </div>
            </div>

            {/* Right Interactive Simulator */}
            <div id="simulator" className="lg:col-span-7 scroll-mt-20">
              <ArenaSimulator />
            </div>
          </div>
        </section>

        {/* Full-Width Telemetry Strip */}
        <section className="border-y border-white/[0.085] bg-panel/70 backdrop-blur-md">
          <div className="max-w-[1360px] mx-auto px-5 sm:px-8 py-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-5">
            {TELEMETRY_STRIPS.map((item) => (
              <div
                key={item.label}
                className="flex flex-col border-l border-white/[0.08] pl-4 first:border-l-0 first:pl-0"
              >
                <span className="font-mono text-[10px] tracking-[0.14em] text-slate-400 uppercase">
                  {item.label}
                </span>
                <span className={`font-display text-2xl sm:text-3xl font-bold mt-1 ${item.accent}`}>
                  {item.value}
                </span>
                <span className="font-mono text-[11px] text-slate-400 mt-0.5">{item.delta}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Interactive Real-Life Robot Arena Studio (Live Backend + Tuning) */}
        <RealRobotStudio />

        {/* Bento Architecture Section */}
        <div id="architecture" className="scroll-mt-20">
          <BentoShowcase />
        </div>

        {/* Benchmark Evidence Suite */}
        <div id="benchmarks" className="scroll-mt-20">
          <BenchmarkSuite />
        </div>

        {/* Enterprise Quickstart Strip */}
        <section
          id="quickstart"
          className="max-w-[1360px] mx-auto px-5 sm:px-8 py-14 sm:py-20 scroll-mt-20"
        >
          <div className="rounded-2xl border border-white/[0.1] bg-panel/90 p-6 sm:p-10 relative overflow-hidden">
            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 mb-8">
              <div>
                <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-cyanbrand mb-2">
                  <Cpu className="w-3.5 h-3.5" />
                  <span>EDGE DEPLOYMENT PROTOCOL</span>
                </div>
                <h2 className="font-display text-2xl sm:text-3xl font-bold tracking-tight text-white">
                  Built for Jetson Orin, ROS 2 Humble & Zenoh Fleets.
                </h2>
              </div>
              <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-white/[0.04] border border-white/[0.08]">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                  Apache-2.0
                </span>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-white/[0.04] border border-white/[0.08]">
                  <Zap className="w-3.5 h-3.5 text-limebrand" />
                  Zero Cloud Dependency
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {PIPELINE_STEPS.map((item) => (
                <div
                  key={item.step}
                  className="rounded-xl border border-white/[0.085] bg-surface/80 p-5 flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-mono text-xs text-limebrand font-semibold">
                        STEP {item.step}
                      </span>
                    </div>
                    <h3 className="font-display text-base font-semibold text-white mb-2">
                      {item.title}
                    </h3>
                    <p className="text-xs text-slate-400 leading-relaxed mb-4">{item.desc}</p>
                  </div>
                  <div className="rounded-lg bg-obsidian border border-white/[0.08] px-3 py-2 font-mono text-[11px] text-slate-200 overflow-x-auto whitespace-nowrap">
                    $ {item.cmd}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.085] bg-panel/60">
        <div className="max-w-[1360px] mx-auto px-5 sm:px-8 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img
              src="/swizzle_avatar.svg"
              alt="Swizzle AI"
              className="w-6 h-6 rounded border border-white/10"
            />
            <span className="font-mono text-xs text-slate-400">
              Swizzle AI · Edge Robotics & Inference Infrastructure
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-5 font-mono text-xs text-slate-400">
            <a
              href="https://github.com/Swizzle-AI/swizzle-robotics"
              target="_blank"
              rel="noreferrer"
              className="hover:text-white transition-colors"
            >
              Swizzle-AI/swizzle-robotics
            </a>
            <a
              href="https://github.com/Swizzle-AI/swizzle-runtime"
              target="_blank"
              rel="noreferrer"
              className="hover:text-white transition-colors"
            >
              Swizzle-AI/swizzle-runtime
            </a>
            <span className="text-slate-600">Apache-2.0 Open Source</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
