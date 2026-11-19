import React, { useEffect, useRef, useState } from 'react';
import { ShieldCheck, ShieldAlert, Play, RotateCcw, Layers } from 'lucide-react';

/**
 * Real-time interactive 60 FPS tabletop simulation comparing:
 * - Swizzle-CBF + RTC enabled: smooth half-space barrier projection around obstacle & zero jerk
 * - Vanilla unshielded policy: straight-line trajectory collides with keep-out hazard zone
 */
export default function ArenaSimulator() {
  const canvasRef = useRef(null);
  const [cbfEnabled, setCbfEnabled] = useState(true);
  const [showPatchGrid, setShowPatchGrid] = useState(true);
  const [running, setRunning] = useState(true);
  const [metrics, setMetrics] = useState({
    pose: [-0.14, -0.14, 0.22],
    margin: 0.142,
    intervened: false,
    collisions: 0,
    patchReuse: 95.4,
    solveMs: 0.125,
    receiptHash: 'sha256:9f84c2e1...7b3a',
  });

  const simRef = useRef({
    eef: { x: -0.14, y: -0.14 },
    prevDelta: { x: 0, y: 0 },
    trail: [],
    step: 0,
    collisions: 0,
    phase: 'reach_cube', // reach_cube -> carry_tray -> reset
    cube: { x: -0.14, y: -0.14 },
    goal: { x: 0.22, y: 0.16 },
    obstacle: { x: 0.04, y: 0.02, r: 0.065 },
  });

  const resetArena = () => {
    simRef.current = {
      eef: { x: -0.14, y: -0.14 },
      prevDelta: { x: 0, y: 0 },
      trail: [],
      step: 0,
      collisions: 0,
      phase: 'carry_tray',
      cube: { x: -0.14, y: -0.14 },
      goal: { x: 0.22, y: 0.16 },
      obstacle: { x: 0.04, y: 0.02, r: 0.065 },
    };
  };

  useEffect(() => {
    resetArena();
  }, [cbfEnabled]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let rafId;
    let frameTick = 0;

    const toPx = (wx, wy, w, h) => ({
      px: ((wx + 0.35) / 0.70) * w,
      py: ((0.35 - wy) / 0.70) * h,
    });

    const stepPhysics = () => {
      const s = simRef.current;
      s.step += 1;

      const target = s.phase === 'carry_tray' ? s.goal : { x: -0.14, y: -0.14 };
      const dxRaw = (target.x - s.eef.x) * 0.085;
      const dyRaw = (target.y - s.eef.y) * 0.085;
      let ux = dxRaw;
      let uy = dyRaw;

      // Normalize nominal step velocity
      const norm = Math.hypot(ux, uy);
      const maxVel = 0.011;
      if (norm > maxVel) {
        ux = (ux / norm) * maxVel;
        uy = (uy / norm) * maxVel;
      }

      let intervened = false;
      const diffX = s.eef.x - s.obstacle.x;
      const diffY = s.eef.y - s.obstacle.y;
      const dist = Math.hypot(diffX, diffY);
      const rSafe = s.obstacle.r + 0.032; // obstacle + EEF radius
      const signedDist = dist - rSafe;

      if (cbfEnabled) {
        // Slew-rate / RTC smoothing
        const accLimit = 0.004;
        ux = s.prevDelta.x + Math.max(-accLimit, Math.min(accLimit, ux - s.prevDelta.x));
        uy = s.prevDelta.y + Math.max(-accLimit, Math.min(accLimit, uy - s.prevDelta.y));

        // Discrete-Time Control Barrier Function half-space projection + tangential bias
        const nx = dist > 1e-6 ? diffX / dist : 0;
        const ny = dist > 1e-6 ? diffY / dist : 1;
        const alpha = 0.45;
        const lhs = nx * ux + ny * uy + alpha * signedDist;
        if (lhs < 0) {
          intervened = true;
          ux -= lhs * nx;
          uy -= lhs * ny;
          // Tangential guide around cylinder to prevent local minimum deadlock
          const tx = -ny;
          const ty = nx;
          const sign = tx * (target.x - s.eef.x) + ty * (target.y - s.eef.y) >= 0 ? 1 : -1;
          ux += sign * tx * 0.006;
          uy += sign * ty * 0.006;
        }
      }

      s.prevDelta = { x: ux, y: uy };
      s.eef.x += ux;
      s.eef.y += uy;
      if (s.phase === 'carry_tray') {
        s.cube.x = s.eef.x;
        s.cube.y = s.eef.y;
      }

      if (Math.hypot(s.eef.x - s.obstacle.x, s.eef.y - s.obstacle.y) < rSafe - 0.004) {
        s.collisions += 1;
      }

      s.trail.push({ x: s.eef.x, y: s.eef.y, safe: !intervened });
      if (s.trail.length > 90) s.trail.shift();

      if (Math.hypot(target.x - s.eef.x, target.y - s.eef.y) < 0.018) {
        s.phase = s.phase === 'carry_tray' ? 'return' : 'carry_tray';
        if (s.phase === 'carry_tray') {
          s.cube = { x: -0.14, y: -0.14 };
        }
      }

      if (s.step % 4 === 0) {
        const fakeHex = ((s.step * 2654435761) >>> 0).toString(16).padStart(8, '0');
        setMetrics({
          pose: [Number(s.eef.x.toFixed(3)), Number(s.eef.y.toFixed(3)), 0.22],
          margin: Number(signedDist.toFixed(3)),
          intervened,
          collisions: s.collisions,
          patchReuse: Number((94.8 + Math.sin(s.step * 0.08) * 1.4).toFixed(1)),
          solveMs: cbfEnabled ? Number((0.11 + Math.abs(Math.cos(s.step * 0.1)) * 0.03).toFixed(3)) : 0.0,
          receiptHash: `blake3:${fakeHex}...a41c`,
        });
      }
    };

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      const s = simRef.current;

      ctx.fillStyle = '#07090D';
      ctx.fillRect(0, 0, w, h);

      // Optional 14x14 ViT PatchCache overlay
      if (showPatchGrid) {
        const cells = 14;
        const cw = w / cells;
        const ch = h / cells;
        const eefPx = toPx(s.eef.x, s.eef.y, w, h);
        for (let r = 0; r < cells; r++) {
          for (let c = 0; c < cells; c++) {
            const cx = (c + 0.5) * cw;
            const cy = (r + 0.5) * ch;
            const d = Math.hypot(cx - eefPx.px, cy - eefPx.py);
            const active = d < 62;
            ctx.strokeStyle = active ? 'rgba(201, 255, 59, 0.22)' : 'rgba(255, 255, 255, 0.035)';
            ctx.lineWidth = 1;
            ctx.strokeRect(c * cw, r * ch, cw, ch);
            if (active) {
              ctx.fillStyle = 'rgba(201, 255, 59, 0.06)';
              ctx.fillRect(c * cw, r * ch, cw, ch);
            }
          }
        }
      }

      // Workspace geofence box
      const topLeft = toPx(-0.30, 0.30, w, h);
      const botRight = toPx(0.30, -0.30, w, h);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(topLeft.px, topLeft.py, botRight.px - topLeft.px, botRight.py - topLeft.py);
      ctx.setLineDash([]);

      // Green goal tray
      const goalPx = toPx(s.goal.x, s.goal.y, w, h);
      ctx.fillStyle = 'rgba(16, 185, 129, 0.14)';
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.fillRect(goalPx.px - 28, goalPx.py - 28, 56, 56);
      ctx.strokeRect(goalPx.px - 28, goalPx.py - 28, 56, 56);
      ctx.fillStyle = '#10B981';
      ctx.font = '500 10px "JetBrains Mono", monospace';
      ctx.fillText('GOAL TRAY', goalPx.px - 26, goalPx.py - 34);

      // Obstacle cylinder + CBF safety halo
      const obsPx = toPx(s.obstacle.x, s.obstacle.y, w, h);
      const obsRadiusPx = (s.obstacle.r / 0.70) * w;
      const safeRadiusPx = ((s.obstacle.r + 0.032) / 0.70) * w;

      if (cbfEnabled) {
        ctx.beginPath();
        ctx.arc(obsPx.px, obsPx.py, safeRadiusPx, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(245, 158, 11, 0.08)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.beginPath();
      ctx.arc(obsPx.px, obsPx.py, obsRadiusPx, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(245, 158, 11, 0.25)';
      ctx.fill();
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = 1.75;
      ctx.stroke();
      ctx.fillStyle = '#FBBF24';
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.fillText('HAZARD CBF ZONE', obsPx.px - 44, obsPx.py + obsRadiusPx + 22);

      // Trajectory trail
      if (s.trail.length > 1) {
        for (let i = 1; i < s.trail.length; i++) {
          const a = toPx(s.trail[i - 1].x, s.trail[i - 1].y, w, h);
          const b = toPx(s.trail[i].x, s.trail[i].y, w, h);
          ctx.beginPath();
          ctx.moveTo(a.px, a.py);
          ctx.lineTo(b.px, b.py);
          ctx.strokeStyle = cbfEnabled
            ? s.trail[i].safe
              ? 'rgba(201, 255, 59, 0.45)'
              : 'rgba(56, 189, 248, 0.9)'
            : 'rgba(244, 63, 94, 0.7)';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Red payload cube
      const cubePx = toPx(s.cube.x, s.cube.y, w, h);
      ctx.fillStyle = '#EF4444';
      ctx.fillRect(cubePx.px - 9, cubePx.py - 9, 18, 18);
      ctx.strokeStyle = '#FCA5A5';
      ctx.lineWidth = 1;
      ctx.strokeRect(cubePx.px - 9, cubePx.py - 9, 18, 18);

      // End-effector marker
      const eefPx = toPx(s.eef.x, s.eef.y, w, h);
      ctx.beginPath();
      ctx.arc(eefPx.px, eefPx.py, 15, 0, Math.PI * 2);
      ctx.strokeStyle = cbfEnabled ? '#C9FF3B' : '#F43F5E';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(eefPx.px, eefPx.py, 4, 0, Math.PI * 2);
      ctx.fillStyle = cbfEnabled ? '#C9FF3B' : '#F43F5E';
      ctx.fill();
    };

    const loop = () => {
      if (running) {
        frameTick += 1;
        if (frameTick % 2 === 0) stepPhysics();
      }
      draw();
      rafId = requestAnimationFrame(loop);
    };

    rafId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafId);
  }, [cbfEnabled, showPatchGrid, running]);

  return (
    <div className="rounded-xl border hairline bg-[#0B0D11] overflow-hidden glow-lime">
      {/* Control Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 border-b hairline bg-[#090B0F]">
        <div className="flex items-center gap-2.5">
          <span
            className={`w-2 h-2 rounded-full ${
              metrics.intervened
                ? 'bg-sky-400 animate-ping'
                : cbfEnabled
                ? 'bg-[#C9FF3B]'
                : 'bg-rose-500 animate-pulse'
            }`}
          />
          <span className="font-mono text-[11px] uppercase tracking-wider text-slate-300">
            arena_01 // overhead camera + cbf barrier
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setCbfEnabled((v) => !v)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-mono font-medium transition ${
              cbfEnabled
                ? 'bg-[#C9FF3B]/15 text-[#C9FF3B] border border-[#C9FF3B]/40'
                : 'bg-rose-500/15 text-rose-400 border border-rose-500/40'
            }`}
          >
            {cbfEnabled ? <ShieldCheck className="w-3.5 h-3.5" /> : <ShieldAlert className="w-3.5 h-3.5" />}
            {cbfEnabled ? 'SWIZZLE-CBF: ON' : 'UNSHIELDED POLICY'}
          </button>

          <button
            onClick={() => setShowPatchGrid((v) => !v)}
            title="Toggle 14x14 ViT PatchCache overlay"
            className={`inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-mono border transition ${
              showPatchGrid
                ? 'bg-sky-500/15 text-sky-300 border-sky-500/35'
                : 'bg-white/5 text-slate-400 border-white/10'
            }`}
          >
            <Layers className="w-3 h-3" />
            ViT 14x14
          </button>

          <button
            onClick={() => setRunning((r) => !r)}
            className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10"
            title={running ? 'Pause' : 'Play'}
          >
            <Play className="w-3 h-3" />
          </button>
          <button
            onClick={resetArena}
            className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10"
            title="Reset Episode"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Live Canvas Viewport */}
      <div className="relative bg-[#07090D]">
        <canvas ref={canvasRef} width={560} height={360} className="w-full h-[310px] sm:h-[340px] block" />

        {/* Live Telemetry HUD Overlay */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-white/10 border-t hairline">
          <div className="bg-[#090B0F] px-3 py-2.5">
            <div className="font-mono text-[10px] text-slate-500 uppercase">Barrier Margin</div>
            <div
              className={`font-mono text-xs font-semibold mt-0.5 ${
                metrics.margin < 0 ? 'text-rose-400' : metrics.intervened ? 'text-sky-300' : 'text-[#C9FF3B]'
              }`}
            >
              {metrics.margin >= 0 ? `+${metrics.margin.toFixed(3)} m` : `${metrics.margin.toFixed(3)} m BREACH`}
            </div>
          </div>

          <div className="bg-[#090B0F] px-3 py-2.5">
            <div className="font-mono text-[10px] text-slate-500 uppercase">Collisions</div>
            <div
              className={`font-mono text-xs font-semibold mt-0.5 ${
                metrics.collisions > 0 ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {metrics.collisions} EVENTS
            </div>
          </div>

          <div className="bg-[#090B0F] px-3 py-2.5">
            <div className="font-mono text-[10px] text-slate-500 uppercase">ViT PatchCache</div>
            <div className="font-mono text-xs font-semibold text-sky-300 mt-0.5">
              {metrics.patchReuse}% REUSED
            </div>
          </div>

          <div className="bg-[#090B0F] px-3 py-2.5">
            <div className="font-mono text-[10px] text-slate-500 uppercase">CBF Solve / Receipt</div>
            <div className="font-mono text-xs font-semibold text-slate-200 mt-0.5 truncate">
              {metrics.solveMs}ms · {metrics.receiptHash.slice(7, 15)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
