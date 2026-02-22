"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Settings2,
  Activity,
  Brain,
  Square,
  FlaskConical,
  Zap,
  BarChart3,
  Trophy,
  Timer,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Eye,
  Sliders,
} from "lucide-react";
import type { JointInfo, SimState } from "@/types/robotControl";
import { useMujocoScene } from "@/hooks/useMujocoScene";
import { useRobot } from "@/hooks/useRobot";
import { ReferenceMotion } from "@/utils/ReferenceMotion";
import type { VmdKeyframe } from "@/utils/VmdLoader";
import { PPOAgent } from "@/rl/PPOAgent";
import { RLEnvironmentCore } from "@/rl/RLEnvironmentCore";
import { makeObsLayout } from "@/rl/reward";
import {
  MUJOCO_JOINT_ORDER,
  JOINT_CONSTRAINTS,
  STANDING_POSE,
  applyJointConstraints,
} from "@/constants/jointConfig";
import type { JointName } from "@/constants/jointConfig";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
);

// ─── iframe RPC ──────────────────────────────────────────────────
type PendingEntry = { resolve: (v: any) => void; reject: (e: any) => void };

function makeIframeRPC(
  iframeWindowRef: React.MutableRefObject<Window | null>,
  iframeRef: React.RefObject<HTMLIFrameElement>,
) {
  let nextId = 1;
  const pending = new Map<number, PendingEntry>();

  function onMessage(ev: MessageEvent) {
    const iframeWin = iframeRef.current?.contentWindow;
    if (!iframeWin || ev.source !== iframeWin) return;
    const msg = ev.data ?? {};
    const { type, requestId, payload, error } = msg;

    if (type === "RPC_ERROR" && typeof requestId === "number") {
      const p = pending.get(requestId);
      if (p) {
        pending.delete(requestId);
        p.reject(new Error(error ?? "RPC_ERROR"));
      }
      return;
    }
    if (typeof requestId === "number" && type && String(type).endsWith("_RESULT")) {
      const p = pending.get(requestId);
      if (p) {
        pending.delete(requestId);
        p.resolve(payload);
      }
    }
  }

  return {
    attach() {
      window.addEventListener("message", onMessage);
    },
    detach() {
      window.removeEventListener("message", onMessage);
      pending.forEach((p) => p.reject(new Error("RPC detached")));
      pending.clear();
    },
    request<T = any>(type: string, payload?: any): Promise<T> {
      const win = iframeWindowRef.current;
      if (!win) return Promise.reject(new Error("No iframe window"));
      const id = nextId++;
      return new Promise<T>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        win.postMessage({ type, requestId: id, payload }, "*");
      });
    },
  };
}

// ─── EuradBench 타입 ─────────────────────────────────────────────

type BenchTier = "TIER_1_STAND" | "TIER_2_BALANCE" | "TIER_3_WALK" | "TIER_4_ADAPTIVE" | "TIER_5_RECOVERY";

type PostureVerdict = "FALLEN" | "FALLING" | "UNSTABLE" | "STANDING" | "WALKING" | "UNKNOWN";

type TorsoState = {
  height: number;
  pitch: number;
  roll: number;
  x: number;
  y: number;
};

type AutonomousConfig = {
  kp?: number;
  kd?: number;
  duration_ms?: number;
  frequency_hz?: number;
};

type LLMAutonomousResponse = {
  analysis?: string;
  motions?: Array<{ joint: string; angle: number; time: number }>;
  config?: AutonomousConfig;
  observe?: string[];
  reset?: boolean;
  hypothesis?: string;
  experiment?: string;
};

type BenchIteration = {
  iter: number;
  timestamp: number;
  verdict: PostureVerdict;
  torsoHeight: number;
  torsoPitch: number;
  standDuration: number;
  walkDistance: number;
  quality: number;
  tier1Pass: boolean;
  tier2Pass: boolean;
  tier3Pass: boolean;
  analysisSnippet: string;
  hypothesis: string;
  experiment: string;
  usedConfig: AutonomousConfig | null;
  requestedSensors: string[];
  autonomyScore: number;
  didReset: boolean;
};

type BenchResult = {
  modelName: string;
  totalIterations: number;
  maxStandDuration: number;
  maxWalkDistance: number;
  tier1Iter: number | null;
  tier2Iter: number | null;
  tier3Iter: number | null;
  highestTier: string;
  avgQuality: number;
  avgAutonomy: number;
  history: BenchIteration[];
};

// ─── 판단 상수 ───────────────────────────────────────────────────

const POSTURE = {
  FALLEN_HEIGHT: 0.25,
  FALLING_HEIGHT: 0.35,
  STANDING_MIN: 0.40,
  FALLEN_TILT: 1.2,
  UNSTABLE_TILT: 0.5,
  STABLE_TILT: 0.15,
  WALK_SPEED_MIN: 0.05,
  STAND_DURATION_TIER1: 3,
  WALK_DISTANCE_TIER3: 1.0,
};

const AVAILABLE_SENSORS = [
  "foot_contact",
  "joint_torque",
  "center_of_mass",
  "velocity",
  "energy",
  "momentum",
  "joint_positions_all",
  "body_positions",
  "ground_reaction",
  "stability",
] as const;

function judgePosture(
  torso: TorsoState,
  prev: TorsoState | null,
  dt: number,
): { verdict: PostureVerdict; quality: number } {
  const { height, pitch, roll } = torso;
  const tilt = Math.sqrt(pitch * pitch + roll * roll);

  if (height < POSTURE.FALLEN_HEIGHT || tilt > POSTURE.FALLEN_TILT) {
    return { verdict: "FALLEN", quality: 0 };
  }
  if (height < POSTURE.FALLING_HEIGHT) {
    return { verdict: "FALLING", quality: 15 };
  }
  if (height >= POSTURE.STANDING_MIN && tilt < POSTURE.STABLE_TILT) {
    if (prev) {
      const dx = torso.x - prev.x;
      const dy = torso.y - prev.y;
      const speed = Math.sqrt(dx * dx + dy * dy) / Math.max(0.001, dt);
      if (speed > POSTURE.WALK_SPEED_MIN) {
        const q = Math.min(100, 70 + (1 - tilt / POSTURE.STABLE_TILT) * 30);
        return { verdict: "WALKING", quality: q };
      }
    }
    const q = Math.min(100, 50 + (height - POSTURE.STANDING_MIN) * 200 + (1 - tilt / POSTURE.STABLE_TILT) * 20);
    return { verdict: "STANDING", quality: Math.min(100, Math.max(0, q)) };
  }
  if (height >= POSTURE.STANDING_MIN && tilt > POSTURE.UNSTABLE_TILT) {
    return { verdict: "UNSTABLE", quality: 25 };
  }
  return { verdict: "UNSTABLE", quality: 20 };
}

function verdictColor(v: PostureVerdict): string {
  switch (v) {
    case "FALLEN": return "#ef4444";
    case "FALLING": return "#f97316";
    case "UNSTABLE": return "#eab308";
    case "STANDING": return "#22c55e";
    case "WALKING": return "#3b82f6";
    default: return "#6b7280";
  }
}

function verdictEmoji(v: PostureVerdict): string {
  switch (v) {
    case "FALLEN": return "🔴";
    case "FALLING": return "🟠";
    case "UNSTABLE": return "🟡";
    case "STANDING": return "🟢";
    case "WALKING": return "🏃";
    default: return "⚪";
  }
}

function computeAutonomyScore(resp: LLMAutonomousResponse, prevHistory: BenchIteration[]): number {
  let score = 0;

  if (resp.observe && resp.observe.length > 0) score += 10;

  if (resp.config && (resp.config.kp !== undefined || resp.config.kd !== undefined || resp.config.duration_ms !== undefined)) {
    score += 15;
  }

  if (resp.hypothesis && resp.hypothesis.length > 10) score += 10;
  if (resp.experiment && resp.experiment.length > 10) score += 10;

  if (resp.reset) {
    const lastVerdict = prevHistory.length > 0 ? prevHistory[prevHistory.length - 1].verdict : "UNKNOWN";
    if (lastVerdict === "FALLEN" || lastVerdict === "FALLING") {
      score += 5;
    } else {
      score += 2;
    }
  }

  if (prevHistory.length > 0) {
    const lastEntry = prevHistory[prevHistory.length - 1];
    const prevConfig = lastEntry.usedConfig;
    const currConfig = resp.config;

    const configChanged = currConfig && prevConfig &&
      (currConfig.kp !== prevConfig.kp || currConfig.kd !== prevConfig.kd);
    const sensorChanged = resp.observe &&
      lastEntry.requestedSensors.join(",") !== resp.observe.join(",");

    if (configChanged || sensorChanged) score += 20;
  }

  if (prevHistory.length >= 2) {
    const prev = prevHistory[prevHistory.length - 1];
    const prevPrev = prevHistory[prevHistory.length - 2];
    if (prev.hypothesis && prev.quality > prevPrev.quality) {
      score += 20;
    }
  }

  return Math.min(100, score);
}

// ─── MuJoCo context builder ─────────────────────────────────────

function buildMujocoContext(): string {
  const jointSpec = MUJOCO_JOINT_ORDER.map((name) => {
    const c = JOINT_CONSTRAINTS[name];
    return `${name}: [${c.min}, ${c.max}] ${c.note}`;
  }).join("\n");
  return [
    "MUJOCO_JOINT_CONTEXT:",
    `Available joints (${MUJOCO_JOINT_ORDER.length}):`,
    jointSpec,
    "",
    "Rules:",
    "- Use ONLY joints from the list above.",
    "- Angles in radians.",
    "- l_knee, r_knee >= 0. l_el <= 0, r_el >= 0.",
    "- Anti-phase legs, arms opposite to same-side leg.",
    "- IMPORTANT: l_hip_roll range is [-1.0472, 0], r_hip_roll range is [0, 1.0472].",
    "  Both hip_roll=0 means legs straight. Non-zero = legs spread outward.",
    "  Always include hip_roll in your motions to keep legs together!",
  ].join("\n");
}

function buildAutonomyPrompt(): string {
  return [
    "",
    "═══ AUTONOMOUS CONTROL ═══",
    "You have full autonomy over the robot. Beyond 'motions', you can use these optional fields:",
    "",
    "1. \"config\": { \"kp\": number, \"kd\": number, \"duration_ms\": number }",
    "   - Adjust PD gains. Default kp=80, kd=6. Lower = softer, higher = stiffer.",
    "   - duration_ms: how long each motion step lasts (default 2000).",
    "",
    `2. "observe": [${AVAILABLE_SENSORS.map(s => `"${s}"`).join(", ")}]`,
    "   - Request additional sensor data. Next iteration will include this data.",
    "   - foot_contact: per-foot contact booleans + estimated forces",
    "   - joint_torque: current torque on each joint",
    "   - center_of_mass: CoM position",
    "   - velocity: linear + angular velocity",
    "   - energy: power consumption",
    "   - momentum: linear momentum",
    "",
    "3. \"reset\": true",
    "   - Request environment reset (robot returns to T-pose). Use after falls.",
    "",
    "4. \"hypothesis\": \"string\"",
    "   - State what you expect to happen. Will be compared with actual result.",
    "",
    "5. \"experiment\": \"string\"",
    "   - Describe what you're testing in this iteration.",
    "",
    "CRITICAL: Every motion MUST include ALL leg joints (hip_pitch, hip_roll, hip_yaw, knee, ank_pitch, ank_roll for both legs).",
    "If you omit a joint, it will still be held at a safe default, but explicit control is better.",
    "",
    "RESPONSE FORMAT (JSON):",
    "{",
    "  \"analysis\": \"한국어로 현재 상태 진단 + 전략\",",
    "  \"hypothesis\": \"이번 모션의 예상 결과\",",
    "  \"experiment\": \"이번에 테스트하는 것\",",
    "  \"config\": { \"kp\": 60, \"kd\": 4 },",
    "  \"observe\": [\"foot_contact\", \"joint_torque\"],",
    "  \"reset\": false,",
    "  \"motions\": [{\"joint\":\"...\", \"angle\":0.0, \"time\":0}, ...]",
    "}",
    "All fields except 'motions' are optional. Be scientific: hypothesize, test, observe, adapt.",
  ].join("\n");
}

function resolveToMujocoJoint(name: string): JointName | null {
  if ((MUJOCO_JOINT_ORDER as readonly string[]).includes(name)) return name as JointName;
  const norm = name.toLowerCase().replace(/[\s\-_.]/g, "");
  for (const jn of MUJOCO_JOINT_ORDER) {
    const jnNorm = jn.toLowerCase().replace(/[\s\-_.]/g, "");
    if (jnNorm === norm || jnNorm.includes(norm) || norm.includes(jnNorm)) return jn;
  }
  return null;
}

// ★ STANDING_POSE를 기본값으로 사용하여 LLM이 지정하지 않은 관절도 안전한 위치 유지
function motorMotionsToKeyframes(
  motions: Array<{ joint: string; angle: number; time: number }>,
  currentJoints?: Record<string, JointInfo>,
): VmdKeyframe[] {
  const timeMap = new Map<number, Record<string, number>>();
  for (const m of motions) {
    const resolved = resolveToMujocoJoint(m.joint);
    if (!resolved) continue;
    if (!timeMap.has(m.time ?? 0)) timeMap.set(m.time ?? 0, {});
    timeMap.get(m.time ?? 0)![resolved] = m.angle;
  }
  const sortedTimes = [...timeMap.keys()].sort((a, b) => a - b);

  // ★ 기본값을 STANDING_POSE로 초기화 (0이 아님!)
  // LLM이 지정하지 않은 관절도 안전한 위치를 유지
  const currentPose: Record<string, number> = {};
  for (const name of MUJOCO_JOINT_ORDER) {
    currentPose[name] =
      currentJoints?.[name]?.val ?? STANDING_POSE[name as JointName] ?? 0;
  }

  const keyframes: VmdKeyframe[] = [];
  for (const timeMs of sortedTimes) {
    Object.assign(currentPose, timeMap.get(timeMs)!);
    const constrained = applyJointConstraints({ ...currentPose });
    const timeSec = timeMs / 1000;
    keyframes.push({ frame: Math.round(timeSec * 30), timeSec, pose: { ...constrained } });
  }
  return keyframes;
}

// ★ 모든 관절에 대한 STANDING_POSE 타겟을 생성하는 헬퍼
function buildStandingTargets(): Record<string, number> {
  const targets: Record<string, number> = {};
  for (const name of MUJOCO_JOINT_ORDER) {
    targets[name] = STANDING_POSE[name as JointName] ?? 0;
  }
  return targets;
}

// ═════════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═════════════════════════════════════════════════════════════════

export default function RobotControlPanel() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const iframeWindowRef = useRef<Window | null>(null);

  const { registerIframeWindow, resetPose, pauseSimulation, resumeSimulation } =
    useMujocoScene();
  const { activeRobotType, setActiveRobotType, setActiveRobotOwner, setActiveRobotName } =
    useRobot();

  const [joints, setJoints] = useState<Record<string, JointInfo>>({});
  const [simState, setSimState] = useState<SimState>("PAUSED");
  const [simTime, setSimTime] = useState(0);
  const [fps, setFps] = useState(0);

  const torsoRef = useRef<TorsoState | null>(null);
  const prevTorsoRef = useRef<TorsoState | null>(null);
  const prevTorsoTimeRef = useRef(0);
  const [currentVerdict, setCurrentVerdict] = useState<PostureVerdict>("UNKNOWN");
  const [currentQuality, setCurrentQuality] = useState(0);
  const [currentTorso, setCurrentTorso] = useState<TorsoState | null>(null);

  const extraSensorsRef = useRef<Record<string, any>>({});
  const [activeSensors, setActiveSensors] = useState<string[]>([]);

  const [benchRunning, setBenchRunning] = useState(false);
  const [benchModelName, setBenchModelName] = useState("gpt-5-mini");
  const [benchMaxIter, setBenchMaxIter] = useState(100);
  const benchAbortRef = useRef(false);

  const [benchHistory, setBenchHistory] = useState<BenchIteration[]>([]);
  const [benchIter, setBenchIter] = useState(0);
  const [benchStatus, setBenchStatus] = useState("");

  const [tier1Iter, setTier1Iter] = useState<number | null>(null);
  const [tier2Iter, setTier2Iter] = useState<number | null>(null);
  const [tier3Iter, setTier3Iter] = useState<number | null>(null);

  const standStartRef = useRef<number | null>(null);
  const [maxStandDuration, setMaxStandDuration] = useState(0);
  const [maxWalkDistance, setMaxWalkDistance] = useState(0);
  const walkOriginRef = useRef<{ x: number; y: number } | null>(null);

  const [analysisLog, setAnalysisLog] = useState<Array<{ iter: number; text: string; hypothesis?: string; experiment?: string }>>([]);

  const driveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [isDriving, setIsDriving] = useState(false);

  const jointsRef = useRef(joints);
  useEffect(() => { jointsRef.current = joints; }, [joints]);

  const rpcRef = useRef<ReturnType<typeof makeIframeRPC> | null>(null);

  const benchHistoryRef = useRef<BenchIteration[]>([]);
  useEffect(() => { benchHistoryRef.current = benchHistory; }, [benchHistory]);

  useEffect(() => {
    if (activeRobotType !== "MJCF") {
      setActiveRobotType("MJCF");
      setActiveRobotOwner("placeholder");
      setActiveRobotName("humanoid");
    }
  }, [activeRobotType, setActiveRobotName, setActiveRobotOwner, setActiveRobotType]);

  const sendMessage = useCallback((msg: any) => {
    iframeWindowRef.current?.postMessage(msg, "*");
  }, []);

  const handleIframeLoad = useCallback(() => {
    const win = iframeRef.current?.contentWindow ?? null;
    if (!win) return;
    iframeWindowRef.current = win;
    registerIframeWindow(win);
  }, [registerIframeWindow]);

  useEffect(() => {
    const rpc = makeIframeRPC(iframeWindowRef, iframeRef);
    rpc.attach();
    rpcRef.current = rpc;
    return () => { rpc.detach(); rpcRef.current = null; };
  }, []);

  useEffect(() => {
    const handler = (ev: Event) => {
      const msg = (ev as CustomEvent).detail;
      if (msg && iframeWindowRef.current) {
        iframeWindowRef.current.postMessage(msg, "*");
      }
    };
    window.addEventListener("mujoco:postMessage", handler);
    return () => window.removeEventListener("mujoco:postMessage", handler);
  }, []);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const iframeWin = iframeRef.current?.contentWindow;
      if (!iframeWin || event.source !== iframeWin) return;
      iframeWindowRef.current = iframeWin;
      const { type, payload } = event.data ?? {};

      switch (type) {
        case "IFRAME_READY":
          registerIframeWindow(iframeWin);
          break;
        case "SCENE_LOADED":
          setJoints({});
          resumeSimulation();
          setSimState("RUNNING");
          sendMessage({ type: "GET_JOINT_INFO" });
          // ★ 씬 로드 직후 STANDING_POSE로 PD 잡아줌
          setTimeout(() => {
            sendMessage({
              type: "SET_JOINT_TARGETS_PD",
              enabled: true,
              targets: buildStandingTargets(),
              kp: 80,
              kd: 6,
            });
          }, 300);
          break;
        case "SYNC_STATE": {
          if (!payload) return;
          setSimTime(payload.time ?? 0);
          setFps(payload.fps ?? 0);

          const incoming = (payload.joints ?? {}) as Record<string, JointInfo>;
          setJoints(incoming);

          if (payload.extraSensors) {
            extraSensorsRef.current = payload.extraSensors;
          }

          if (payload.torso) {
            const torso = payload.torso as TorsoState;
            const now = performance.now();
            const dt = (now - prevTorsoTimeRef.current) / 1000;
            const { verdict, quality } = judgePosture(torso, prevTorsoRef.current, dt);

            torsoRef.current = torso;
            setCurrentVerdict(verdict);
            setCurrentQuality(quality);
            setCurrentTorso(torso);
            prevTorsoRef.current = torso;
            prevTorsoTimeRef.current = now;
          }
          break;
        }
        case "ERROR":
          console.error("[MuJoCo iframe ERROR]", event.data?.error);
          break;
      }
    };
    window.addEventListener("message", handleMessage);
    return () => {
      window.removeEventListener("message", handleMessage);
      registerIframeWindow(null);
    };
  }, [registerIframeWindow, resumeSimulation, sendMessage]);

  const setSensorConfig = useCallback((sensors: string[]) => {
    setActiveSensors(sensors);
    sendMessage({ type: "SET_SENSOR_CONFIG", payload: { sensors } });
  }, [sendMessage]);

  // ★ 리셋 후 즉시 STANDING_POSE로 PD 적용
  const resetSimulationRPC = useCallback(async () => {
    const rpc = rpcRef.current;
    if (rpc) {
      try {
        await rpc.request("RESET");
        await new Promise((r) => setTimeout(r, 500));
        await rpc.request("RESUME");
      } catch {
        sendMessage({ type: "RESET" });
        await new Promise((r) => setTimeout(r, 800));
        sendMessage({ type: "RESUME" });
      }
    } else {
      sendMessage({ type: "RESET" });
      await new Promise((r) => setTimeout(r, 800));
      sendMessage({ type: "RESUME" });
    }

    // ★ 리셋 직후 STANDING_POSE로 PD 잡아서 다리 벌어짐 방지
    sendMessage({
      type: "SET_JOINT_TARGETS_PD",
      enabled: true,
      targets: buildStandingTargets(),
      kp: 80,
      kd: 6,
    });

    await new Promise((r) => setTimeout(r, 500));
  }, [sendMessage]);

  function formatExtraSensors(sensors: Record<string, any>): string {
    const lines: string[] = [];
    for (const [key, val] of Object.entries(sensors)) {
      if (!val) continue;
      switch (key) {
        case "foot_contact": {
          const fc = val as any;
          if (fc.contacts) {
            const parts = Object.entries(fc.contacts).map(
              ([foot, c]: [string, any]) => `${foot}: contact=${c.contact}, force≈${(c.estimatedForce ?? 0).toFixed(1)}N`
            );
            lines.push(`[Foot Contact] ${parts.join(" | ")}`);
          }
          break;
        }
        case "joint_torque": {
          const jt = val as Record<string, number>;
          const top5 = Object.entries(jt)
            .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
            .slice(0, 5)
            .map(([name, torque]) => `${name}=${torque.toFixed(2)}`)
            .join(", ");
          lines.push(`[Joint Torque (top5)] ${top5}`);
          break;
        }
        case "center_of_mass": {
          const com = val as any;
          lines.push(`[CoM] x=${com.x?.toFixed(3)}, y=${com.y?.toFixed(3)}, z=${com.z?.toFixed(3)}`);
          break;
        }
        case "velocity": {
          const v = val as any;
          if (v.linear) lines.push(`[Velocity] lin=(${v.linear.x?.toFixed(3)}, ${v.linear.y?.toFixed(3)}, ${v.linear.z?.toFixed(3)})`);
          if (v.angular) lines.push(`[AngVel] (${v.angular.x?.toFixed(3)}, ${v.angular.y?.toFixed(3)}, ${v.angular.z?.toFixed(3)})`);
          break;
        }
        case "energy": {
          const e = val as any;
          lines.push(`[Energy] instantPower=${(e.instantPower ?? 0).toFixed(2)}W, cumulative=${(e.cumulativeEnergy ?? 0).toFixed(2)}J`);
          break;
        }
        case "momentum": {
          const m = val as any;
          if (m.linear) lines.push(`[Momentum] (${m.linear.x?.toFixed(3)}, ${m.linear.y?.toFixed(3)}, ${m.linear.z?.toFixed(3)})`);
          break;
        }
        case "stability": {
          const s = val as any;
          lines.push(`[Stability] supportPolygon=${s.supportPolygonArea?.toFixed(4)}, comInSupport=${s.comInsideSupport}`);
          break;
        }
        default: {
          lines.push(`[${key}] ${JSON.stringify(val).slice(0, 120)}`);
          break;
        }
      }
    }
    return lines.join("\n");
  }

  // ─── Motion Drive ─────────────────────────────────────────────

  // ★ 모션 실행 중 + 완료 후 모든 관절에 PD 타겟 보장
  const startMotionDrive = useCallback(
    (keyframes: VmdKeyframe[], kp = 80, kd = 6): Promise<void> => {
      return new Promise((resolve) => {
        if (driveIntervalRef.current) {
          clearInterval(driveIntervalRef.current);
          driveIntervalRef.current = null;
        }
        if (keyframes.length < 2) { resolve(); return; }

        sendMessage({ type: "RESUME" });
        const jointOrder = [...MUJOCO_JOINT_ORDER] as string[];
        const refMotion = new ReferenceMotion(jointOrder, keyframes, 30, false);
        const duration = refMotion.durationSec();
        const startWall = performance.now();
        setIsDriving(true);

        const checkStopped = setInterval(() => {
          if (!driveIntervalRef.current) { clearInterval(checkStopped); resolve(); }
        }, 100);

        driveIntervalRef.current = setInterval(() => {
          const elapsed = (performance.now() - startWall) / 1000;
          if (elapsed >= duration) {
            // ★ 모션 끝나면 STANDING_POSE로 유지 (다리 벌어짐 방지)
            sendMessage({
              type: "SET_JOINT_TARGETS_PD",
              enabled: true,
              targets: buildStandingTargets(),
              kp,
              kd,
            });

            if (driveIntervalRef.current) clearInterval(driveIntervalRef.current);
            driveIntervalRef.current = null;
            clearInterval(checkStopped);
            setIsDriving(false);
            resolve();
            return;
          }
          const { qRef } = refMotion.sample(elapsed);

          // ★ 모든 관절에 대해 타겟 설정 — 빠진 관절은 STANDING_POSE
          const targets: Record<string, number> = {};
          for (let i = 0; i < jointOrder.length; i++) {
            const name = jointOrder[i];
            const c = JOINT_CONSTRAINTS[name as JointName];
            let q = qRef[i] ?? STANDING_POSE[name as JointName] ?? 0;
            if (c) q = Math.max(c.min, Math.min(c.max, q));
            targets[name] = q;
          }
          sendMessage({ type: "SET_JOINT_TARGETS_PD", enabled: true, targets, kp, kd });
        }, 16);
      });
    },
    [sendMessage],
  );

  // ★ 정지 시에도 PD를 끄지 않고 STANDING_POSE로 유지
  const stopMotionDrive = useCallback(() => {
    if (driveIntervalRef.current) {
      clearInterval(driveIntervalRef.current);
      driveIntervalRef.current = null;
    }
    setIsDriving(false);

    // ★ PD를 끄지 않고 STANDING_POSE로 유지하여 다리 벌어짐 방지
    sendMessage({
      type: "SET_JOINT_TARGETS_PD",
      enabled: true,
      targets: buildStandingTargets(),
      kp: 80,
      kd: 6,
    });
  }, [sendMessage]);

  useEffect(() => {
    return () => { if (driveIntervalRef.current) clearInterval(driveIntervalRef.current); };
  }, []);

  // ═════════════════════════════════════════════════════════════
  // EuradBench 루프
  // ═════════════════════════════════════════════════════════════

  const startBench = useCallback(async () => {
    benchAbortRef.current = false;
    setBenchRunning(true);
    setBenchHistory([]);
    setBenchIter(0);
    setTier1Iter(null);
    setTier2Iter(null);
    setTier3Iter(null);
    setMaxStandDuration(0);
    setMaxWalkDistance(0);
    setAnalysisLog([]);
    standStartRef.current = null;
    walkOriginRef.current = null;
    setActiveSensors([]);

    const context = buildMujocoContext();
    const autonomyPrompt = buildAutonomyPrompt();
    const failureHistory: string[] = [];
    let currentStandStart: number | null = null;
    let localMaxStand = 0;
    let localMaxWalk = 0;
    let localTier1: number | null = null;
    let localTier2: number | null = null;
    let localTier3: number | null = null;
    let currentActiveSensors: string[] = [];

    await resetSimulationRPC();

    for (let iter = 0; iter < benchMaxIter; iter++) {
      if (benchAbortRef.current) break;

      setBenchIter(iter);
      setBenchStatus(`iter ${iter}: 모션 생성 중...`);

      const torso = torsoRef.current;
      const pose = jointsRef.current;
      const stateStr = MUJOCO_JOINT_ORDER.map(
        (n) => `${n}=${(pose[n]?.val ?? 0).toFixed(3)}`,
      ).join(", ");

      const torsoStr = torso
        ? `height=${torso.height.toFixed(3)}, pitch=${(torso.pitch * 180 / Math.PI).toFixed(1)}°, roll=${(torso.roll * 180 / Math.PI).toFixed(1)}°, x=${torso.x.toFixed(3)}, y=${torso.y.toFixed(3)}`
        : "no torso data";

      const extraSensorStr = currentActiveSensors.length > 0
        ? formatExtraSensors(extraSensorsRef.current)
        : "";

      const currentBenchHistory = benchHistoryRef.current;
      const prevIterTorso = currentBenchHistory.length > 0
        ? { height: currentBenchHistory[currentBenchHistory.length - 1].torsoHeight }
        : null;

      const deltaStr = prevIterTorso && torso
        ? `Delta from last iter: Δheight=${(torso.height - prevIterTorso.height).toFixed(4)}`
        : "";

      const motorReqBody = {
        intent: {
          goal: "walk forward stably",
          style: failureHistory.length > 5 ? "very_conservative" : "steady",
          duration_ms: 2000,
        },
        context,
        message: [
          `[EuradBench iteration ${iter}/${benchMaxIter}]`,
          `Task: walk forward without falling`,
          `Torso state: ${torsoStr}`,
          `Current joints: ${stateStr}`,
          deltaStr ? `\n${deltaStr}` : "",
          extraSensorStr ? `\n── Extra Sensor Data ──\n${extraSensorStr}` : "",
          failureHistory.length > 0
            ? `\nRecent history (last ${Math.min(5, failureHistory.length)}):\n${failureHistory.slice(-5).join("\n")}`
            : "No history yet (first attempt)",
          autonomyPrompt,
        ].filter(Boolean).join("\n"),
      };

      let llmResponse: LLMAutonomousResponse = {};

      try {
        const resp = await fetch("/api/motor", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(motorReqBody),
        });
        const data = await resp.json();
        llmResponse = {
          motions: data?.motions ?? [],
          analysis: data?.analysis ?? "",
          config: data?.config ?? null,
          observe: data?.observe ?? null,
          reset: data?.reset ?? false,
          hypothesis: data?.hypothesis ?? "",
          experiment: data?.experiment ?? "",
        };
      } catch (err) {
        console.error("[bench] motor call failed:", err);
        failureHistory.push(`iter${iter}: API_ERROR`);
        if (failureHistory.length > 20) failureHistory.shift();
        await new Promise((r) => setTimeout(r, 2000));
        continue;
      }

      if (benchAbortRef.current) break;

      const { motions, analysis, config, observe, reset: shouldReset, hypothesis, experiment } = llmResponse;

      if (shouldReset) {
        setBenchStatus(`iter ${iter}: 🔄 LLM이 리셋 요청`);
        await resetSimulationRPC();
        walkOriginRef.current = null;
        currentStandStart = null;
      }

      if (observe && observe.length > 0) {
        const validSensors = observe.filter((s) =>
          (AVAILABLE_SENSORS as readonly string[]).includes(s)
        );
        if (validSensors.length > 0) {
          currentActiveSensors = validSensors;
          setSensorConfig(validSensors);
          setBenchStatus(`iter ${iter}: 📡 센서 활성화: ${validSensors.join(", ")}`);
          await new Promise((r) => setTimeout(r, 200));
        }
      }

      if (!motions || motions.length === 0) {
        failureHistory.push(`iter${iter}: EMPTY_MOTIONS`);
        if (failureHistory.length > 20) failureHistory.shift();
        setBenchStatus(`iter ${iter}: 빈 응답, 재시도`);
        await new Promise((r) => setTimeout(r, 1000));
        continue;
      }

      if (analysis || hypothesis || experiment) {
        setAnalysisLog((prev) => [...prev.slice(-49), {
          iter,
          text: analysis ?? "",
          hypothesis: hypothesis ?? "",
          experiment: experiment ?? "",
        }]);
      }

      const useKp = config?.kp ?? 80;
      const useKd = config?.kd ?? 6;
      const clampedKp = Math.max(10, Math.min(200, useKp));
      const clampedKd = Math.max(1, Math.min(30, useKd));

      setBenchStatus(`iter ${iter}: ▶ 모션 실행 (${motions.length} cmds, kp=${clampedKp}, kd=${clampedKd})`);

      const keyframes = motorMotionsToKeyframes(motions, jointsRef.current);

      if (keyframes.length >= 2) {
        await startMotionDrive(keyframes, clampedKp, clampedKd);
        await new Promise((r) => setTimeout(r, 300));
      } else {
        await new Promise((r) => setTimeout(r, 1000));
      }

      if (benchAbortRef.current) break;

      let postTorso: TorsoState | null = null;
      let postJoints: Record<string, JointInfo> = {};
      const rpc = rpcRef.current;
      if (rpc) {
        try {
          const state = await rpc.request<{
            joints: Record<string, JointInfo>;
            torso: TorsoState | null;
            time: number;
            extraSensors: Record<string, any>;
          }>("GET_STATE");
          postTorso = state.torso;
          postJoints = state.joints;
          if (state.extraSensors) {
            extraSensorsRef.current = state.extraSensors;
          }
          if (postTorso) {
            torsoRef.current = postTorso;
            setCurrentTorso(postTorso);
          }
        } catch {
          postTorso = torsoRef.current;
        }
      } else {
        postTorso = torsoRef.current;
      }

      const { verdict, quality } = postTorso
        ? judgePosture(postTorso, prevTorsoRef.current, 0.5)
        : { verdict: "UNKNOWN" as PostureVerdict, quality: 0 };

      let standDuration = 0;
      if (verdict === "STANDING" || verdict === "WALKING") {
        if (currentStandStart === null) currentStandStart = performance.now();
        standDuration = (performance.now() - currentStandStart) / 1000;
        if (standDuration > localMaxStand) {
          localMaxStand = standDuration;
          setMaxStandDuration(localMaxStand);
        }
      } else {
        currentStandStart = null;
      }

      let walkDistance = 0;
      if (verdict === "WALKING" && postTorso) {
        if (!walkOriginRef.current) {
          walkOriginRef.current = { x: postTorso.x, y: postTorso.y };
        }
        const dx = postTorso.x - walkOriginRef.current.x;
        const dy = postTorso.y - walkOriginRef.current.y;
        walkDistance = Math.sqrt(dx * dx + dy * dy);
        if (walkDistance > localMaxWalk) {
          localMaxWalk = walkDistance;
          setMaxWalkDistance(localMaxWalk);
        }
      } else if (verdict === "FALLEN") {
        walkOriginRef.current = null;
      }

      const t1 = standDuration >= POSTURE.STAND_DURATION_TIER1;
      const t3 = walkDistance >= POSTURE.WALK_DISTANCE_TIER3;

      if (t1 && localTier1 === null) {
        localTier1 = iter;
        setTier1Iter(iter);
      }
      if (t3 && localTier3 === null) {
        localTier3 = iter;
        setTier3Iter(iter);
      }

      const autoScore = computeAutonomyScore(llmResponse, benchHistoryRef.current);

      const record: BenchIteration = {
        iter,
        timestamp: Date.now(),
        verdict,
        torsoHeight: postTorso?.height ?? 0,
        torsoPitch: postTorso?.pitch ?? 0,
        standDuration,
        walkDistance,
        quality,
        tier1Pass: t1,
        tier2Pass: false,
        tier3Pass: t3,
        analysisSnippet: (analysis ?? "").slice(0, 120),
        hypothesis: hypothesis ?? "",
        experiment: experiment ?? "",
        usedConfig: config ?? null,
        requestedSensors: currentActiveSensors,
        autonomyScore: autoScore,
        didReset: shouldReset ?? false,
      };
      setBenchHistory((prev) => [...prev, record]);

      const historyEntry = [
        `iter${iter}: ${verdict}`,
        `q=${quality}`,
        `h=${(postTorso?.height ?? 0).toFixed(3)}`,
        `stand=${standDuration.toFixed(1)}s`,
        `auto=${autoScore}`,
        hypothesis ? `hyp="${hypothesis.slice(0, 40)}"` : "",
        analysis ? `(${(analysis ?? "").slice(0, 60)})` : "",
      ].filter(Boolean).join(" ");
      failureHistory.push(historyEntry);
      if (failureHistory.length > 20) failureHistory.shift();

      if (verdict === "FALLEN" && !shouldReset) {
        setBenchStatus(`iter ${iter}: 넘어짐, 자동 리셋`);
        await resetSimulationRPC();
      }

      setBenchStatus(`iter ${iter}: ${verdictEmoji(verdict)} ${verdict} q=${quality} auto=${autoScore}`);
    }

    setBenchRunning(false);
    setBenchStatus("벤치마크 완료");
  }, [benchMaxIter, sendMessage, startMotionDrive, resetSimulationRPC, setSensorConfig]);

  const stopBench = useCallback(() => {
    benchAbortRef.current = true;
    stopMotionDrive();
    setBenchRunning(false);
    setBenchStatus("중단됨");
  }, [stopMotionDrive]);

  // ─── 차트 데이터 ──────────────────────────────────

  const chartData = useMemo(() => {
    const labels = benchHistory.map((h) => h.iter.toString());
    const qualityData = benchHistory.map((h) => h.quality);
    const heightData = benchHistory.map((h) => h.torsoHeight * 100);
    const autonomyData = benchHistory.map((h) => h.autonomyScore);

    return {
      labels,
      datasets: [
        {
          label: "Quality Score",
          data: qualityData,
          borderColor: "#00d2ff",
          backgroundColor: "rgba(0, 210, 255, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          pointBackgroundColor: benchHistory.map((h) => verdictColor(h.verdict)),
          yAxisID: "y",
        },
        {
          label: "Autonomy Score",
          data: autonomyData,
          borderColor: "#a855f7",
          backgroundColor: "rgba(168, 85, 247, 0.1)",
          borderWidth: 2,
          fill: false,
          tension: 0.3,
          pointRadius: 2,
          yAxisID: "y",
        },
        {
          label: "Torso Height (cm)",
          data: heightData,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.05)",
          borderWidth: 1.5,
          fill: false,
          tension: 0.3,
          pointRadius: 2,
          borderDash: [4, 2],
          yAxisID: "y1",
        },
      ],
    };
  }, [benchHistory]);

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 } as const,
      interaction: { intersect: false, mode: "index" as const },
      scales: {
        y: {
          min: 0,
          max: 100,
          position: "left" as const,
          ticks: { color: "#888", callback: (v: any) => v + "%" },
          grid: { color: "rgba(255,255,255,0.05)" },
          title: { display: true, text: "Score", color: "#888", font: { size: 10 } },
        },
        y1: {
          min: 0,
          max: 80,
          position: "right" as const,
          ticks: { color: "#22c55e", callback: (v: any) => v + "cm" },
          grid: { display: false },
          title: { display: true, text: "Height", color: "#22c55e", font: { size: 10 } },
        },
        x: {
          ticks: { color: "#888", maxTicksLimit: 20 },
          grid: { color: "rgba(255,255,255,0.03)" },
          title: { display: true, text: "Iteration", color: "#888", font: { size: 10 } },
        },
      },
      plugins: {
        legend: { labels: { color: "#ccc", font: { size: 10 } } },
        tooltip: {
          callbacks: {
            afterBody: (ctx: any) => {
              const idx = ctx[0]?.dataIndex;
              if (idx != null && benchHistory[idx]) {
                const h = benchHistory[idx];
                return [
                  `Verdict: ${h.verdict}`,
                  `Stand: ${h.standDuration.toFixed(1)}s`,
                  `Walk: ${h.walkDistance.toFixed(2)}m`,
                  `Autonomy: ${h.autonomyScore}`,
                  h.hypothesis ? `Hypothesis: ${h.hypothesis.slice(0, 50)}` : "",
                  h.usedConfig ? `Config: kp=${h.usedConfig.kp ?? 80}, kd=${h.usedConfig.kd ?? 6}` : "",
                  h.analysisSnippet ? `AI: ${h.analysisSnippet.slice(0, 60)}...` : "",
                ].filter(Boolean);
              }
              return [];
            },
          },
        },
      },
    }),
    [benchHistory],
  );

  const onPlay = () => { resumeSimulation(); setSimState("RUNNING"); };
  const onPause = () => { pauseSimulation(); setSimState("PAUSED"); };
  const onReset = () => {
    resetPose();
    setSimState("PAUSED");
    setSimTime(0);
    // ★ 수동 리셋 시에도 STANDING_POSE PD 적용
    setTimeout(() => {
      sendMessage({
        type: "SET_JOINT_TARGETS_PD",
        enabled: true,
        targets: buildStandingTargets(),
        kp: 80,
        kd: 6,
      });
    }, 300);
  };

  const highestTier = tier3Iter !== null ? "Tier 3 🏃 Walk" : tier1Iter !== null ? "Tier 1 🟢 Stand" : "—";

  const avgQuality = benchHistory.length > 0
    ? benchHistory.reduce((s, h) => s + h.quality, 0) / benchHistory.length
    : 0;
  const avgAutonomy = benchHistory.length > 0
    ? benchHistory.reduce((s, h) => s + h.autonomyScore, 0) / benchHistory.length
    : 0;

  // ═════════════════════════════════════════════════════════════
  // RENDER
  // ═════════════════════════════════════════════════════════════

  return (
    <div className="w-full h-full flex flex-row bg-[#1a1a2e] text-[#e8e8e8]">
      {/* ── 뷰포트 ────────────────────────────────────────────── */}
      <div className="flex-grow relative border-r border-[#0f3460] min-h-0">
        <iframe
          ref={iframeRef}
          src="/mujoco/mujoco.html"
          onLoad={handleIframeLoad}
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-downloads"
          className="w-full h-full border-none block bg-[#0a0a1a]"
          title="MuJoCo Viewer"
        />

        {/* HUD */}
        <div className="absolute top-4 left-4 flex gap-4 text-xs font-mono bg-black/70 p-3 rounded-lg border border-[#0f3460]">
          <div className="flex items-center gap-2">
            <Activity size={14} className={simState === "RUNNING" ? "text-green-500" : "text-yellow-500"} />
            <span>{simState}</span>
          </div>
          <div>TIME: {simTime.toFixed(2)}s</div>
          <div>FPS: {fps}</div>
          {benchRunning && (
            <div className="font-bold text-[#00d2ff] flex items-center gap-1">
              <BarChart3 size={14} /> BENCH iter {benchIter}
            </div>
          )}
          {activeSensors.length > 0 && (
            <div className="text-[#a855f7] flex items-center gap-1">
              <Eye size={12} /> {activeSensors.length} sensors
            </div>
          )}
        </div>

        {/* 실시간 자세 판단 오버레이 */}
        {currentTorso && (
          <div
            className="absolute bottom-4 left-4 px-4 py-2 rounded-lg font-mono text-sm border"
            style={{
              backgroundColor: `${verdictColor(currentVerdict)}15`,
              borderColor: `${verdictColor(currentVerdict)}40`,
              color: verdictColor(currentVerdict),
            }}
          >
            <span className="text-lg mr-2">{verdictEmoji(currentVerdict)}</span>
            <span className="font-bold">{currentVerdict}</span>
            <span className="ml-3 opacity-70 text-xs">
              h={currentTorso.height.toFixed(3)}m
              p={(currentTorso.pitch * 180 / Math.PI).toFixed(1)}°
              q={currentQuality}
            </span>
          </div>
        )}
      </div>

      {/* ── 사이드패널 ────────────────────────────────────────── */}
      <div className="w-[380px] flex flex-col bg-[#16213e] min-h-0">
        {/* 헤더 */}
        <div className="h-12 bg-[#0f3460] flex items-center justify-center border-b border-[#1a1a4e]">
          <span className="font-bold text-sm flex items-center gap-2 text-[#00d2ff]">
            <Trophy size={16} /> EuradBench
          </span>
        </div>

        <div className="flex-1 overflow-y-auto">
          {/* ── 시뮬레이션 제어 ──────────────────────────────── */}
          <div className="p-4 border-b border-[#0f3460] space-y-3">
            <div className="grid grid-cols-3 gap-2">
              <button onClick={onPlay}
                className={`p-2 rounded flex items-center justify-center gap-1 text-xs font-bold transition-all ${
                  simState === "RUNNING" ? "bg-[#22c55e] text-white" : "bg-[#1a1a2e] hover:bg-[#1e2d50] border border-[#0f3460]"
                }`}>
                <Play size={13} /> PLAY
              </button>
              <button onClick={onPause}
                className={`p-2 rounded flex items-center justify-center gap-1 text-xs font-bold transition-all ${
                  simState === "PAUSED" ? "bg-[#f97316] text-white" : "bg-[#1a1a2e] hover:bg-[#1e2d50] border border-[#0f3460]"
                }`}>
                <Pause size={13} /> PAUSE
              </button>
              <button onClick={onReset}
                className="p-2 rounded flex items-center justify-center gap-1 text-xs font-bold bg-[#1a1a2e] hover:bg-[#ef4444] border border-[#0f3460] transition-all">
                <RotateCcw size={13} /> RESET
              </button>
            </div>
          </div>

          {/* ── 벤치마크 설정 ────────────────────────────────── */}
          <div className="p-4 border-b border-[#0f3460] space-y-3">
            <div className="text-xs font-mono text-[#00d2ff] flex items-center gap-2">
              <Settings2 size={14} /> Benchmark Settings
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="text-[10px] text-[#888] mb-1">Model Name</div>
                <input
                  type="text"
                  value={benchModelName}
                  onChange={(e) => setBenchModelName(e.target.value)}
                  disabled={benchRunning}
                  className="w-full bg-[#1a1a2e] border border-[#0f3460] rounded p-1.5 text-xs text-[#e8e8e8] disabled:opacity-40"
                />
              </div>
              <div>
                <div className="text-[10px] text-[#888] mb-1">Max Iterations</div>
                <input
                  type="number"
                  value={benchMaxIter}
                  onChange={(e) => setBenchMaxIter(Number(e.target.value))}
                  disabled={benchRunning}
                  className="w-full bg-[#1a1a2e] border border-[#0f3460] rounded p-1.5 text-xs text-[#e8e8e8] disabled:opacity-40"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={startBench}
                disabled={benchRunning}
                className="p-2.5 rounded text-xs font-bold bg-[#00d2ff] hover:bg-[#00b8e6] text-[#1a1a2e] transition-all disabled:opacity-40"
              >
                <Zap size={14} className="inline mr-1" />
                START BENCH
              </button>
              <button
                onClick={stopBench}
                disabled={!benchRunning}
                className="p-2.5 rounded text-xs font-bold bg-[#ef4444] hover:bg-[#dc2626] text-white transition-all disabled:opacity-40"
              >
                <Square size={14} className="inline mr-1" />
                STOP
              </button>
            </div>

            {benchStatus && (
              <div className="text-[10px] font-mono text-[#00d2ff] bg-[#1a1a2e] rounded p-2 border border-[#0f3460]">
                {benchStatus}
              </div>
            )}
          </div>

          {/* ── 스코어 카드 ──────────────────────────────────── */}
          <div className="p-4 border-b border-[#0f3460] space-y-3">
            <div className="text-xs font-mono text-[#00d2ff] flex items-center gap-2">
              <Trophy size={14} /> Score Card — {benchModelName}
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460] text-center">
                <div className="text-2xl font-bold text-[#00d2ff]">{avgQuality.toFixed(1)}</div>
                <div className="text-[10px] text-[#888]">Avg Quality</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460] text-center">
                <div className="text-2xl font-bold text-[#a855f7]">{avgAutonomy.toFixed(1)}</div>
                <div className="text-[10px] text-[#888]">Avg Autonomy</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460] text-center">
                <div className="text-2xl font-bold text-[#22c55e]">{maxStandDuration.toFixed(1)}s</div>
                <div className="text-[10px] text-[#888]">Max Stand</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460] text-center">
                <div className="text-2xl font-bold text-[#3b82f6]">{maxWalkDistance.toFixed(2)}m</div>
                <div className="text-[10px] text-[#888]">Max Walk</div>
              </div>
              <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460] text-center col-span-2">
                <div className="text-sm font-bold text-[#eab308]">{highestTier}</div>
                <div className="text-[10px] text-[#888]">Highest Tier</div>
              </div>
            </div>

            {/* Tier 달성 표 */}
            <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460] space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2">
                  {tier1Iter !== null ? <CheckCircle size={12} className="text-[#22c55e]" /> : <AlertTriangle size={12} className="text-[#888]" />}
                  Tier 1 — Stand 3s+
                </span>
                <span className={`font-mono font-bold ${tier1Iter !== null ? "text-[#22c55e]" : "text-[#888]"}`}>
                  {tier1Iter !== null ? `iter ${tier1Iter}` : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2">
                  {tier2Iter !== null ? <CheckCircle size={12} className="text-[#22c55e]" /> : <AlertTriangle size={12} className="text-[#888]" />}
                  Tier 2 — Balance (push)
                </span>
                <span className="font-mono font-bold text-[#888]">—</span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2">
                  {tier3Iter !== null ? <CheckCircle size={12} className="text-[#3b82f6]" /> : <AlertTriangle size={12} className="text-[#888]" />}
                  Tier 3 — Walk 1m+
                </span>
                <span className={`font-mono font-bold ${tier3Iter !== null ? "text-[#3b82f6]" : "text-[#888]"}`}>
                  {tier3Iter !== null ? `iter ${tier3Iter}` : "—"}
                </span>
              </div>
            </div>
          </div>

          {/* ── 실시간 그래프 ────────────────────────────────── */}
          <div className="p-4 border-b border-[#0f3460]">
            <div className="text-xs font-mono text-[#00d2ff] flex items-center gap-2 mb-3">
              <TrendingUp size={14} /> Real-time Score Graph
            </div>
            <div className="bg-[#1a1a2e] rounded-lg p-3 border border-[#0f3460]" style={{ height: 220 }}>
              {benchHistory.length > 0 ? (
                <Line data={chartData} options={chartOptions} />
              ) : (
                <div className="flex items-center justify-center h-full text-[#888] text-xs">
                  벤치마크를 시작하면 그래프가 표시됩니다
                </div>
              )}
            </div>
          </div>

          {/* ── LLM 분석 로그 ────────────────────────────────── */}
          <div className="p-4 space-y-2">
            <div className="text-xs font-mono text-[#00d2ff] flex items-center gap-2">
              <Brain size={14} /> LLM Analysis Log
            </div>
            <div className="max-h-60 overflow-y-auto space-y-2">
              {analysisLog.length === 0 ? (
                <div className="text-[10px] text-[#888] text-center py-4">
                  벤치마크 실행 시 LLM의 사고 과정이 여기에 표시됩니다
                </div>
              ) : (
                analysisLog.slice().reverse().map((entry, i) => (
                  <div
                    key={`${entry.iter}-${i}`}
                    className="bg-[#1a1a2e] rounded p-2 border border-[#0f3460]"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] font-mono font-bold text-[#00d2ff]">
                        iter {entry.iter}
                      </span>
                      {benchHistory[entry.iter] && (
                        <>
                          <span style={{ color: verdictColor(benchHistory[entry.iter].verdict) }}>
                            {verdictEmoji(benchHistory[entry.iter].verdict)}
                          </span>
                          <span className="text-[9px] font-mono text-[#a855f7]">
                            auto:{benchHistory[entry.iter].autonomyScore}
                          </span>
                        </>
                      )}
                    </div>
                    {entry.experiment && (
                      <div className="text-[10px] text-[#f97316] mb-1 flex items-center gap-1">
                        <FlaskConical size={10} /> {entry.experiment}
                      </div>
                    )}
                    {entry.hypothesis && (
                      <div className="text-[10px] text-[#eab308] mb-1 flex items-center gap-1">
                        <Zap size={10} /> 가설: {entry.hypothesis}
                      </div>
                    )}
                    <p className="text-[11px] text-[#ccc] leading-relaxed">
                      {entry.text}
                    </p>
                    {benchHistory[entry.iter]?.usedConfig && (
                      <div className="text-[9px] text-[#888] mt-1 flex items-center gap-1">
                        <Sliders size={9} />
                        kp={benchHistory[entry.iter].usedConfig?.kp ?? 80}
                        kd={benchHistory[entry.iter].usedConfig?.kd ?? 6}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
