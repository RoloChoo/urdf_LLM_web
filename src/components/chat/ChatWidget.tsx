// ChatWidget.tsx

"use client";

import { FormEvent, useEffect, useRef, useState, useCallback } from "react";
import {
  ChevronDown,
  Loader2,
  Mic,
  Plus,
  Sparkles,
  RefreshCw,
} from "lucide-react";
import { ReferenceMotion } from "@/utils/ReferenceMotion";
import type { VmdKeyframe } from "@/utils/VmdLoader";
import {
  MUJOCO_JOINT_ORDER,
  JOINT_CONSTRAINTS,
  applyJointConstraints,
} from "@/constants/jointConfig";
import type { JointName } from "@/constants/jointConfig";

// ─── 타입 ────────────────────────────────────────────────────

type ConversationMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type IntentResponse = {
  text: string;
  intent: {
    goal: string;
    style?: string;
    duration_ms?: number;
    sketch?: Array<{
      joint_hint: string;
      delta_rad?: number;
      target_angle_rad?: number;
    }>;
    constraints?: Record<string, unknown>;
  };
  error?: string;
};

type MotorResponse = {
  motions: Array<{
    joint: string;
    angle: number;
    time: number;
    speed?: number;
  }>;
  analysis?: string | null;
  error?: string;
};

type ExecuteResponse = {
  ok?: boolean;
  motions?: Array<{
    joint: string;
    angle: number;
    time?: number;
    speed?: number;
  }>;
  warnings?: string[];
  error?: string;
};

type RLTrainingSummary = {
  when: string;
  policyLabel: string;
  episodes: number;
  bestReturn: number;
  lastReturn: number | null;
  actionDim: number;
  obsDim: number;
  frameSkip: number;
  maxSteps: number;
  baseBodyId?: number | null;
  targetHeight?: number | null;
  termReasonCounts?: Record<string, number>;
};

type TaskPlannerResponse = {
  text: string;
  plan: {
    goal: string;
    rlConfig?: {
      reward?: Record<string, number>;
      terminate?: Record<string, number>;
      frameSkip?: number;
      maxSteps?: number;
      actionMode?: "normalized" | "direct";
    };
    actionHint?: {
      kind: "sine";
      amp?: number;
      speed?: number;
      bias?: number[];
      scale?: number[];
      phase?: number[];
    };
  };
  error?: string;
};

type JointInfo = { val: number; min: number; max: number };

// ─── 상수 ────────────────────────────────────────────────────

const EMPTY_HINT = "LLM 답변은 이 자리에서 바로 확인할 수 있어요.";

// ─── MuJoCo 관절 컨텍스트 빌더 ──────────────────────────────

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
    "- Use ONLY joints from the list above (exact names).",
    "- Angles are in radians.",
    "- LEFT leg joints are sign-mirrored vs RIGHT:",
    "  l_knee <= 0 (bend = negative), r_knee >= 0 (bend = positive)",
    "  l_hip_roll <= 0, r_hip_roll >= 0",
    "  l_hip_pitch: positive = forward swing, r_hip_pitch: negative = forward swing",
    "- l_el <= 0, r_el >= 0.",
    "- Keep hip_roll close to 0 (±0.05) for standing balance.",
    "- time is milliseconds.",
    "- For walking: opposite legs anti-phase, arms swing opposite to same-side leg.",
    "- PRIORITY: stay standing first. Only attempt walking after stable stance.",
  ].join("\n");
}

// ─── motor motions → VmdKeyframe[] 변환 ──────────────────────

function motorMotionsToKeyframes(
  motions: Array<{ joint: string; angle: number; time: number }>,
  currentJoints?: Record<string, JointInfo> | null,
): VmdKeyframe[] {
  const timeMap = new Map<number, Record<string, number>>();

  for (const m of motions) {
    const resolved = resolveToMujocoJoint(m.joint);
    if (!resolved) continue;

    const timeMs = m.time ?? 0;
    if (!timeMap.has(timeMs)) {
      timeMap.set(timeMs, {});
    }
    timeMap.get(timeMs)![resolved] = m.angle;
  }

  const sortedTimes = [...timeMap.keys()].sort((a, b) => a - b);

  // 현재 관절 상태가 있으면 사용, 없으면 전부 0 (T-pose 세팅은 여기서 하지 않음)
  const currentPose: Record<string, number> = {};
  if (currentJoints) {
    for (const name of MUJOCO_JOINT_ORDER) {
      currentPose[name] = currentJoints[name]?.val ?? 0;
    }
  } else {
    for (const name of MUJOCO_JOINT_ORDER) {
      currentPose[name] = 0;
    }
  }

  const keyframes: VmdKeyframe[] = [];

  for (const timeMs of sortedTimes) {
    const updates = timeMap.get(timeMs)!;
    Object.assign(currentPose, updates);
    const constrained = applyJointConstraints({ ...currentPose });
    const timeSec = timeMs / 1000;
    const frame = Math.round(timeSec * 30);
    keyframes.push({
      frame,
      timeSec,
      pose: { ...constrained } as Record<string, number>,
    });
  }

  return keyframes;
}

function resolveToMujocoJoint(name: string): JointName | null {
  if ((MUJOCO_JOINT_ORDER as readonly string[]).includes(name)) {
    return name as JointName;
  }
  const norm = name.toLowerCase().replace(/[\s\-_.]/g, "");
  for (const jn of MUJOCO_JOINT_ORDER) {
    const jnNorm = jn.toLowerCase().replace(/[\s\-_.]/g, "");
    if (jnNorm === norm) return jn;
  }
  for (const jn of MUJOCO_JOINT_ORDER) {
    const jnNorm = jn.toLowerCase().replace(/[\s\-_.]/g, "");
    if (jnNorm.includes(norm) || norm.includes(jnNorm)) return jn;
  }
  return null;
}

// ─── ai:stage 이벤트 ─────────────────────────────────────────

type Stage = "intent" | "motor" | "execute" | "task_planner";
type Phase = "start" | "end";

function estTok(chars: number) {
  if (!Number.isFinite(chars) || chars <= 0) return 0;
  return Math.max(1, Math.round(chars / 4));
}

function emitAiStage(detail: {
  stage: Stage;
  phase: Phase;
  ok?: boolean;
  ms?: number;
  inChars?: number;
  outChars?: number;
  inTok?: number;
  outTok?: number;
}) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent("ai:stage", { detail }));
}

// ═══════════════════════════════════════════════════════════════
// 메인 컴포넌트
// ═══════════════════════════════════════════════════════════════

export default function ChatWidget() {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [plannerLoading, setPlannerLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // drive 상태
  const [isDriving, setIsDriving] = useState(false);
  const driveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ─── GamingAgent 자동 루프 상태 ────────────────────────────
  const autoLoopRef = useRef(false);
  const [autoLoopRunning, setAutoLoopRunning] = useState(false);
  const [autoLoopIter, setAutoLoopIter] = useState(0);
  const [autoLoopStatus, setAutoLoopStatus] = useState("");

  const messagesRef = useRef<ConversationMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // ─── 현재 관절 상태 저장 ───────────────────────────────────

  const currentPoseRef = useRef<Record<string, number> | null>(null);
  const jointsRef = useRef<Record<string, JointInfo> | null>(null);

  useEffect(() => {
    const handler = (ev: MessageEvent) => {
      const d = ev.data;
      if (!d || typeof d !== "object") return;
      if (d.type === "SYNC_STATE" && d.payload?.joints) {
        const joints = d.payload.joints as Record<string, JointInfo>;
        jointsRef.current = joints;
        const pose: Record<string, number> = {};
        for (const [name, info] of Object.entries(joints)) {
          pose[name] = info.val;
        }
        currentPoseRef.current = pose;
      }
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, []);

  const getCurrentPose = useCallback((): Record<string, number> | null => {
    return currentPoseRef.current;
  }, []);

  // ─── MuJoCo iframe 통신 ────────────────────────────────────

  const sendToMujocoIframe = useCallback((msg: unknown) => {
    window.dispatchEvent(
      new CustomEvent("mujoco:postMessage", { detail: msg }),
    );
  }, []);

  // ─── ReferenceMotion drive ─────────────────────────────────

  const startMotionDrive = useCallback(
    (keyframes: VmdKeyframe[], kp = 80, kd = 6): Promise<void> => {
      return new Promise((resolve) => {
        if (driveIntervalRef.current) {
          clearInterval(driveIntervalRef.current);
          driveIntervalRef.current = null;
        }

        if (keyframes.length < 2) {
          resolve();
          return;
        }

        sendToMujocoIframe({ type: "RESUME" });

        const jointOrder = [...MUJOCO_JOINT_ORDER] as string[];
        const refMotion = new ReferenceMotion(
          jointOrder,
          keyframes,
          30,
          false,
        );
        const duration = refMotion.durationSec();
        const startWall = performance.now();
        setIsDriving(true);

        driveIntervalRef.current = setInterval(() => {
          const elapsed = (performance.now() - startWall) / 1000;

          if (elapsed >= duration) {
            if (driveIntervalRef.current) {
              clearInterval(driveIntervalRef.current);
              driveIntervalRef.current = null;
            }
            sendToMujocoIframe({
              type: "SET_JOINT_TARGETS_PD",
              enabled: false,
            });
            setIsDriving(false);
            resolve();
            return;
          }

          const { qRef } = refMotion.sample(elapsed);
          const targets: Record<string, number> = {};
          for (let i = 0; i < jointOrder.length; i++) {
            const name = jointOrder[i];
            const c = JOINT_CONSTRAINTS[name as JointName];
            let q = qRef[i] ?? 0;
            if (c) q = Math.max(c.min, Math.min(c.max, q));
            targets[name] = q;
          }

          sendToMujocoIframe({
            type: "SET_JOINT_TARGETS_PD",
            enabled: true,
            targets,
            kp,
            kd,
          });
        }, 16);
      });
    },
    [sendToMujocoIframe],
  );

  const stopMotionDrive = useCallback(() => {
    if (driveIntervalRef.current) {
      clearInterval(driveIntervalRef.current);
      driveIntervalRef.current = null;
    }
    setIsDriving(false);
    sendToMujocoIframe({
      type: "SET_JOINT_TARGETS_PD",
      enabled: false,
    });
  }, [sendToMujocoIframe]);

  useEffect(() => {
    return () => {
      if (driveIntervalRef.current) clearInterval(driveIntervalRef.current);
    };
  }, []);

  // ─── T-pose 안정화 헬퍼 (리셋/스타팅 시에만 사용) ─────────

  const stabilizeAfterReset = useCallback(
    async (waitMs = 2000) => {
      sendToMujocoIframe({ type: "RESET" });
      sendToMujocoIframe({ type: "RESUME" });

      const tposeTargets: Record<string, number> = {};
      for (const name of MUJOCO_JOINT_ORDER) tposeTargets[name] = 0;
      sendToMujocoIframe({
        type: "SET_JOINT_TARGETS_PD",
        enabled: true,
        targets: tposeTargets,
        kp: 80,
        kd: 8,
      });
      await new Promise((r) => setTimeout(r, waitMs));
      sendToMujocoIframe({
        type: "SET_JOINT_TARGETS_PD",
        enabled: false,
      });
      await new Promise((r) => setTimeout(r, 300));
    },
    [sendToMujocoIframe],
  );

  // ═══════════════════════════════════════════════════════════
  // GamingAgent 스타일 자동 루프 (비전 통합)
  // ═══════════════════════════════════════════════════════════

  const startAutoLoop = useCallback(
    async (task: string = "걷기 학습") => {
      autoLoopRef.current = true;
      setAutoLoopRunning(true);
      setAutoLoopIter(0);
      setAutoLoopStatus("시작...");

      let iteration = 0;
      const failureHistory: string[] = [];
      const context = buildMujocoContext();

      function captureFrame(): string | null {
        try {
          const iframe = document.querySelector(
            'iframe[title="MuJoCo Viewer"]',
          ) as HTMLIFrameElement | null;
          if (!iframe) return null;

          let sourceCanvas: HTMLCanvasElement | null = null;
          try {
            sourceCanvas =
              iframe.contentDocument?.querySelector("canvas") ?? null;
          } catch {
            return null;
          }
          if (!sourceCanvas) return null;

          const tmpCanvas = document.createElement("canvas");
          const scale = Math.min(
            1,
            384 / Math.max(sourceCanvas.width, sourceCanvas.height),
          );
          tmpCanvas.width = Math.round(sourceCanvas.width * scale);
          tmpCanvas.height = Math.round(sourceCanvas.height * scale);
          const ctx = tmpCanvas.getContext("2d");
          if (!ctx) return null;

          ctx.drawImage(
            sourceCanvas,
            0,
            0,
            tmpCanvas.width,
            tmpCanvas.height,
          );
          return tmpCanvas.toDataURL("image/png", 0.7);
        } catch {
          return null;
        }
      }

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `🔄 비전 기반 GamingAgent 루프를 시작합니다.\n목표: ${task}\n📷 매 사이클 화면을 캡처해서 VLM이 평가합니다.`,
        },
      ]);

      while (autoLoopRef.current) {
        try {
          const currentPose = getCurrentPose();
          const stateStr = currentPose
            ? MUJOCO_JOINT_ORDER.map(
                (name) =>
                  `${name}=${(currentPose[name] ?? 0).toFixed(3)}`,
              ).join(", ")
            : "no state";

          setAutoLoopIter(iteration);

          const frame = captureFrame();
          let visionEval = {
            posture: "unknown",
            movement: "unknown",
            quality: 5,
            fallen: false,
            suggestion: "continue with conservative motion",
          };

          if (frame) {
            setAutoLoopStatus(`iter ${iteration}: 📷 비전 평가 중...`);

            try {
              const visionResp = await fetch("/api/vision-eval", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  image: frame,
                  jointState: stateStr,
                  iteration,
                  task,
                }),
              });

              if (visionResp.ok) {
                visionEval = await visionResp.json();
              }
            } catch (err) {
              console.warn("[auto-loop] vision eval failed:", err);
            }
          } else {
            setAutoLoopStatus(
              `iter ${iteration}: 📷 캡처 불가, 상태값만 사용`,
            );
          }

          if (visionEval.fallen) {
            failureHistory.push(
              `iter${iteration}: FALLEN (quality=${visionEval.quality}, posture=${visionEval.posture})`,
            );
            if (failureHistory.length > 20) failureHistory.shift();

            if (iteration % 3 === 0) {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant",
                  content: `⚠️ iter ${iteration}: 넘어짐 감지! (${visionEval.posture}) 리셋 후 안정화...`,
                },
              ]);
            }

            await stabilizeAfterReset(2000);
            iteration++;
            continue;
          }

          setAutoLoopStatus(
            `iter ${iteration}: 🧠 모션 생성 중 (quality=${visionEval.quality})...`,
          );

          const motorReqBody = {
            intent: {
              goal: `continue ${task}`,
              style:
                visionEval.quality < 3 ? "very_conservative" : "steady",
              duration_ms: 2000,
            },
            context,
            message: [
              `[Vision-guided auto-loop iteration ${iteration}]`,
              `Task: ${task}`,
              `Vision evaluation: posture=${visionEval.posture}, movement=${visionEval.movement}, quality=${visionEval.quality}/10`,
              `VLM suggestion: ${visionEval.suggestion}`,
              `Current joints: ${stateStr}`,
              failureHistory.length > 0
                ? `Recent failures: ${failureHistory.slice(-3).join("; ")}`
                : "",
              visionEval.quality < 3
                ? "PRIORITY: Very poor state. Use small, safe movements only."
                : "Generate next 1-2 second walking segment.",
              `Spread motions across time (0, 200, 400... ms). All 20 joints.`,
            ]
              .filter(Boolean)
              .join("\n"),
          };

          const motorResp = await fetch("/api/motor", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(motorReqBody),
          });

          if (!autoLoopRef.current) break;

          const motorData = (await motorResp.json()) as MotorResponse;

          if (!motorResp.ok || !motorData?.motions?.length) {
            failureHistory.push(`iter${iteration}: motor failed`);
            if (failureHistory.length > 20) failureHistory.shift();
            setAutoLoopStatus(
              `iter ${iteration}: motor 실패, 2초 후 재시도`,
            );
            await new Promise((r) => setTimeout(r, 2000));
            iteration++;
            continue;
          }

          if (motorData.analysis) {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: `🧠 **iter ${iteration}** — ${motorData.analysis}`,
              },
            ]);
          }

          const keyframes = motorMotionsToKeyframes(
            motorData.motions,
            jointsRef.current,
          );

          if (keyframes.length >= 2) {
            setAutoLoopStatus(
              `iter ${iteration}: ▶ ${motorData.motions.length} motions (q=${visionEval.quality})`,
            );
            await startMotionDrive(keyframes);
            await new Promise((r) => setTimeout(r, 300));
          } else {
            await new Promise((r) => setTimeout(r, 1000));
          }

          if (iteration % 5 === 0 && iteration > 0) {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: `🔄 iter ${iteration} | quality: ${visionEval.quality}/10 | ${visionEval.posture} / ${visionEval.movement} | 실패: ${failureHistory.length}회`,
              },
            ]);
          }

          iteration++;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.error("[auto-loop]", err);
          failureHistory.push(`iter${iteration}: error (${msg})`);
          if (failureHistory.length > 20) failureHistory.shift();
          setAutoLoopStatus(`iter ${iteration}: 에러, 2초 후 재시도`);
          await new Promise((r) => setTimeout(r, 2000));
          iteration++;
        }
      }

      setAutoLoopRunning(false);
      setAutoLoopStatus("정지됨");
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `⏹ 비전 루프 종료 (${iteration}회 실행)`,
        },
      ]);
    },
    [getCurrentPose, startMotionDrive, sendToMujocoIframe, stabilizeAfterReset],
  );

  const stopAutoLoop = useCallback(() => {
    autoLoopRef.current = false;
    stopMotionDrive();
    setAutoLoopStatus("정지 중...");
  }, [stopMotionDrive]);

  // ═══════════════════════════════════════════════════════════
  // RL training end → planner
  // ═══════════════════════════════════════════════════════════

  useEffect(() => {
    const onTrainingSummary = (ev: Event) => {
      const summary = (ev as CustomEvent<RLTrainingSummary>).detail;
      void runTaskPlanner(summary);
    };
    window.addEventListener("rl:trainingSummary", onTrainingSummary as any);
    return () =>
      window.removeEventListener(
        "rl:trainingSummary",
        onTrainingSummary as any,
      );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runTaskPlanner(summary: RLTrainingSummary) {
    try {
      setPlannerLoading(true);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `학습 종료 요약 수신 (best=${summary.bestReturn.toFixed(2)}, eps=${summary.episodes}). 다음 목표를 계획중…`,
        },
      ]);

      const context = buildMujocoContext();
      const historyPayload = messagesRef.current.map(({ role, content }) => ({
        role,
        content,
      }));
      const reqBody = { summary, history: historyPayload, context };
      const inChars = JSON.stringify(reqBody).length;
      emitAiStage({
        stage: "task_planner",
        phase: "start",
        inChars,
        inTok: estTok(inChars),
      });

      const t0 = performance.now();
      const resp = await fetch("/api/task-planner", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqBody),
      });
      const data = (await resp.json()) as TaskPlannerResponse;
      const ms = Math.round(performance.now() - t0);
      const outChars = JSON.stringify(data).length;
      emitAiStage({
        stage: "task_planner",
        phase: "end",
        ok: resp.ok,
        ms,
        inChars,
        outChars,
        inTok: estTok(inChars),
        outTok: estTok(outChars),
      });

      if (!resp.ok) throw new Error(data?.error || "task-planner 실패");
      if (data?.text?.trim()) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: data.text.trim(),
          },
        ]);
      }
      if (data?.plan) {
        window.dispatchEvent(
          new CustomEvent("rl:applyPlan", { detail: data.plan }),
        );
      }
    } catch (e) {
      console.error("[task-planner] error:", e);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `task-planner 오류: ${e instanceof Error ? e.message : String(e)}`,
        },
      ]);
    } finally {
      setPlannerLoading(false);
    }
  }

  // ═══════════════════════════════════════════════════════════
  // chat submit: intent → motor → execute
  // ═══════════════════════════════════════════════════════════

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) return;

    const userMessage: ConversationMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    const historyPayload = messages.map(({ role, content }) => ({
      role,
      content,
    }));

    setMessages((prev) => [...prev, userMessage]);
    setMessage("");
    setIsLoading(true);
    setError(null);

    try {
      const context = buildMujocoContext();

      // ── 자동 루프 명령 감지 ────────────────────────────
      const lower = trimmed.toLowerCase();
      const isAutoStart =
        lower.includes("자동") ||
        lower.includes("루프") ||
        lower.includes("gaming") ||
        lower.includes("agent") ||
        lower.includes("반복") ||
        (lower.includes("계속") && lower.includes("학습"));

      if (isAutoStart && !autoLoopRunning) {
        startAutoLoop(trimmed);
        return;
      }

      const isAutoStop =
        lower.includes("멈") ||
        lower.includes("스톱") ||
        lower.includes("stop") ||
        lower.includes("중단") ||
        lower.includes("그만");

      if (isAutoStop && autoLoopRunning) {
        stopAutoLoop();
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "자동 루프를 멈춥니다.",
          },
        ]);
        return;
      }

      // ── test 명령 ──────────────────────────────────────
      if (lower === "test") {
        const rawTestKeyframes: VmdKeyframe[] = [
          {
            frame: 0,
            timeSec: 0,
            pose: {
              l_hip_pitch: 0,
              r_hip_pitch: 0,
              l_knee: -0.2,
              r_knee: 0.2,
              l_sho_roll: 0.52,
              r_sho_roll: -0.52,
              l_el: -0.3,
              r_el: 0.3,
              l_hip_roll: 0,
              r_hip_roll: 0,
              l_hip_yaw: 0,
              r_hip_yaw: 0,
              l_ank_pitch: 0,
              r_ank_pitch: 0,
              l_ank_roll: 0,
              r_ank_roll: 0,
              l_sho_pitch: 0.3,
              r_sho_pitch: -0.3,
              head_pan: 0,
              head_tilt: 0,
            },
          },
          {
            frame: 15,
            timeSec: 0.5,
            pose: {
              l_hip_pitch: 0.4,
              r_hip_pitch: -0.3,
              l_knee: -0.5,
              r_knee: 0.15,
              l_sho_roll: 0.52,
              r_sho_roll: -0.52,
              l_el: -0.3,
              r_el: 0.3,
              l_hip_roll: 0,
              r_hip_roll: 0,
              l_hip_yaw: 0,
              r_hip_yaw: 0,
              l_ank_pitch: -0.2,
              r_ank_pitch: 0.15,
              l_ank_roll: 0,
              r_ank_roll: 0,
              l_sho_pitch: -0.3,
              r_sho_pitch: 0.3,
              head_pan: 0,
              head_tilt: 0,
            },
          },
          {
            frame: 30,
            timeSec: 1.0,
            pose: {
              l_hip_pitch: -0.3,
              r_hip_pitch: 0.4,
              l_knee: -0.15,
              r_knee: 0.5,
              l_sho_roll: 0.52,
              r_sho_roll: -0.52,
              l_el: -0.3,
              r_el: 0.3,
              l_hip_roll: 0,
              r_hip_roll: 0,
              l_hip_yaw: 0,
              r_hip_yaw: 0,
              l_ank_pitch: 0.15,
              r_ank_pitch: -0.2,
              l_ank_roll: 0,
              r_ank_roll: 0,
              l_sho_pitch: 0.3,
              r_sho_pitch: -0.3,
              head_pan: 0,
              head_tilt: 0,
            },
          },
          {
            frame: 45,
            timeSec: 1.5,
            pose: {
              l_hip_pitch: 0,
              r_hip_pitch: 0,
              l_knee: -0.2,
              r_knee: 0.2,
              l_sho_roll: 0.52,
              r_sho_roll: -0.52,
              l_el: -0.3,
              r_el: 0.3,
              l_hip_roll: 0,
              r_hip_roll: 0,
              l_hip_yaw: 0,
              r_hip_yaw: 0,
              l_ank_pitch: 0,
              r_ank_pitch: 0,
              l_ank_roll: 0,
              r_ank_roll: 0,
              l_sho_pitch: 0,
              r_sho_pitch: 0,
              head_pan: 0,
              head_tilt: 0,
            },
          },
        ];

        const testKeyframes = rawTestKeyframes.map((kf) => ({
          ...kf,
          pose: applyJointConstraints({ ...kf.pose }),
        }));

        startMotionDrive(testKeyframes);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "테스트 모션 실행 중...",
          },
        ]);
        return;
      }

      // ── 1) intent ──────────────────────────────────────
      const intentReqBody = {
        message: trimmed,
        history: historyPayload,
        context,
      };
      const intentInChars = JSON.stringify(intentReqBody).length;
      emitAiStage({
        stage: "intent",
        phase: "start",
        inChars: intentInChars,
        inTok: estTok(intentInChars),
      });

      const tIntent = performance.now();
      const intentResp = await fetch("/api/intent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(intentReqBody),
      });
      const intentData = (await intentResp.json()) as IntentResponse;
      emitAiStage({
        stage: "intent",
        phase: "end",
        ok: intentResp.ok,
        ms: Math.round(performance.now() - tIntent),
        inChars: intentInChars,
        outChars: JSON.stringify(intentData).length,
        inTok: estTok(intentInChars),
        outTok: estTok(JSON.stringify(intentData).length),
      });

      if (!intentResp.ok)
        throw new Error(intentData?.error || "명령 해석(intent) 실패");

      const displayText = intentData?.text?.trim();
      if (displayText) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: displayText,
          },
        ]);
      }

      // ── RL 시작 명령 체크 ──────────────────────────────
      const goalRaw = intentData?.intent?.goal ?? "";
      const goal = String(goalRaw).toLowerCase();

      const isRLStart =
        goal === "start_reinforcement_learning" ||
        goal === "start_reinforcement_learning_for_human" ||
        goal === "start_reinforcement_learning_for_humanoid" ||
        goal === "start_reinforcement_learning_for_mujoco" ||
        goal === "start_reinforcement_learning_session" ||
        (goal.includes("reinforcement") &&
          goal.includes("learning") &&
          goal.includes("start"));

      if (isRLStart) {
        window.dispatchEvent(
          new CustomEvent("rl:startTraining", {
            detail: {
              durationMs: intentData?.intent?.duration_ms ?? 15000,
            },
          }),
        );
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "강화학습 트레이닝을 시작합니다. (MuJoCo RL 모드)",
          },
        ]);
        return;
      }

      // ── stop 체크 ──────────────────────────────────────
      if (
        (goal.includes("stop") ||
          goal.includes("stand") ||
          goal.includes("pause")) &&
        isDriving
      ) {
        stopMotionDrive();
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "모션을 멈췄습니다.",
          },
        ]);
        return;
      }

      // ── 2) motor ───────────────────────────────────────
      const motorReqBody = {
        intent: intentData.intent,
        context,
        message: trimmed,
      };
      const motorInChars = JSON.stringify(motorReqBody).length;
      emitAiStage({
        stage: "motor",
        phase: "start",
        inChars: motorInChars,
        inTok: estTok(motorInChars),
      });

      const tMotor = performance.now();
      const motorResp = await fetch("/api/motor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(motorReqBody),
      });
      const motorData = (await motorResp.json()) as MotorResponse;
      emitAiStage({
        stage: "motor",
        phase: "end",
        ok: motorResp.ok,
        ms: Math.round(performance.now() - tMotor),
        inChars: motorInChars,
        outChars: JSON.stringify(motorData).length,
        inTok: estTok(motorInChars),
        outTok: estTok(JSON.stringify(motorData).length),
      });

      if (!motorResp.ok)
        throw new Error(motorData?.error || "motor compile 실패");
      if (
        !Array.isArray(motorData?.motions) ||
        motorData.motions.length === 0
      )
        throw new Error("motor API가 motions를 반환하지 않았습니다.");

      if (motorData.analysis) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `🧠 ${motorData.analysis}`,
          },
        ]);
      }

      // ── 3) execute ─────────────────────────────────────
      void (async () => {
        const execReqBody = { motions: motorData.motions, context };
        const execInChars = JSON.stringify(execReqBody).length;
        emitAiStage({
          stage: "execute",
          phase: "start",
          inChars: execInChars,
          inTok: estTok(execInChars),
        });

        const tExec = performance.now();
        try {
          const execResponse = await fetch("/api/execute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(execReqBody),
          });
          const execData: ExecuteResponse | null = await execResponse
            .json()
            .catch(() => null);

          emitAiStage({
            stage: "execute",
            phase: "end",
            ok: execResponse.ok,
            ms: Math.round(performance.now() - tExec),
            inChars: execInChars,
            outChars: execData ? JSON.stringify(execData).length : 0,
            inTok: estTok(execInChars),
            outTok: execData
              ? estTok(JSON.stringify(execData).length)
              : 0,
          });

          if (!execResponse.ok) {
            console.warn(
              "[execute] failed:",
              execResponse.status,
              execData,
            );
            return;
          }

          const finalMotions =
            Array.isArray(execData?.motions) &&
            execData!.motions!.length > 0
              ? execData!.motions!
              : motorData.motions;

          const keyframes = motorMotionsToKeyframes(
            finalMotions as Array<{
              joint: string;
              angle: number;
              time: number;
            }>,
            jointsRef.current,
          );

          if (keyframes.length >= 2) {
            startMotionDrive(keyframes);
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: `모션 적용 중 (${keyframes.length}개 키프레임, ${keyframes[keyframes.length - 1].timeSec.toFixed(1)}초)`,
              },
            ]);
          } else if (keyframes.length === 1) {
            sendToMujocoIframe({
              type: "SET_JOINT_TARGETS_PD",
              enabled: true,
              targets: keyframes[0].pose,
              kp: 80,
              kd: 6,
            });
          }

          window.dispatchEvent(
            new CustomEvent("robot:moveJoints", {
              detail: {
                motions: finalMotions,
                options: { animate: true, defaultDurationMs: 350 },
              },
            }),
          );
        } catch (e) {
          emitAiStage({
            stage: "execute",
            phase: "end",
            ok: false,
            ms: Math.round(performance.now() - tExec),
          });
          console.error("[execute] error:", e);
        }
      })();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "알 수 없는 오류";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════

  return (
    <div className="fixed bottom-4 left-4 right-4 z-40 sm:right-auto sm:w-[360px]">
      <div className="rounded-[28px] border border-white/70 bg-white/90 p-5 shadow-[0_24px_60px_rgba(0,0,0,0.15)] backdrop-blur-xl">
        <h2 className="text-lg font-semibold text-[#1c1c1c]">
          어디서부터 시작할까요?
        </h2>

        <div className="mt-4 space-y-3">
          {/* ── 메시지 영역 ───────────────────────────────── */}
          <div className="max-h-64 space-y-2 overflow-y-auto pr-1 text-sm text-[#2f2f2f]">
            {messages.length === 0 ? (
              <p className="rounded-2xl border border-[#f3e9ce] bg-[#fffbf3] px-4 py-3 text-[#7d7256]">
                {EMPTY_HINT}
              </p>
            ) : (
              messages.map((item) => (
                <div
                  key={item.id}
                  className={`flex ${
                    item.role === "assistant"
                      ? "justify-start"
                      : "justify-end"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 shadow-sm ${
                      item.role === "assistant"
                        ? "border border-[#e7dcbd] bg-white text-[#3f381f]"
                        : "bg-[#121212] text-white"
                    }`}
                  >
                    {item.content}
                  </div>
                </div>
              ))
            )}

            {(isLoading || plannerLoading) && (
              <div className="flex items-center gap-2 text-xs text-[#7d7256]">
                <Loader2 className="h-4 w-4 animate-spin" />
                {plannerLoading
                  ? "플래너가 다음 목표를 만드는 중…"
                  : "생각을 정리하고 있어요…"}
              </div>
            )}
          </div>

          {/* ── GamingAgent 자동 루프 상태 ─────────────────── */}
          {autoLoopRunning && (
            <div className="rounded-2xl border border-[#e0d4f5] bg-[#f3eeff] px-3 py-2">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-xs font-semibold text-[#5b21b6]">
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  GamingAgent 루프
                </span>
                <button
                  type="button"
                  onClick={stopAutoLoop}
                  className="rounded-lg bg-[#c62828] px-2 py-1 text-[10px] font-bold text-white transition hover:bg-[#b71c1c]"
                >
                  STOP
                </button>
              </div>
              <div className="mt-1 font-mono text-[10px] text-[#7c3aed]">
                iter: {autoLoopIter} | {autoLoopStatus}
              </div>
            </div>
          )}

          {/* ── drive 상태 표시 ─────────────────────────────── */}
          {isDriving && !autoLoopRunning && (
            <div className="flex items-center justify-between rounded-2xl border border-[#c8e6c9] bg-[#e8f5e9] px-3 py-2">
              <span className="flex items-center gap-2 text-xs text-[#2e7d32]">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#4caf50]" />
                모션 드라이브 실행 중
              </span>
              <button
                type="button"
                onClick={stopMotionDrive}
                className="rounded-lg bg-[#c62828] px-2 py-1 text-[10px] font-bold text-white transition hover:bg-[#b71c1c]"
              >
                STOP
              </button>
            </div>
          )}

          {/* ── 에러 ───────────────────────────────────────── */}
          {error && (
            <p className="rounded-xl border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-600">
              {error}
            </p>
          )}

          {/* ── 입력 폼 ───────────────────────────────────── */}
          <form
            className="flex flex-col gap-3"
            onSubmit={handleSubmit}
            role="search"
          >
            <div className="flex items-center gap-3 rounded-2xl border border-[#efe4c8] bg-[#fffbf3] px-2.5 py-1.5">
              <button
                type="button"
                className="flex h-11 w-11 items-center justify-center rounded-2xl border border-[#f2e7cc] bg-white text-[#968812] transition-colors hover:bg-[#fbf3dc]"
                aria-label="파일이나 프롬프트 템플릿 추가"
                disabled={isLoading}
              >
                <Plus className="h-5 w-5" strokeWidth={2.4} />
              </button>

              <input
                aria-label="LLM에게 질문하기"
                autoComplete="off"
                className="flex-1 bg-transparent text-base text-[#3a3425] placeholder:text-[#b9ae8f] focus:outline-none"
                onChange={(event) => setMessage(event.target.value)}
                placeholder="무엇이든 물어보세요"
                value={message}
              />

              <button
                type="submit"
                className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#121212] text-white transition-colors hover:bg-black disabled:opacity-60"
                aria-label="메시지 전송"
                disabled={isLoading}
              >
                {isLoading ? (
                  <Loader2
                    className="h-5 w-5 animate-spin"
                    strokeWidth={2.2}
                  />
                ) : (
                  <Mic className="h-5 w-5" strokeWidth={2.2} />
                )}
              </button>
            </div>

            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <button
                type="button"
                className="flex w-max items-center gap-1.5 rounded-full border border-[#eee2c3] bg-white/80 px-4 py-2 text-sm text-[#5f5a4a] transition hover:bg-[#fdf7e6]"
                disabled
              >
                <Sparkles
                  className="h-4 w-4 text-[#c59f34]"
                  strokeWidth={2.4}
                />
                Extended thinking
                <ChevronDown className="h-4 w-4" strokeWidth={2.2} />
              </button>

              <p className="text-xs text-[#9b9076]">{EMPTY_HINT}</p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
