// src/types/rlTaskPlanner.ts

export const RL_EVENTS = {
  trainingSummary: "rl:trainingSummary",
  applyPlan: "rl:applyPlan",
} as const;

export type RLTrainingSummary = {
  when: string; // ISO string
  policyLabel: string;
  episodes: number;

  bestReturn: number;
  lastReturn: number | null;

  actionDim: number;
  obsDim: number;

  frameSkip: number;
  maxSteps: number;

  actionMode?: string;

  baseBodyId?: number | null;
  targetHeight?: number | null;

  termReasonCounts?: Record<string, number>;
};

export type TaskPlannerActionHint = {
  kind: "sine";
  amp?: number; // 기본 진폭
  speed?: number; // 시간 스케일 계수 (1.0이 기본)
  bias?: number[]; // actionDim 길이면 사용
  scale?: number[]; // actionDim 길이면 사용
  phase?: number[]; // actionDim 길이면 사용
};

export type TaskPlannerRLConfig = {
  reward?: Partial<{
    alive: number;
    wUpright: number;
    wHeight: number;
    wCtrl: number;
    minUprightForReward: number;
    heightK: number;
  }>;
  terminate?: Partial<{
    minUpright: number;
    fallHeightFrac: number;
    minHeightAbs: number;
  }>;
  frameSkip?: number;
  maxSteps?: number;
  actionMode?: "normalized" | "direct";
};

export type TaskPlannerPlan = {
  goal: string;
  rlConfig?: TaskPlannerRLConfig;
  actionHint?: TaskPlannerActionHint;
};

export type TaskPlannerResponse = {
  text: string;
  plan: TaskPlannerPlan;
  error?: string;
};

// ---- optional runtime guards (프론트 안전용) -------------------------------

export function isTaskPlannerPlan(x: unknown): x is TaskPlannerPlan {
  if (!x || typeof x !== "object") return false;
  const any = x as any;
  return typeof any.goal === "string";
}