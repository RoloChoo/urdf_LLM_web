// /api/motor/route.ts

export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { Prompter } from "@/llm/prompter";
import { getModelForStage, shouldDebugModels } from "@/server/llm/modelConfig";

function isLegacyGpt5(modelName: string) {
  const m = modelName.toLowerCase();
  if (!m.startsWith("gpt-5")) return false;
  return !(m.startsWith("gpt-5.1") || m.startsWith("gpt-5.2"));
}

function readIntEnv(name: string, fallback: number) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

// ─── MuJoCo context 판별 ─────────────────────────────────────

function isMujocoContext(context?: string): boolean {
  if (!context) return false;
  return (
    context.includes("MUJOCO_JOINT_CONTEXT") ||
    context.includes("l_hip_pitch") ||
    context.includes("r_knee")
  );
}

// ─── system prompt: 기존 URDF용 ──────────────────────────────

const SYSTEM_PROMPT_URDF = `
You are the Motor Compiler (Stage-2) for Mechaverse robots.

IMPORTANT: Output must be valid json. (json only)
Return ONLY a JSON object with this shape:
{ "motions": [ { "joint": "<string>", "angle": <number radians>, "time": <number ms>, "speed": <optional number> } ] }

Hard rules:
- No extra keys. No text. No reasoning. JSON only.
- motions must be a non-empty array.
- angle is radians, time is milliseconds.

URDF rules (if provided):
- If availableJoints is provided: motions[].joint MUST be exactly one of them.
- Respect jointLimitsRadians if present (stay within bounds).
- Keep it conservative and <= 20 motions.
`;

// ─── MuJoCo 관절 정의 (★ MJCF 기준 정확한 range) ────────────

const MUJOCO_JOINT_ORDER = [
  "l_hip_pitch",
  "l_hip_roll",
  "l_hip_yaw",
  "l_knee",
  "l_ank_pitch",
  "l_ank_roll",
  "r_hip_pitch",
  "r_hip_roll",
  "r_hip_yaw",
  "r_knee",
  "r_ank_pitch",
  "r_ank_roll",
  "l_sho_pitch",
  "l_sho_roll",
  "l_el",
  "r_sho_pitch",
  "r_sho_roll",
  "r_el",
  "head_pan",
  "head_tilt",
] as const;

type JointName = (typeof MUJOCO_JOINT_ORDER)[number];

// ★ MJCF <joint range="..."> 에서 직접 추출
const JOINT_CONSTRAINTS: Record<
  JointName,
  { min: number; max: number; note: string }
> = {
  l_hip_pitch:  { min: -0.524, max: 1.745,  note: "// positive=forward swing" },
  l_hip_roll:   { min: -1.047, max: 0,      note: "// NEGATIVE ONLY — 0=straight" },
  l_hip_yaw:    { min: -0.785, max: 2.618,  note: "// hip twist" },
  l_knee:       { min: -2.269, max: 0,       note: "// NEGATIVE ONLY — bend is negative" },
  l_ank_pitch:  { min: -1.047, max: 1.047,  note: "" },
  l_ank_roll:   { min: -0.524, max: 1.047,  note: "" },
  r_hip_pitch:  { min: -1.745, max: 0.524,  note: "// negative=forward swing (mirrored)" },
  r_hip_roll:   { min: 0,      max: 1.047,  note: "// POSITIVE ONLY — 0=straight" },
  r_hip_yaw:    { min: -2.618, max: 0.785,  note: "// hip twist (mirrored)" },
  r_knee:       { min: 0,      max: 2.269,  note: "// POSITIVE ONLY — bend is positive" },
  r_ank_pitch:  { min: -1.047, max: 1.047,  note: "" },
  r_ank_roll:   { min: -0.524, max: 1.047,  note: "" },
  l_sho_pitch:  { min: -4.363, max: 4.363,  note: "" },
  l_sho_roll:   { min: -1.745, max: 1.745,  note: "" },
  l_el:         { min: -2.793, max: 0,       note: "// NEGATIVE ONLY" },
  r_sho_pitch:  { min: -4.363, max: 4.363,  note: "" },
  r_sho_roll:   { min: -1.745, max: 1.745,  note: "" },
  r_el:         { min: 0,      max: 2.793,  note: "// POSITIVE ONLY" },
  head_pan:     { min: -2.618, max: 2.618,  note: "" },
  head_tilt:    { min: -1.047, max: 0.524,  note: "" },
};

// ★ 서버 사이드 관절 클램프 + 부호 보정
function clampMotionToMjcf(joint: string, angle: number): { joint: string; angle: number; clamped: boolean } {
  const c = JOINT_CONSTRAINTS[joint as JointName];
  if (!c) return { joint, angle, clamped: false };

  let fixed = angle;

  // ★ 부호 자동 보정: 범위가 한쪽만 허용인데 반대 부호로 온 경우
  if (c.min >= 0 && fixed < 0) {
    // 양수만 허용 (r_knee, r_hip_roll, r_el)인데 음수가 옴 → 절대값으로
    fixed = Math.abs(fixed);
  } else if (c.max <= 0 && fixed > 0) {
    // 음수만 허용 (l_knee, l_hip_roll, l_el)인데 양수가 옴 → 부호 반전
    fixed = -Math.abs(fixed);
  }

  // range 클램프
  fixed = Math.max(c.min, Math.min(c.max, fixed));

  return { joint, angle: fixed, clamped: fixed !== angle };
}

// ─── system prompt: MuJoCo용 (★ 자율성 + 정확한 부호 규칙) ──

const SYSTEM_PROMPT_MUJOCO = `You are a MuJoCo bipedal humanoid motor controller with FULL AUTONOMY.

You must respond with a JSON object. Required field: "motions". All other fields are optional but encouraged.

RESPONSE FORMAT:
{
  "analysis": "한국어로 현재 상태 진단 + 전략 설명",
  "hypothesis": "이번 모션의 예상 결과",
  "experiment": "이번에 테스트하는 것",
  "config": { "kp": 80, "kd": 6 },
  "observe": ["foot_contact", "joint_torque"],
  "reset": false,
  "motions": [
    {"joint": "l_hip_pitch", "angle": 0.3, "time": 0},
    {"joint": "l_knee", "angle": -0.4, "time": 0}
  ]
}

FIELD DESCRIPTIONS:
- "analysis": Korean text explaining your reasoning (현재 상태 진단, 이전 문제, 전략, 예상 결과)
- "hypothesis": what you expect to happen
- "experiment": what you're testing this iteration
- "config": { "kp": number(10-200), "kd": number(1-30), "duration_ms": number } — PD gains. Default kp=80, kd=6
- "observe": request sensor data for next iteration. Options: "foot_contact", "joint_torque", "center_of_mass", "velocity", "energy", "momentum", "stability"
- "reset": true to request environment reset (use after falls)
- "motions": array of joint commands (REQUIRED)

JOINTS (${MUJOCO_JOINT_ORDER.length}) — ★ LEFT AND RIGHT LEGS ARE SIGN-MIRRORED:
${MUJOCO_JOINT_ORDER.map((name) => {
  const c = JOINT_CONSTRAINTS[name as JointName];
  return `  ${name}: [${c.min}, ${c.max}] ${c.note}`;
}).join("\n")}

CRITICAL RULES — SIGN CONVENTIONS:
- LEFT leg forward swing: l_hip_pitch POSITIVE (e.g. +0.3)
- RIGHT leg forward swing: r_hip_pitch NEGATIVE (e.g. -0.3)
- LEFT knee bend: l_knee NEGATIVE (e.g. -0.4)
- RIGHT knee bend: r_knee POSITIVE (e.g. +0.4)
- LEFT hip roll: l_hip_roll NEGATIVE only (0=straight, negative=inward)
- RIGHT hip roll: r_hip_roll POSITIVE only (0=straight, positive=inward)
- l_el <= 0, r_el >= 0
- hip_roll should stay CLOSE TO ZERO for standing (±0.05 max for balance)
- DO NOT set hip_roll to large values — this causes legs to spread apart

WALKING RULES:
- Legs MUST be anti-phase (when left swings forward, right swings back)
- Arms swing opposite to same-side leg
- Same magnitude, OPPOSITE sign for mirrored joints
- Keep movements smooth, max delta ±0.15 rad between consecutive keyframes
- Spread motions across time (0, 200, 400... ms)
- Start conservative, increase amplitude gradually
- FIRST PRIORITY: stay standing. Only attempt walking after stable stand confirmed.

BE SCIENTIFIC: hypothesize → test → observe → adapt.
analysis는 반드시 한국어로 작성할 것.`;

// ─── 타입 ────────────────────────────────────────────────────

type MotorRequest = {
  intent: unknown;
  context?: string;
  message?: string;
};

type MotorResponse = {
  motions: Array<{
    joint: string;
    angle: number;
    time: number;
    speed?: number;
  }>;
  analysis?: string | null;
  config?: { kp?: number; kd?: number; duration_ms?: number; frequency_hz?: number } | null;
  observe?: string[] | null;
  reset?: boolean;
  hypothesis?: string | null;
  experiment?: string | null;
};

function createPrompter(useMujoco: boolean) {
  const modelName = getModelForStage("motor");
  if (shouldDebugModels()) console.log("[motor] model =", modelName);

  const params: Record<string, unknown> = {
    response_format: { type: "json_object" },
    max_completion_tokens: readIntEnv(
      "OPENAI_MAX_COMPLETION_TOKENS_MOTOR",
      2000,
    ),
  };

  if (isLegacyGpt5(modelName)) {
    params.reasoning_effort =
      process.env.OPENAI_REASONING_EFFORT_MOTOR ?? "low";
    params.verbosity = process.env.OPENAI_VERBOSITY_MOTOR ?? "low";
  } else {
    params.temperature = Number(
      process.env.OPENAI_TEMPERATURE_MOTOR ?? "0.1",
    );
  }

  return new Prompter({
    name: "MotorCompiler",
    modelName,
    baseUrl: process.env.OPENAI_BASE_URL,
    systemMessage: useMujoco ? SYSTEM_PROMPT_MUJOCO : SYSTEM_PROMPT_URDF,
    params,
  });
}

export async function POST(request: Request) {
  try {
    const body = (await request.json().catch(() => ({}))) as Partial<MotorRequest>;
    const intent = body.intent;
    const context =
      typeof body.context === "string" ? body.context : undefined;
    const userMessage =
      typeof body.message === "string" ? body.message.trim() : "";

    if (!intent || typeof intent !== "object") {
      return NextResponse.json(
        { error: "intent 객체가 필요합니다." },
        { status: 400 },
      );
    }

    const useMujoco = isMujocoContext(context);

    const inputLines = [
      context ? `Context:\n${context}` : "",
      userMessage ? `USER_COMMAND=${userMessage}` : "",
      `INTENT_JSON=${JSON.stringify(intent)}`,
    ].filter(Boolean);

    const prompter = createPrompter(useMujoco);
    const reply = await prompter.prompt([
      { role: "user", content: inputLines.join("\n\n") },
    ]);

    // ─── JSON 파싱 ───────────────────────────────────────────

    let parsed: Record<string, any>;
    try {
      parsed = JSON.parse(reply);
    } catch {
      const jsonMatch = reply?.match(/\{[\s\S]*"motions"[\s\S]*\}/);
      if (jsonMatch) {
        try {
          parsed = JSON.parse(jsonMatch[0]);
        } catch {
          console.error("[motor] JSON parse failed (regex):", reply);
          return NextResponse.json(
            { error: "LLM 결과를 JSON으로 파싱하지 못했습니다." },
            { status: 500 },
          );
        }
      } else {
        console.error("[motor] JSON parse failed:", reply);
        return NextResponse.json(
          { error: "LLM 결과를 JSON으로 파싱하지 못했습니다." },
          { status: 500 },
        );
      }
    }

    // ─── 필드 추출 ───────────────────────────────────────────

    const analysis =
      typeof parsed?.analysis === "string"
        ? parsed.analysis.trim()
        : null;

    const hypothesis =
      typeof parsed?.hypothesis === "string"
        ? parsed.hypothesis.trim()
        : null;

    const experiment =
      typeof parsed?.experiment === "string"
        ? parsed.experiment.trim()
        : null;

    const config =
      parsed?.config && typeof parsed.config === "object"
        ? {
            kp: typeof parsed.config.kp === "number" && Number.isFinite(parsed.config.kp)
              ? parsed.config.kp : undefined,
            kd: typeof parsed.config.kd === "number" && Number.isFinite(parsed.config.kd)
              ? parsed.config.kd : undefined,
            duration_ms: typeof parsed.config.duration_ms === "number" && Number.isFinite(parsed.config.duration_ms)
              ? parsed.config.duration_ms : undefined,
            frequency_hz: typeof parsed.config.frequency_hz === "number" && Number.isFinite(parsed.config.frequency_hz)
              ? parsed.config.frequency_hz : undefined,
          }
        : null;

    const observe =
      Array.isArray(parsed?.observe)
        ? parsed.observe.filter((s: unknown) => typeof s === "string")
        : null;

    const reset =
      typeof parsed?.reset === "boolean"
        ? parsed.reset
        : false;

    // ─── motions 검증 ────────────────────────────────────────

    if (
      !parsed?.motions ||
      !Array.isArray(parsed.motions) ||
      parsed.motions.length === 0
    ) {
      return NextResponse.json(
        { error: "LLM이 motions 배열을 반환하지 않았습니다." },
        { status: 500 },
      );
    }

    const validMotions = parsed.motions.filter(
      (m: any) =>
        m &&
        typeof m.joint === "string" &&
        m.joint.trim() &&
        typeof m.angle === "number" &&
        Number.isFinite(m.angle) &&
        typeof m.time === "number" &&
        Number.isFinite(m.time),
    );

    if (validMotions.length === 0) {
      return NextResponse.json(
        { error: "motions가 유효한 형식이 아닙니다." },
        { status: 500 },
      );
    }

    // ★ 서버 사이드 클램프: LLM이 잘못된 부호를 보내도 자동 보정
    let clampCount = 0;
    const cleaned = validMotions.map((m: any) => {
      if (!useMujoco) return m;
      const result = clampMotionToMjcf(m.joint.trim(), m.angle);
      if (result.clamped) {
        clampCount++;
        console.log(`[motor] CLAMPED: ${m.joint} ${m.angle.toFixed(3)} → ${result.angle.toFixed(3)}`);
      }
      return { ...m, joint: result.joint, angle: result.angle };
    });

    console.log("[motor] analysis:", analysis?.slice(0, 100));
    console.log("[motor] motions:", cleaned.length, clampCount > 0 ? `(${clampCount} clamped)` : "");
    console.log("[motor] motions detail:", JSON.stringify(cleaned.slice(0, 8).map((m: any) => `${m.joint}=${m.angle.toFixed(3)}@${m.time}`)));
    if (config) console.log("[motor] config:", JSON.stringify(config));
    if (observe) console.log("[motor] observe:", observe);
    if (hypothesis) console.log("[motor] hypothesis:", hypothesis.slice(0, 80));
    if (reset) console.log("[motor] reset requested");

    const response: MotorResponse = {
      motions: cleaned,
      analysis,
      config,
      observe,
      reset,
      hypothesis,
      experiment,
    };

    return NextResponse.json(response);
  } catch (error: unknown) {
    console.error("[motor route]", error);

    const msg =
      error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : "알 수 없는 오류가 발생했습니다.";

    if (
      msg.toLowerCase().includes("openai_api_key") ||
      msg.toLowerCase().includes("api key")
    ) {
      return NextResponse.json(
        {
          error:
            "OPENAI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인하세요.",
        },
        { status: 500 },
      );
    }

    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
