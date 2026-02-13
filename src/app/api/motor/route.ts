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

// ─── MuJoCo 관절 정의 ───────────────────────────────────────

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

const JOINT_CONSTRAINTS: Record<
  JointName,
  { min: number; max: number; note: string }
> = {
  l_hip_pitch:  { min: -1.2, max: 0.8,  note: "// negative = forward swing" },
  l_hip_roll:   { min: -0.3, max: 0.3,  note: "" },
  l_hip_yaw:    { min: -0.2, max: 0.2,  note: "" },
  l_knee:       { min: 0.0,  max: 1.5,  note: "// MUST be >= 0" },
  l_ank_pitch:  { min: -0.8, max: 0.8,  note: "" },
  l_ank_roll:   { min: -0.3, max: 0.3,  note: "" },
  r_hip_pitch:  { min: -1.2, max: 0.8,  note: "// negative = forward swing" },
  r_hip_roll:   { min: -0.3, max: 0.3,  note: "" },
  r_hip_yaw:    { min: -0.2, max: 0.2,  note: "" },
  r_knee:       { min: 0.0,  max: 1.5,  note: "// MUST be >= 0" },
  r_ank_pitch:  { min: -0.8, max: 0.8,  note: "" },
  r_ank_roll:   { min: -0.3, max: 0.3,  note: "" },
  l_sho_pitch:  { min: -1.5, max: 1.5,  note: "" },
  l_sho_roll:   { min: -0.5, max: 1.5,  note: "" },
  l_el:         { min: -2.0, max: 0.0,  note: "// MUST be <= 0" },
  r_sho_pitch:  { min: -1.5, max: 1.5,  note: "" },
  r_sho_roll:   { min: -1.5, max: 0.5,  note: "" },
  r_el:         { min: 0.0,  max: 2.0,  note: "// MUST be >= 0" },
  head_pan:     { min: -0.8, max: 0.8,  note: "" },
  head_tilt:    { min: -0.5, max: 0.5,  note: "" },
};

// ─── system prompt: MuJoCo용 (analysis 포함) ─────────────────

const SYSTEM_PROMPT_MUJOCO = `You are a MuJoCo bipedal humanoid motor controller.

You must respond with a JSON object containing TWO fields:

1. "analysis": a string in Korean with your reasoning. Include:
   - 현재 상태 진단 (넘어져 있는지, 서 있는지, 어떤 자세인지)
   - 이전 시도에서 뭐가 문제였는지
   - 이번에 어떤 전략으로 접근하는지
   - 구체적으로 어떤 관절을 왜 그 값으로 설정하는지
   - 예상되는 결과와 리스크
   
2. "motions": an array of joint commands

Example response format:
{
  "analysis": "현재 torso 높이가 낮고 pitch가 크다. 이전 iter에서 hip_pitch를 -0.6으로 줬더니 앞으로 넘어졌다. 이번엔 hip_pitch를 -0.2로 줄이고 knee를 더 구부려서 무게중심을 낮추되 안정적으로 유지하겠다. ankle_pitch로 미세 보정한다.",
  "motions": [
    {"joint": "l_hip_pitch", "angle": -0.2, "time": 0},
    {"joint": "l_knee", "angle": 0.4, "time": 0},
    {"joint": "r_hip_pitch", "angle": 0.2, "time": 0},
    {"joint": "r_knee", "angle": 0.1, "time": 0}
  ]
}

JOINTS (${MUJOCO_JOINT_ORDER.length}):
${MUJOCO_JOINT_ORDER.map((name) => {
  const c = JOINT_CONSTRAINTS[name as JointName];
  return `  ${name}: [${c.min}, ${c.max}] ${c.note}`;
}).join("\n")}

RULES:
- Legs must be anti-phase (when left leg swings forward, right leg swings back)
- Arms swing opposite to same-side leg
- l_knee, r_knee >= 0
- l_el <= 0, r_el >= 0
- Keep movements smooth, max delta ±0.15 rad between consecutive keyframes
- Spread motions across time (0, 200, 400... ms)
- analysis는 반드시 한국어로 작성
- 항상 analysis와 motions 둘 다 포함할 것`;

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

    // ─── JSON 파싱 (analysis + motions 지원) ───────────────────

    let parsed: { analysis?: string; motions?: MotorResponse["motions"] };
    try {
      parsed = JSON.parse(reply);
    } catch {
      // JSON 파싱 실패 시 정규식으로 추출 시도
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

    // ─── analysis 추출 ──────────────────────────────────────────

    const analysis =
      typeof parsed?.analysis === "string"
        ? parsed.analysis.trim()
        : null;

    // ─── motions 검증 ───────────────────────────────────────────

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

    const cleaned = parsed.motions.filter(
      (m) =>
        m &&
        typeof m.joint === "string" &&
        m.joint.trim() &&
        typeof m.angle === "number" &&
        Number.isFinite(m.angle) &&
        typeof m.time === "number" &&
        Number.isFinite(m.time),
    );

    if (cleaned.length === 0) {
      return NextResponse.json(
        { error: "motions가 유효한 형식이 아닙니다." },
        { status: 500 },
      );
    }

    console.log("[motor] analysis:", analysis?.slice(0, 100));
    console.log("[motor] motions:", cleaned.length);

    return NextResponse.json({
      motions: cleaned,
      analysis, // ← MuJoCo일 때 LLM의 한국어 분석 포함
    });
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
