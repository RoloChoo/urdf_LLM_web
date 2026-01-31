// src/server/llm/modelConfig.ts

export type LlmStage = "intent" | "motor";

export function getModelForStage(stage: LlmStage): string {
  const stageKey = stage === "intent" ? "OPENAI_MODEL_INTENT" : "OPENAI_MODEL_MOTOR";

  // stage 전용 env가 최우선
  const model =
    process.env[stageKey] ??
    process.env.OPENAI_MODEL ??
    process.env.NEXT_PUBLIC_OPENAI_MODEL ??
    "gpt-4o-mini";

  return model;
}

export function shouldDebugModels(): boolean {
  return process.env.DEBUG_LLM_MODELS === "1";
}
