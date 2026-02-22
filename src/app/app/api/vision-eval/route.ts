export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { getModelForStage, shouldDebugModels } from "@/server/llm/modelConfig";

// 비전 평가는 GPT-4o / GPT-5 계열의 vision 기능 사용
function getVisionModel(): string {
  return process.env.LLM_MODEL_VISION ?? "gpt-4o";
}

interface VisionEvalRequest {
  image: string;        // base64 data URL
  jointState: string;
  iteration: number;
  task: string;
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Partial<VisionEvalRequest>;
    const { image, jointState, iteration, task } = body;

    if (!image) {
      return NextResponse.json({ error: "image 필요" }, { status: 400 });
    }

    const model = getVisionModel();
    if (shouldDebugModels()) console.log("[vision-eval] model =", model);

    // base64에서 data:image/png;base64, 접두사 제거
    const base64Data = image.replace(/^data:image\/\w+;base64,/, "");

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model,
        max_tokens: 300,
        response_format: { type: "json_object" },
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: `You are evaluating a bipedal robot in MuJoCo simulation.
Joint state: ${jointState ?? "unknown"}
Iteration: ${iteration ?? 0}
Task: ${task ?? "walking"}

Evaluate the robot's posture from the screenshot.
Respond ONLY valid JSON:
{"posture":"standing|leaning|fallen|crawling","movement":"walking|twitching|stuck|falling|recovering","quality":<0-10>,"fallen":<boolean>,"suggestion":"<advice>"}`,
              },
              {
                type: "image_url",
                image_url: {
                  url: `data:image/png;base64,${base64Data}`,
                  detail: "low",
                },
              },
            ],
          },
        ],
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error("[vision-eval] API error:", err);
      return NextResponse.json({ error: "Vision API 실패" }, { status: 500 });
    }

    const data = await response.json();
    const text = data?.choices?.[0]?.message?.content ?? "";

    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      console.error("[vision-eval] parse failed:", text);
      // fallback
      parsed = {
        posture: "unknown",
        movement: "unknown",
        quality: 5,
        fallen: false,
        suggestion: "continue",
      };
    }

    return NextResponse.json(parsed);
  } catch (error) {
    console.error("[vision-eval]", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "unknown error" },
      { status: 500 },
    );
  }
}