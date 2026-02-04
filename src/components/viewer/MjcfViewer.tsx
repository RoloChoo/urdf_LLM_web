"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCcw, Settings2, Activity, Upload, FileVideo } from "lucide-react";
import type { JointInfo, SimState } from "@/types/robotControl";
import { useMujocoScene } from "@/hooks/useMujocoScene";
import { useRobot } from "@/hooks/useRobot";
import { VmdLoader, type VmdKeyframe } from "@/utils/VmdLoader";
import { autoMapVmdBonesToMujocoJoints } from "@/utils/autoVmdMujocoMap";

import { createIframeRPC } from "@/utils/iframeRpc";
import { ReferenceMotion } from "@/utils/ReferenceMotion";
import { RLEnvironmentCore } from "@/rl/RLEnvironmentCore";
import { PPOAgent } from "@/rl/PPOAgent";

type AgentMode = "MANUAL" | "RANDOM" | "IMITATION_ZERO" | "PPO";

export default function RobotControlPanel() {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const iframeWindowRef = useRef<Window | null>(null);

  const { registerIframeWindow, resetPose, pauseSimulation, resumeSimulation } = useMujocoScene();
  const { activeRobotType, setActiveRobotType, setActiveRobotOwner, setActiveRobotName } = useRobot();

  const [joints, setJoints] = useState<Record<string, JointInfo>>({});
  const [simState, setSimState] = useState<SimState>("PAUSED");
  const [simTime, setSimTime] = useState(0);
  const [fps, setFps] = useState(0);
  const [draggingJoint, setDraggingJoint] = useState<string | null>(null);

  // VMD (reference motion)
  const [vmdMotion, setVmdMotion] = useState<VmdKeyframe[]>([]);
  const [boneMap, setBoneMap] = useState<Record<string, string>>({});

  // RL
  const [agentMode, setAgentMode] = useState<AgentMode>("MANUAL");
  const [isTraining, setIsTraining] = useState(false);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [lastReward, setLastReward] = useState(0);
  const [rlSteps, setRlSteps] = useState(0);

  // PPO stats
  const [ppoStats, setPpoStats] = useState<{piLoss:number; vLoss:number; ent:number; kl:number} | null>(null);
  const [ppoEpisodes, setPpoEpisodes] = useState(0);

  const rlAbortRef = useRef<AbortController | null>(null);
  const envRef = useRef<RLEnvironmentCore | null>(null);
  const agentRef = useRef<PPOAgent | null>(null);

  // MJCF 로봇 강제 선택
  useEffect(() => {
    if (activeRobotType !== "MJCF") {
      setActiveRobotType("MJCF");
      setActiveRobotOwner("placeholder");
      setActiveRobotName("humanoid");
    }
  }, [activeRobotType, setActiveRobotName, setActiveRobotOwner, setActiveRobotType]);

  const sendMessage = useCallback((msg: any) => {
    const win = iframeWindowRef.current;
    if (!win) return;
    win.postMessage(msg, "*");
  }, []);

  const handleIframeLoad = useCallback(() => {
    const win = iframeRef.current?.contentWindow ?? null;
    if (!win) return;
    iframeWindowRef.current = win;
    registerIframeWindow(win);
  }, [registerIframeWindow]);

  // iframe 메시지(기존 SYNC_STATE 등 UI 업데이트)
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const iframeWin = iframeRef.current?.contentWindow;
      if (!iframeWin) return;
      if (event.source !== iframeWin) return;

      iframeWindowRef.current = iframeWin;
      const { type, payload } = event.data ?? {};

      switch (type) {
        case "IFRAME_READY":
          registerIframeWindow(iframeWin);
          break;
        case "SCENE_LOADED":
          resumeSimulation();
          setSimState("RUNNING");
          sendMessage({ type: "GET_JOINT_INFO" });
          break;
        case "SYNC_STATE":
          if (!payload) return;
          setSimTime(payload.time ?? 0);
          setFps(payload.fps ?? 0);

          setJoints((prev) => {
            const next = { ...prev };
            const incoming = (payload.joints ?? {}) as Record<string, JointInfo>;
            for (const [name, info] of Object.entries(incoming)) {
              if (draggingJoint !== name) next[name] = info;
            }
            return next;
          });
          break;
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
  }, [draggingJoint, registerIframeWindow, resumeSimulation, sendMessage]);

  const handleSliderChange = (name: string, val: number) => {
    // manual control = training 중이면 강제로 끄기
    if (isTraining) setIsTraining(false);

    setJoints((prev) => ({ ...prev, [name]: { ...prev[name], val } }));
    sendMessage({ type: "CONTROL_JOINT", jointName: name, value: val });
  };

  // joints 키(이름)만 바뀔 때 자동매핑
  const jointNamesString = useMemo(() => Object.keys(joints).sort().join(","), [joints]);

  useEffect(() => {
    const robotJointNames = jointNamesString ? jointNamesString.split(",") : [];
    if (robotJointNames.length === 0) return;

    const { map, unsure } = autoMapVmdBonesToMujocoJoints(robotJointNames);
    setBoneMap(map);

    console.log(`[AutoMap] Mapped ${Object.keys(map).length} bones.`);
    if (unsure.length > 0) console.warn("[AutoMap] Unsure mappings:", unsure);
  }, [jointNamesString]);

  // VMD 업로드: reference motion으로만 저장 (자동 재생 X)
  const handleVmdUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const buffer = await file.arrayBuffer();
      const robotJointNames = Object.keys(joints);

      const motion = VmdLoader.load(buffer, robotJointNames, { boneMap });
      setVmdMotion(motion);

      console.log(`✅ Loaded VMD: ${motion.length} keyframes`);
      alert("VMD loaded as Reference Motion. (RL에서 추종 대상으로 사용)");
    } catch (err) {
      console.error("VMD Load Failed:", err);
      alert("VMD 파일을 읽는데 실패했습니다. (콘솔 확인)");
    }
  };

  const jointOrder = useMemo(() => Object.keys(joints).sort(), [joints]);

  const buildEnv = useCallback(async () => {
    const winGetter = () => iframeWindowRef.current;

    if (jointOrder.length === 0) throw new Error("No joints loaded yet.");
    if (vmdMotion.length === 0) throw new Error("No VMD loaded (reference motion required).");

    const rpc = createIframeRPC(winGetter);

    // env config (튜닝 가능)
    const cfg = {
      jointOrder,
      dt: 1 / 60,
      substeps: 4,
      actionScale: 0.35,  // rad (residual)
      kp: 60,
      kd: 4,
    };

    const ref = new ReferenceMotion(jointOrder, vmdMotion, 30, true);
    const env = new RLEnvironmentCore(rpc, cfg, ref);
    await env.init();

    envRef.current = env;
    return { env, rpc };
  }, [jointOrder, vmdMotion]);

  const stopTraining = useCallback(() => {
    rlAbortRef.current?.abort();
    rlAbortRef.current = null;
    setIsTraining(false);

    agentRef.current?.dispose();
    agentRef.current = null;
  }, []);

  const startTrainingLoop = useCallback(async () => {
    stopTraining();

    const abort = new AbortController();
    rlAbortRef.current = abort;

    setIsTraining(true);
    setEpisodeReward(0);
    setLastReward(0);
    setRlSteps(0);

    setSimState("RUNNING");
    resumeSimulation();

    const { env, rpc } = await buildEnv();

    // 간단 에이전트:
    // - RANDOM: [-1,1] 랜덤 residual
    // - IMITATION_ZERO: action=0 (즉 qTarget=qRef) = "PD가 ref만 따라가게"
    const actionDim = jointOrder.length;

    let obs = await env.reset(true);

    // 빠르게 돌리고 싶으면 batch step로 확장 가능(iframe에 ENV_STEP_BATCH 구현)
    while (!abort.signal.aborted) {
      let action = new Float32Array(actionDim);

      if (agentMode === "RANDOM") {
        for (let i = 0; i < actionDim; i++) action[i] = (Math.random() * 2 - 1);
      } else if (agentMode === "IMITATION_ZERO") {
        // zeros
      } else {
        // MANUAL이면 루프 자체를 안 돌게 해야 함
      }

      const { reward, done } = await env.step(action);
      setLastReward(reward);
      setEpisodeReward(env.getEpisodeReward());
      setRlSteps(env.getStepCount());

      if (done) {
        obs = await env.reset(true);
        // episodeReward는 reset에서 다시 0으로, UI는 setEpisodeReward에서 업데이트됨
      } else {
        // obs = nextObs; // 지금은 사용 안 함
      }
    }

    // cleanup
    rpc.dispose();
  }, [agentMode, buildEnv, jointOrder.length, resumeSimulation, stopTraining]);

  // PPO 학습 루프
  const startPPO = useCallback(async () => {
    stopTraining();

    const abort = new AbortController();
    rlAbortRef.current = abort;

    setIsTraining(true);
    setPpoStats(null);
    setPpoEpisodes(0);
    setEpisodeReward(0);
    setLastReward(0);
    setRlSteps(0);

    setSimState("RUNNING");
    resumeSimulation();

    const { env, rpc } = await buildEnv();

    let obs = await env.reset(true);

    const obsDim = obs.length;
    const actDim = jointOrder.length;

    const agent = new PPOAgent();
    await agent.init(obsDim, actDim);
    agentRef.current = agent;

    const stepsPerUpdate = 512; // 브라우저는 작게 시작
    let uiTick = 0;
    let episodeCount = 0;

    while (!abort.signal.aborted) {
      // rollout buffers
      const obBuf = new Float32Array(stepsPerUpdate * obsDim);
      const acBuf = new Float32Array(stepsPerUpdate * actDim);
      const rwBuf = new Float32Array(stepsPerUpdate);
      const dnBuf = new Uint8Array(stepsPerUpdate);
      const lpBuf = new Float32Array(stepsPerUpdate);
      const vlBuf = new Float32Array(stepsPerUpdate);

      for (let t = 0; t < stepsPerUpdate; t++) {
        const { action, logp, value } = await agent.act(obs);

        // env.step(action) => reward/done은 React쪽 imitation reward
        const stepRes = await env.step(action);

        obBuf.set(obs, t * obsDim);
        acBuf.set(action, t * actDim);
        rwBuf[t] = stepRes.reward;
        dnBuf[t] = stepRes.done ? 1 : 0;
        lpBuf[t] = logp;
        vlBuf[t] = value;

        if (stepRes.done) {
          episodeCount++;
          setPpoEpisodes(episodeCount);
          obs = await env.reset(true);
        } else {
          obs = stepRes.obs;
        }

        // UI 업데이트는 너무 자주하면 느려짐
        if ((uiTick++ % 20) === 0) {
          setLastReward(stepRes.reward);
          setEpisodeReward(env.getEpisodeReward());
          setRlSteps(env.getStepCount());
        }

        if (abort.signal.aborted) break;
      }

      if (abort.signal.aborted) break;

      // bootstrap value
      const lastVal = await agent.value(obs);

      const stats = await agent.update({
        obsDim,
        actDim,
        steps: stepsPerUpdate,
        obs: obBuf,
        act: acBuf,
        rew: rwBuf,
        done: dnBuf,
        logp: lpBuf,
        val: vlBuf,
        lastVal,
      });

      setPpoStats(stats);
    }

    // cleanup
    try { await rpc.request("ENV_DISABLE"); } catch {}
    rpc.dispose();
  }, [buildEnv, jointOrder.length, resumeSimulation, stopTraining]);

  // 버튼: Play/Pause/Reset
  const onPlay = () => {
    resumeSimulation();
    setSimState("RUNNING");
  };

  const onPause = () => {
    pauseSimulation();
    setSimState("PAUSED");
    stopTraining();
  };

  const onReset = () => {
    resetPose();
    setSimState("PAUSED");
    setSimTime(0);
    stopTraining();
  };

  const rlActive = isTraining && agentMode !== "MANUAL";

  return (
    <div className="w-full h-full flex flex-row bg-[#1e1e1e] text-[#e8e8e8]">
      <div className="flex-grow relative border-r border-[#3a3a3a] min-h-0">
        <iframe
          ref={iframeRef}
          src="/mujoco/mujoco.html"
          onLoad={handleIframeLoad}
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-downloads"
          className="w-full h-full border-none block bg-[#181818]"
          title="MuJoCo Viewer"
        />

        {/* HUD */}
        <div className="absolute top-4 left-4 flex gap-4 text-xs font-mono bg-black/60 p-2 rounded text-[#888]">
          <div className="flex items-center gap-2">
            <Activity size={14} className={simState === "RUNNING" ? "text-green-500" : "text-yellow-500"} />
            <span>{simState}</span>
          </div>
          <div>TIME: {simTime.toFixed(2)}s</div>
          <div>FPS: {fps}</div>
          {rlActive && (
            <div className={`font-bold ${agentMode === "PPO" ? "text-[#9C27B0]" : "text-[#4CAF50]"}`}>
              {agentMode === "PPO" ? "PPO LEARNING" : "RL ACTIVE"}
            </div>
          )}
        </div>
      </div>

      {/* 우측 패널 */}
      <div className="w-[320px] flex flex-col bg-[#252525] min-h-0">
        <div className="h-10 bg-[#353535] flex items-center justify-center border-b border-[#3a3a3a]">
          <span className="font-bold text-sm flex items-center gap-2">
            <Settings2 size={16} /> Robot Control (RL + PPO)
          </span>
        </div>

        {/* 제어 버튼 영역 */}
        <div className="p-4 border-b border-[#3a3a3a] space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={onPlay}
              className={`p-2 rounded flex items-center justify-center gap-2 text-xs font-bold transition-all ${
                simState === "RUNNING" ? "bg-[#4CAF50] text-white" : "bg-[#2a2a2a] hover:bg-[#404040]"
              }`}
            >
              <Play size={14} /> PLAY
            </button>
            <button
              onClick={onPause}
              className={`p-2 rounded flex items-center justify-center gap-2 text-xs font-bold transition-all ${
                simState === "PAUSED" ? "bg-[#FF9800] text-white" : "bg-[#2a2a2a] hover:bg-[#404040]"
              }`}
            >
              <Pause size={14} /> PAUSE
            </button>
            <button
              onClick={onReset}
              className="p-2 rounded flex items-center justify-center gap-2 text-xs font-bold bg-[#2a2a2a] hover:bg-[#F44336] transition-all"
            >
              <RotateCcw size={14} /> RESET
            </button>
          </div>

          {/* VMD 업로드 */}
          <div>
            <input
              type="file"
              accept=".vmd"
              id="vmd-upload"
              className="hidden"
              onChange={handleVmdUpload}
            />
            <label
              htmlFor="vmd-upload"
              className="flex items-center justify-center gap-2 p-2 text-xs border rounded cursor-pointer transition-all bg-[#2a2a2a] text-[#e8e8e8] border-[#444] hover:bg-[#404040]"
            >
              {vmdMotion.length > 0 ? <FileVideo size={14} /> : <Upload size={14} />}
              {vmdMotion.length > 0 ? `Reference VMD (${vmdMotion.length} keys)` : "Load VMD (Reference)"}
            </label>

            {Object.keys(boneMap).length > 0 && (
              <div className="text-[10px] text-center mt-1 text-gray-500">
                Auto-mapped {Object.keys(boneMap).length} joints
              </div>
            )}
          </div>

          {/* RL 모드 */}
          <div className="bg-[#2a2a2a] p-3 rounded border border-[#3a3a3a] space-y-2">
            <div className="text-xs text-[#888] font-mono">RL MODE</div>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => { setAgentMode("MANUAL"); stopTraining(); }}
                className={`p-2 rounded text-xs font-bold ${agentMode === "MANUAL" ? "bg-[#2196F3]" : "bg-[#1f1f1f] hover:bg-[#404040]"}`}
              >
                MANUAL
              </button>
              <button
                onClick={() => setAgentMode("RANDOM")}
                className={`p-2 rounded text-xs font-bold ${agentMode === "RANDOM" ? "bg-[#2196F3]" : "bg-[#1f1f1f] hover:bg-[#404040]"}`}
              >
                RANDOM
              </button>
              <button
                onClick={() => setAgentMode("IMITATION_ZERO")}
                className={`p-2 rounded text-xs font-bold ${agentMode === "IMITATION_ZERO" ? "bg-[#2196F3]" : "bg-[#1f1f1f] hover:bg-[#404040]"}`}
              >
                IMI-0
              </button>
              <button
                onClick={() => setAgentMode("PPO")}
                className={`p-2 rounded text-xs font-bold ${agentMode === "PPO" ? "bg-[#9C27B0]" : "bg-[#1f1f1f] hover:bg-[#404040]"}`}
              >
                PPO
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => {
                  if (agentMode === "PPO") {
                    startPPO().catch(e => alert(e.message));
                  } else if (agentMode !== "MANUAL") {
                    startTrainingLoop().catch(e => alert(e.message));
                  }
                }}
                disabled={agentMode === "MANUAL"}
                className={`p-2 rounded text-xs font-bold ${
                  isTraining 
                    ? (agentMode === "PPO" ? "bg-[#9C27B0]" : "bg-[#4CAF50]")
                    : "bg-[#1f1f1f] hover:bg-[#404040]"
                } disabled:opacity-50`}
              >
                {isTraining ? "RUNNING" : "START"}
              </button>
              <button
                onClick={stopTraining}
                className="p-2 rounded text-xs font-bold bg-[#1f1f1f] hover:bg-[#404040]"
              >
                STOP
              </button>
            </div>

            <div className="text-[11px] font-mono text-[#aaa] space-y-1">
              <div>steps: {rlSteps}</div>
              <div>last r: {lastReward.toFixed(4)}</div>
              <div>ep r: {episodeReward.toFixed(3)}</div>
              {agentMode === "PPO" && <div>episodes: {ppoEpisodes}</div>}
            </div>

            {ppoStats && (
              <div className="bg-[#1f1f1f] p-2 rounded border border-[#3a3a3a]">
                <div className="text-[10px] text-[#888] mb-1">PPO STATS</div>
                <div className="text-[11px] font-mono text-[#aaa] space-y-1">
                  <div>π loss: {ppoStats.piLoss.toFixed(4)}</div>
                  <div>V loss: {ppoStats.vLoss.toFixed(4)}</div>
                  <div>entropy: {ppoStats.ent.toFixed(3)}</div>
                  <div>KL div: {ppoStats.kl.toFixed(5)}</div>
                </div>
              </div>
            )}

            <div className="text-[10px] text-[#666]">
              IMI-0 = action=0 → qTarget=qRef (PD가 ref 추종)
              <br />
              PPO = full neural policy learning
            </div>
          </div>
        </div>

        {/* 관절 리스트 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="text-xs text-[#888] font-mono mb-2">
            JOINTS ({Object.keys(joints).length})
          </div>

          {Object.entries(joints).map(([name, info]) => (
            <div key={name} className="bg-[#2a2a2a] p-3 rounded border border-[#3a3a3a]">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-[#e8e8e8] truncate w-24" title={name}>
                  {name}
                </span>
                <span className="text-[10px] font-mono text-[#4CAF50]">{info.val.toFixed(3)} rad</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-[#555]">{info.min.toFixed(1)}</span>
                <input
                  type="range"
                  min={info.min}
                  max={info.max}
                  step={0.01}
                  value={info.val}
                  onPointerDown={() => setDraggingJoint(name)}
                  onPointerUp={() => setDraggingJoint(null)}
                  onChange={(e) => handleSliderChange(name, parseFloat(e.target.value))}
                  className="flex-1 h-1 bg-[#404040] rounded-lg appearance-none cursor-pointer
                    [&::-webkit-slider-thumb]:appearance-none
                    [&::-webkit-slider-thumb]:w-3
                    [&::-webkit-slider-thumb]:h-3
                    [&::-webkit-slider-thumb]:bg-[#4CAF50]
                    [&::-webkit-slider-thumb]:rounded-full"
                  disabled={rlActive}
                />
                <span className="text-[10px] text-[#555]">{info.max.toFixed(1)}</span>
              </div>
              {rlActive && (
                <div className="text-[10px] text-[#666] mt-1">disabled (RL controlling)</div>
              )}
            </div>
          ))}

          {Object.keys(joints).length === 0 && (
            <div className="text-center text-[#555] text-xs py-10">No joints loaded.</div>
          )}
        </div>
      </div>
    </div>
  );
}
