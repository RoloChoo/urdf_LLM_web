"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Play, Pause, RotateCcw, Settings2, Activity, Upload, FileVideo } from "lucide-react";
import type { JointInfo, SimState } from "@/types/robotControl";
import { useMujocoScene } from "@/hooks/useMujocoScene";
import { useRobot } from "@/hooks/useRobot";
import { VmdLoader, type VmdKeyframe } from "@/utils/VmdLoader";
// ▼ [추가] 자동 매핑 유틸리티 임포트
import { autoMapVmdBonesToMujocoJoints } from "@/utils/autoVmdMujocoMap";

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

  // === VMD 관련 상태 ===
  const [vmdMotion, setVmdMotion] = useState<VmdKeyframe[]>([]);
  const [isPlayingVmd, setIsPlayingVmd] = useState(false);
  // ▼ [추가] 매핑된 본 정보를 저장할 상태
  const [boneMap, setBoneMap] = useState<Record<string, string>>({});
  
  const vmdRequestRef = useRef<number>();
  const vmdStartTimeRef = useRef<number>(0);

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
      if (vmdRequestRef.current) cancelAnimationFrame(vmdRequestRef.current);
    };
  }, [draggingJoint, registerIframeWindow, resumeSimulation, sendMessage]);

  const handleSliderChange = (name: string, val: number) => {
    setJoints((prev) => ({ ...prev, [name]: { ...prev[name], val } }));
    sendMessage({ type: "CONTROL_JOINT", jointName: name, value: val });
  };

  // === [추가] 관절 이름이 변경될 때만 자동 매핑 실행 (최적화) ===
  // joints 객체는 매 프레임 위치값 때문에 갱신되므로, 키(이름) 목록만 문자열로 만들어 비교
  const jointNamesString = useMemo(() => Object.keys(joints).sort().join(","), [joints]);

  useEffect(() => {
    const robotJointNames = jointNamesString ? jointNamesString.split(",") : [];
    if (robotJointNames.length === 0) return;

    // 자동 매핑 실행
    const { map, unsure } = autoMapVmdBonesToMujocoJoints(robotJointNames);
    setBoneMap(map);

    console.log(`[AutoMap] Mapped ${Object.keys(map).length} bones.`);
    
    if (unsure.length > 0) {
      console.warn("[AutoMap] Unsure mappings (check console):", unsure);
      // 추후 여기서 사용자에게 수동 매핑 모달을 띄울 수 있음
    }
  }, [jointNamesString]);

  // === VMD 업로드 핸들러 ===
  const handleVmdUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const buffer = await file.arrayBuffer();
      const robotJointNames = Object.keys(joints);
      
      // ▼ [변경] boneMap 옵션 전달
      const motion = VmdLoader.load(buffer, robotJointNames, { boneMap });
      
      setVmdMotion(motion);
      console.log(`✅ Loaded VMD: ${motion.length} keyframes`);
      
      startVmdPlayback(motion);
    } catch (err) {
      console.error("VMD Load Failed:", err);
      alert("VMD 파일을 읽는데 실패했습니다. (콘솔 확인)");
    }
  };

  // === VMD 재생 루프 ===
  const startVmdPlayback = (motion: VmdKeyframe[]) => {
    if (motion.length === 0) return;
    
    setIsPlayingVmd(true);
    setSimState("RUNNING");
    sendMessage({ type: "RESUME_SIMULATION" });

    vmdStartTimeRef.current = performance.now();

    const loop = () => {
      const now = performance.now();
      const elapsed = (now - vmdStartTimeRef.current) / 1000; 
      const currentFrame = elapsed * 30; // VMD 30fps 기준

      const keyframe = motion.find(k => k.frame >= currentFrame);

      if (!keyframe) {
        vmdStartTimeRef.current = now; // 반복 재생
        vmdRequestRef.current = requestAnimationFrame(loop);
        return;
      }

      // 관절 명령 전송
      Object.entries(keyframe.pose).forEach(([jointName, angle]) => {
        sendMessage({ type: "CONTROL_JOINT", jointName, value: angle });
        
        setJoints(prev => {
            if (!prev[jointName]) return prev;
            return { ...prev, [jointName]: { ...prev[jointName], val: angle } };
        });
      });

      vmdRequestRef.current = requestAnimationFrame(loop);
    };

    if (vmdRequestRef.current) cancelAnimationFrame(vmdRequestRef.current);
    vmdRequestRef.current = requestAnimationFrame(loop);
  };

  const stopVmd = () => {
    if (vmdRequestRef.current) cancelAnimationFrame(vmdRequestRef.current);
    setIsPlayingVmd(false);
  };

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
          {isPlayingVmd && <div className="text-[#4CAF50] font-bold">VMD PLAYING</div>}
        </div>
      </div>

      {/* 우측 패널 */}
      <div className="w-[320px] flex flex-col bg-[#252525] min-h-0">
        <div className="h-10 bg-[#353535] flex items-center justify-center border-b border-[#3a3a3a]">
          <span className="font-bold text-sm flex items-center gap-2">
            <Settings2 size={16} /> Robot Control
          </span>
        </div>

        {/* 제어 버튼 영역 */}
        <div className="p-4 border-b border-[#3a3a3a] space-y-3">
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => { resumeSimulation(); setSimState("RUNNING"); }}
              className={`p-2 rounded flex items-center justify-center gap-2 text-xs font-bold transition-all ${
                simState === "RUNNING" ? "bg-[#4CAF50] text-white" : "bg-[#2a2a2a] hover:bg-[#404040]"
              }`}
            >
              <Play size={14} /> PLAY
            </button>
            <button
              onClick={() => { pauseSimulation(); setSimState("PAUSED"); stopVmd(); }}
              className={`p-2 rounded flex items-center justify-center gap-2 text-xs font-bold transition-all ${
                simState === "PAUSED" ? "bg-[#FF9800] text-white" : "bg-[#2a2a2a] hover:bg-[#404040]"
              }`}
            >
              <Pause size={14} /> PAUSE
            </button>
            <button
              onClick={() => { resetPose(); setSimState("PAUSED"); setSimTime(0); stopVmd(); }}
              className="p-2 rounded flex items-center justify-center gap-2 text-xs font-bold bg-[#2a2a2a] hover:bg-[#F44336] transition-all"
            >
              <RotateCcw size={14} /> RESET
            </button>
          </div>

          {/* VMD 업로드 버튼 */}
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
              className={`flex items-center justify-center gap-2 p-2 text-xs border rounded cursor-pointer transition-all ${
                isPlayingVmd 
                  ? "bg-[#2a2a2a] border-[#4CAF50] text-[#4CAF50]" 
                  : "bg-[#2a2a2a] text-[#e8e8e8] border-[#444] hover:bg-[#404040]"
              }`}
            >
              {isPlayingVmd ? <FileVideo size={14} /> : <Upload size={14} />}
              {vmdMotion.length > 0 ? `Playing VMD (${vmdMotion.length} frames)` : "Load VMD File"}
            </label>
            {/* 매핑 상태 표시 (간단히) */}
            {Object.keys(boneMap).length > 0 && (
              <div className="text-[10px] text-center mt-1 text-gray-500">
                 Auto-mapped {Object.keys(boneMap).length} joints
              </div>
            )}
          </div>
        </div>

        {/* 관절 리스트 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="text-xs text-[#888] font-mono mb-2">JOINTS ({Object.keys(joints).length})</div>
          {Object.entries(joints).map(([name, info]) => (
            <div key={name} className="bg-[#2a2a2a] p-3 rounded border border-[#3a3a3a]">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-[#e8e8e8] truncate w-24" title={name}>{name}</span>
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
                  className="flex-1 h-1 bg-[#404040] rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#4CAF50] [&::-webkit-slider-thumb]:rounded-full"
                />
                <span className="text-[10px] text-[#555]">{info.max.toFixed(1)}</span>
              </div>
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
