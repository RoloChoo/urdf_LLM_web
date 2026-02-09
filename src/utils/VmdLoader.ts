import { Quaternion, Euler, MathUtils } from "three";

// =========================================================
// 🗺️ [매핑 정답지] Java 코드의 VMD_TO_URDF 100% 이식
// =========================================================
const VMD_TO_URDF: { [key: string]: string[] } = {
  // ===== 몸통/척추 =====
  "センター": ["torso", "Hips", "hips", "Pelvis", "pelvis", "base_link"],
  "下半身": ["hip", "Hips", "hips", "Pelvis", "pelvis", "lower_body"],
  "上半身": ["torso", "Spine", "spine", "Torso", "chest"],
  "上半身2": ["chest", "Chest", "Spine1", "Spine2", "upper_chest"],

  // ===== 머리/목 =====
  "首": ["head_pan", "Neck", "neck", "HeadYaw", "head_yaw"],
  "頭": ["head_tilt", "Head", "head", "HeadPitch", "head_pitch"],

  // ===== 왼팔 =====
  "左肩": ["l_sho_pitch", "LShoulderPitch", "LeftShoulder", "left_shoulder_pitch", "l_shoulder_pitch"],
  "左腕": ["l_sho_roll", "LShoulderRoll", "LeftUpperArm", "left_shoulder_roll", "l_shoulder_roll"],
  "左ひじ": ["l_el", "LElbowYaw", "LElbowRoll", "LeftLowerArm", "left_elbow", "l_elbow"],
  "左手首": ["l_wrist", "LWristYaw", "LeftHand", "left_wrist", "l_wrist_yaw"],

  // ===== 오른팔 =====
  "右肩": ["r_sho_pitch", "RShoulderPitch", "RightShoulder", "right_shoulder_pitch", "r_shoulder_pitch"],
  "右腕": ["r_sho_roll", "RShoulderRoll", "RightUpperArm", "right_shoulder_roll", "r_shoulder_roll"],
  "右ひじ": ["r_el", "RElbowYaw", "RElbowRoll", "RightLowerArm", "right_elbow", "r_elbow"],
  "右手首": ["r_wrist", "RWristYaw", "RightHand", "right_wrist", "r_wrist_yaw"],

  // ===== 왼다리 =====
  "左足": ["l_hip_yaw", "l_hip_pitch", "LHipYawPitch", "LHipPitch", "LeftUpLeg", "left_hip", "l_leg"],
  "左ひざ": ["l_knee", "LKneePitch", "LeftLeg", "left_knee", "l_knee_pitch"],
  "左足首": ["l_ank_pitch", "LAnklePitch", "LeftFoot", "left_ankle", "l_ankle"],
  "左つま先": ["l_ank_roll", "LAnkleRoll", "LeftToeBase", "left_toe", "l_toe"],

  // ===== 오른다리 =====
  "右足": ["r_hip_yaw", "r_hip_pitch", "RHipYawPitch", "RHipPitch", "RightUpLeg", "right_hip", "r_leg"],
  "右ひざ": ["r_knee", "RKneePitch", "RightLeg", "right_knee", "r_knee_pitch"],
  "右足首": ["r_ank_pitch", "RAnklePitch", "RightFoot", "right_ankle", "r_ankle"],
  "右つま先": ["r_ank_roll", "RAnkleRoll", "RightToeBase", "right_toe", "r_toe"],

  // ===== IK 본 (참고용) =====
  "左足ＩＫ": ["l_ank_pitch", "LAnklePitch", "LeftFoot"],
  "右足ＩＫ": ["r_ank_pitch", "RAnklePitch", "RightFoot"],
  "左つま先ＩＫ": ["l_ank_roll", "LAnkleRoll", "LeftToeBase"],
  "右つま先ＩＫ": ["r_ank_roll", "RAnkleRoll", "RightToeBase"],
};

export type VmdKeyframe = {
  frame: number;
  pose: { [key: string]: number };
};

export type VmdLoaderOptions = {
  boneMap?: Record<string, string>;
};

export class VmdLoader {
  static load(
    buffer: ArrayBuffer,
    robotJointNames: string[],
    opts?: VmdLoaderOptions
  ): VmdKeyframe[] {
    const data = new DataView(buffer);
    const decoder = new TextDecoder("shift-jis");
    let offset = 0;

    // 헤더 파싱
    const magicBytes = new Uint8Array(buffer, 0, 30);
    const magic = decoder.decode(magicBytes).replace(/\0/g, "").trim();
    if (!magic.startsWith("Vocaloid Motion Data")) {
      throw new Error("Invalid VMD file");
    }
    offset += 50; // Magic(30) + ModelName(20)

    const motionCount = data.getUint32(offset, true);
    offset += 4;

    const frameMap: { [frame: number]: { [joint: string]: number } } = {};

    for (let i = 0; i < motionCount; i++) {
      // 본 이름
      const boneNameBytes = new Uint8Array(buffer, offset, 15);
      const boneName = decoder.decode(boneNameBytes).replace(/\0/g, "").trim();
      offset += 15;

      // 프레임 번호
      const frameNum = data.getUint32(offset, true);
      offset += 4;

      // 위치 (Skip)
      offset += 12;

      // 회전 (Quaternion)
      const qx = data.getFloat32(offset, true);
      const qy = data.getFloat32(offset + 4, true);
      const qz = data.getFloat32(offset + 8, true);
      const qw = data.getFloat32(offset + 12, true);
      offset += 16;

      // 보간 (Skip)
      offset += 64;

      // 매핑 확인
      let targetJoint: string | undefined = opts?.boneMap?.[boneName];
      if (!targetJoint) {
        const candidates = VMD_TO_URDF[boneName];
        if (candidates) {
          targetJoint = this.findMatchingJoint(candidates, robotJointNames) || undefined;
        }
      }

      if (targetJoint) {
        // 순수 Java 로직 기반 변환 (보정 없음)
        const angle = this.convertToUrdfAngle(boneName, qx, qy, qz, qw);
        
        if (!frameMap[frameNum]) frameMap[frameNum] = {};
        frameMap[frameNum][targetJoint] = angle;
      }
    }

    // 정렬
    return Object.keys(frameMap)
      .map(Number)
      .sort((a, b) => a - b)
      .map(f => ({ frame: f, pose: frameMap[f] }));
  }

  private static findMatchingJoint(candidates: string[], robotJoints: string[]): string | null {
    for (const cand of candidates) {
      if (robotJoints.includes(cand)) return cand;
      const exactMatch = robotJoints.find(rj => rj.toLowerCase() === cand.toLowerCase());
      if (exactMatch) return exactMatch;
      const partialMatch = robotJoints.find(rj => rj.toLowerCase().includes(cand.toLowerCase()));
      if (partialMatch) return partialMatch;
    }
    return null;
  }

  /**
   * 🔥 [Pure Logic] Java 코드를 그대로 TS로 번역
   * 인위적인 오프셋(KNEE_BEND 등) 제거됨.
   */
  private static convertToUrdfAngle(boneName: string, x: number, y: number, z: number, w: number): number {
    const q = new Quaternion(x, y, z, w);
    const e = new Euler().setFromQuaternion(q, "XYZ");
    
    // Java 코드의 Math.toRadians(30) 같은 상수만 남김
    const A_POSE_OFFSET = MathUtils.degToRad(30);

    switch (boneName) {
      // ===== 머리 (Java: -euler.y / -euler.x) =====
      case "首": return -e.y;
      case "頭": return -e.x;

      // ===== 어깨 (Java: -euler.x) =====
      case "左肩":
      case "右肩": return -e.x;

      // ===== 팔 (Java: euler.z +/- 30deg) =====
      // *참고: Java 코드에 30도 오프셋이 있어서 이건 남겨둠 (A-Pose 보정용)
      case "左腕": return e.z + A_POSE_OFFSET;
      case "右腕": return e.z - A_POSE_OFFSET;

      // ===== 팔꿈치 (Java: +/- abs(euler.x)) =====
      case "左ひじ": return -Math.abs(e.x);
      case "右ひじ": return Math.abs(e.x);

      // ===== 다리 (Java: -euler.x) =====
      // 인위적인 HIP_LEAN 제거됨
      case "左足": return -e.x;
      case "右足": return -e.x;

      // ===== 무릎 (Java: euler.x) =====
      // 인위적인 KNEE_BEND 제거됨
      case "左ひざ": return e.x;
      case "右ひざ": return e.x;

      // ===== 발목 (Java: -euler.x) =====
      // 인위적인 ANKLE_LIFT 제거됨
      case "左足首": return -e.x;
      case "右足首": return -e.x;

      // ===== 발끝 (Java: euler.z) =====
      case "左つま先":
      case "右つま先": return e.z;

      // ===== 몸통 (Java: euler.y) =====
      case "センター":
      case "下半身":
      case "上半身":
      case "上半身2": return e.y;

      default: return e.z;
    }
  }
}
