import { Quaternion, Euler } from "three";

// 기본 VMD 본 이름 → URDF 조인트 후보 매핑 테이블
const VMD_TO_URDF: { [key: string]: string[] } = {
  // ===== 몸통 =====
  "センター": ["base_link", "hips", "pelvis", "torso", "root"],
  "下半身": ["lower_body", "hips", "pelvis"],
  "上半身": ["spine", "torso", "chest"],
  "上半身2": ["upper_chest", "chest", "spine1", "spine2"],
  "首": ["neck", "head_pan"],
  "頭": ["head", "head_tilt"],

  // ===== 왼팔 =====
  "左肩": ["l_shoulder_pitch", "left_shoulder_pitch", "l_sho_pitch", "LShoulderPitch"],
  "左腕": ["l_shoulder_roll", "left_shoulder_roll", "l_sho_roll", "LShoulderRoll"],
  "左ひじ": ["l_elbow", "left_elbow", "l_el", "LElbow"],
  "左手首": ["l_wrist", "left_wrist", "LWrist"],

  // ===== 오른팔 =====
  "右肩": ["r_shoulder_pitch", "right_shoulder_pitch", "r_sho_pitch", "RShoulderPitch"],
  "右腕": ["r_shoulder_roll", "right_shoulder_roll", "r_sho_roll", "RShoulderRoll"],
  "右ひじ": ["r_elbow", "right_elbow", "r_el", "RElbow"],
  "右手首": ["r_wrist", "right_wrist", "RWrist"],

  // ===== 다리 =====
  "左足": ["l_hip_pitch", "left_hip_pitch", "l_leg", "LHip"],
  "左ひざ": ["l_knee", "left_knee", "LKnee"],
  "左足首": ["l_ankle_pitch", "left_ankle_pitch", "LAnkle"],

  "右足": ["r_hip_pitch", "right_hip_pitch", "r_leg", "RHip"],
  "右ひざ": ["r_knee", "right_knee", "RKnee"],
  "右足首": ["r_ankle_pitch", "right_ankle_pitch", "RAnkle"],
};

export type VmdKeyframe = {
  frame: number;
  pose: { [key: string]: number }; // { "joint_name": angle }
};

export type VmdLoaderOptions = {
  boneMap?: Record<string, string>; // VMD 본 이름 → URDF 조인트 이름 직접 매핑
};

export class VmdLoader {
  /**
   * VMD 파일을 로드하여 키프레임 배열로 변환
   * @param buffer VMD 파일의 ArrayBuffer
   * @param robotJointNames 로봇의 조인트 이름 목록
   * @param opts 옵션 (boneMap: 외부에서 제공하는 본 매핑 테이블)
   * @returns VmdKeyframe 배열
   */
  static load(
    buffer: ArrayBuffer,
    robotJointNames: string[],
    opts?: VmdLoaderOptions
  ): VmdKeyframe[] {
    const data = new DataView(buffer);
    const decoder = new TextDecoder("shift-jis");
    let offset = 0;

    // VMD 파일 헤더 검증
    const magicBytes = new Uint8Array(buffer, 0, 30);
    const magic = decoder.decode(magicBytes).replace(/\0/g, "").trim();
    if (!magic.startsWith("Vocaloid Motion Data")) {
      throw new Error("Invalid VMD file");
    }
    offset += 30 + 20;

    // 모션 데이터 개수 읽기
    const motionCount = data.getUint32(offset, true);
    offset += 4;

    const frames: VmdKeyframe[] = [];
    const frameMap: { [frame: number]: { [joint: string]: number } } = {};

    // 각 모션 데이터 파싱
    for (let i = 0; i < motionCount; i++) {
      // 본 이름 읽기 (15바이트)
      const boneNameBytes = new Uint8Array(buffer, offset, 15);
      const boneName = decoder.decode(boneNameBytes).replace(/\0/g, "").trim();
      offset += 15;

      // 프레임 번호 읽기
      const frameNum = data.getUint32(offset, true);
      offset += 4;

      // 위치 데이터 건너뛰기 (12바이트)
      offset += 12;

      // 쿼터니언 읽기
      const qx = data.getFloat32(offset, true);
      offset += 4;
      const qy = data.getFloat32(offset, true);
      offset += 4;
      const qz = data.getFloat32(offset, true);
      offset += 4;
      const qw = data.getFloat32(offset, true);
      offset += 4;

      // 보간 데이터 건너뛰기 (64바이트)
      offset += 64;

      // ✅ 1순위: 외부에서 제공한 boneMap 사용
      const mapped = opts?.boneMap?.[boneName];
      if (mapped && robotJointNames.includes(mapped)) {
        const angle = this.convertToUrdfAngle(boneName, qx, qy, qz, qw);
        if (!frameMap[frameNum]) frameMap[frameNum] = {};
        frameMap[frameNum][mapped] = angle;
        continue;
      }

      // ✅ 2순위: 기본 VMD_TO_URDF 매핑 테이블 사용 (백업)
      const candidates = VMD_TO_URDF[boneName];
      if (!candidates) continue;

      const targetJoint = this.findMatchingJoint(candidates, robotJointNames);
      if (targetJoint) {
        const angle = this.convertToUrdfAngle(boneName, qx, qy, qz, qw);
        if (!frameMap[frameNum]) frameMap[frameNum] = {};
        frameMap[frameNum][targetJoint] = angle;
      }
    }

    // 프레임 번호 순으로 정렬하여 키프레임 배열 생성
    const sortedFrames = Object.keys(frameMap)
      .map(Number)
      .sort((a, b) => a - b);

    for (const f of sortedFrames) {
      frames.push({ frame: f, pose: frameMap[f] });
    }

    return frames;
  }

  /**
   * 후보 조인트 이름 목록에서 로봇 조인트와 매칭되는 것 찾기
   */
  private static findMatchingJoint(candidates: string[], robotJoints: string[]): string | null {
    for (const cand of candidates) {
      // 정확히 일치하는 조인트
      if (robotJoints.includes(cand)) return cand;
      
      // 대소문자 무시하고 일치하는 조인트
      const found = robotJoints.find(rj => rj.toLowerCase() === cand.toLowerCase());
      if (found) return found;
      
      // 부분 문자열로 포함하는 조인트
      const foundPart = robotJoints.find(rj => rj.toLowerCase().includes(cand.toLowerCase()));
      if (foundPart) return foundPart;
    }
    return null;
  }

  /**
   * VMD 쿼터니언을 URDF 각도로 변환
   * (본 이름에 따라 축 방향 및 오프셋 조정)
   */
  private static convertToUrdfAngle(boneName: string, x: number, y: number, z: number, w: number): number {
    const q = new Quaternion(x, y, z, w);
    const e = new Euler().setFromQuaternion(q, "XYZ");

    // 팔 (腕)
    if (boneName.includes("腕")) {
      return boneName.includes("左") ? e.z + 0.5 : e.z - 0.5;
    }
    
    // 팔꿈치 (ひじ)
    if (boneName.includes("ひじ")) {
      return boneName.includes("左") ? -Math.abs(e.x) : Math.abs(e.x);
    }

    // 다리 (足, 발목 제외)
    if (boneName.includes("足") && !boneName.includes("首")) {
      return -e.x;
    }

    // 무릎 (ひざ)
    if (boneName.includes("ひざ")) {
      return Math.abs(e.x);
    }

    return e.z;
  }
}