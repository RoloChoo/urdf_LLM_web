import * as THREE from "three";

import {
  loadSceneFromURL,
  getPosition,
  getQuaternion,
  toMujocoPos,
  standardNormal,
  stageMjcfSceneToVfs,
  removeAllMujocoRoots,
} from "./mujocoUtils.js";
import { DragStateManager } from "./utils/DragStateManager.js";
import { JointDragManager } from "./utils/JointDragManager.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import load_mujoco from "./wasm/mujoco_wasm.js";

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System (no prefetch)
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

// =========================================================
// Robust WASM/Embind access helpers (pointer OR TypedArray)
// =========================================================
const _utf8 = new TextDecoder("utf-8");

function isPtrNumber(x) {
  return typeof x === "number" && Number.isFinite(x);
}

function getI32(fieldOrPtr, index, mujocoInstance) {
  if (isPtrNumber(fieldOrPtr)) {
    return mujocoInstance.HEAP32[(fieldOrPtr >> 2) + index];
  }
  return fieldOrPtr[index];
}

function getNum(fieldOrPtr, index, mujocoInstance) {
  if (isPtrNumber(fieldOrPtr)) {
    return mujocoInstance.HEAPF64[(fieldOrPtr >> 3) + index];
  }
  return fieldOrPtr[index];
}

function readCStringFromHeapU8(HEAPU8, ptr) {
  let end = ptr;
  while (HEAPU8[end] !== 0) end++;
  return _utf8.decode(HEAPU8.subarray(ptr, end));
}

function readCStringFromU8Array(u8, start) {
  let end = start;
  while (end < u8.length && u8[end] !== 0) end++;
  return _utf8.decode(u8.subarray(start, end));
}

function getJointNameRobust(model, mujocoInstance, jointId) {
  const offset = getI32(model.name_jntadr, jointId, mujocoInstance);
  if (isPtrNumber(model.names)) {
    return readCStringFromHeapU8(mujocoInstance.HEAPU8, model.names + offset);
  }
  return readCStringFromU8Array(model.names, offset);
}

// body name robust
function getBodyNameRobust(model, mujocoInstance, bodyId) {
  const offset = getI32(model.name_bodyadr, bodyId, mujocoInstance);
  if (isPtrNumber(model.names)) {
    return readCStringFromHeapU8(mujocoInstance.HEAPU8, model.names + offset);
  }
  return readCStringFromU8Array(model.names, offset);
}

// keyframe name robust (if model.name_keyadr exists)
function getKeyNameRobust(model, mujocoInstance, keyId) {
  if (!model.name_keyadr) return "";
  const offset = getI32(model.name_keyadr, keyId, mujocoInstance);
  if (isPtrNumber(model.names)) {
    return readCStringFromHeapU8(mujocoInstance.HEAPU8, model.names + offset);
  }
  return readCStringFromU8Array(model.names, offset);
}

function getNjntRobust(model) {
  return typeof model.njnt === "function" ? model.njnt() : model.njnt;
}

function getNbodyRobust(model) {
  return typeof model.nbody === "function" ? model.nbody() : model.nbody;
}

function getNlightRobust(model) {
  return typeof model.nlight === "function" ? model.nlight() : model.nlight;
}

function getNkeyRobust(model) {
  return typeof model.nkey === "function" ? model.nkey() : model.nkey;
}

function getNqRobust(model) {
  return typeof model.nq === "function" ? model.nq() : model.nq;
}

function getNvRobust(model) {
  return typeof model.nv === "function" ? model.nv() : model.nv;
}

// =========================================================
// ready_pose qpos (fallback if keyframe reset unavailable)
// =========================================================
const READY_POSE_QPOS = [
  0, 0, 0.32, // body x, y, z
  1, 0, 0, 0, // body quaternion w, x, y, z
  0, 0, // head_pan, head_tilt
  0.2, -1.0, 0, // l_sho_pitch, l_sho_roll, l_el
  -0.2, 1.0, 0, // r_sho_pitch, r_sho_roll, r_el
  0, 0, 0.6, -1.2, 0.6, 0, // l_hip_yaw ~ l_ank_roll
  0, 0, 0.6, 1.2, -0.6, 0, // r_hip_yaw ~ r_ank_roll
];

// =========================================================
// Keyframe reset (TPOSE = key name "t_pose" in your MJCF)
// =========================================================
function findKeyframeIdByNameHints(model, mujocoInstance, hints) {
  const nkey = getNkeyRobust(model);
  if (!nkey || nkey <= 0) return -1;

  const hs = (hints ?? ["t_pose", "tpose", "t-pose", "t pose"])
    .map((x) => String(x).toLowerCase());

  for (let k = 0; k < nkey; k++) {
    let name = "";
    try {
      name = getKeyNameRobust(model, mujocoInstance, k) || "";
    } catch {
      name = "";
    }
    const n = name.toLowerCase();
    if (!n) continue;
    if (hs.includes(n)) return k;
  }

  for (let k = 0; k < nkey; k++) {
    let name = "";
    try {
      name = getKeyNameRobust(model, mujocoInstance, k) || "";
    } catch {
      name = "";
    }
    const n = name.toLowerCase();
    if (!n) continue;
    if (hs.some((h) => h && n.includes(h))) return k;
  }

  return -1;
}

function resetToKeyframeByIdRobust(model, simulation, mujocoInstance, keyId) {
  simulation.resetData();

  const nkey = getNkeyRobust(model);
  if (!(nkey > 0) || keyId < 0 || keyId >= nkey) {
    throw new Error(`No such keyframe id: ${keyId} (nkey=${nkey})`);
  }

  if (typeof mujocoInstance.mj_resetDataKeyframe === "function") {
    try {
      mujocoInstance.mj_resetDataKeyframe(model, simulation, keyId);
      simulation.forward();
      return;
    } catch (e) {
      console.warn("[POSE] mj_resetDataKeyframe failed; try manual key_qpos copy", e);
    }
  }

  const nq = getNqRobust(model);
  if (!model.key_qpos) {
    throw new Error("[POSE] model.key_qpos not accessible; cannot manual-reset to keyframe");
  }

  const base = keyId * nq;
  for (let i = 0; i < nq && i < simulation.qpos.length; i++) {
    simulation.qpos[i] = getNum(model.key_qpos, base + i, mujocoInstance);
  }

  if (model.key_qvel && simulation.qvel) {
    const nv = getNvRobust(model);
    const vbase = keyId * nv;
    for (let i = 0; i < nv && i < simulation.qvel.length; i++) {
      simulation.qvel[i] = getNum(model.key_qvel, vbase + i, mujocoInstance);
    }
  }

  simulation.forward();
}

function resetToTPose(model, simulation, mujocoInstance) {
  const nkey = getNkeyRobust(model);

  if (nkey > 0) {
    let k = -1;
    try {
      k = findKeyframeIdByNameHints(model, mujocoInstance, ["t_pose", "tpose", "t-pose", "t pose"]);
    } catch {
      k = -1;
    }
    if (k < 0) k = 0;

    try {
      resetToKeyframeByIdRobust(model, simulation, mujocoInstance, k);
      console.log("[POSE] Reset to keyframe:", k, "(t_pose)");
      return;
    } catch (e) {
      console.warn("[POSE] Keyframe reset failed; will fallback to READY_POSE_QPOS", e);
    }
  }

  simulation.resetData();
  for (let i = 0; i < READY_POSE_QPOS.length && i < simulation.qpos.length; i++) {
    simulation.qpos[i] = READY_POSE_QPOS[i];
  }
  simulation.forward();
  console.log("[POSE] Reset by READY_POSE_QPOS fallback");
}

// =========================================================
// Joint Class (Wrapper for Joint Control)
// =========================================================
class Joint {
  constructor(simulation, model, jointId, mujocoInstance) {
    this.simulation = simulation;
    this.model = model;
    this.mj = mujocoInstance;
    this.id = jointId;

    this.type = getI32(model.jnt_type, jointId, this.mj);
    this.qposAddr = getI32(model.jnt_qposadr, jointId, this.mj);
    this.dofAddr = getI32(model.jnt_dofadr, jointId, this.mj);

    this.min = getNum(model.jnt_range, 2 * jointId, this.mj);
    this.max = getNum(model.jnt_range, 2 * jointId + 1, this.mj);

    if (this.min === 0 && this.max === 0) {
      this.min = -3.14;
      this.max = 3.14;
    }
  }

  get value() {
    return this.simulation.qpos[this.qposAddr];
  }

  set value(val) {
    this.simulation.qpos[this.qposAddr] = val;
  }
}

export class Mujoco {
  constructor() {
    this.mujoco = mujoco;
    this.model = null;
    this.state = null;
    this.simulation = null;

    // RL environment state (external stepping)
    this.rl = {
      enabled: false,
      jointOrder: [],
      qposAdr: [],
      dofAdr: [],
      substeps: 4,
      kp: 60,
      kd: 4,

      obsLayout: "ROOT+Q+QD+PHASE",
      withFeet: false,

      foot: {
        leftBodyHint: "l_foot",
        rightBodyHint: "r_foot",
        leftBodyId: -1,
        rightBodyId: -1,
        contactZ: 0.03,
        contactVz: 0.5,
      },
    };

    // PD drive state for VMD-based walking
    this.pdDrive = {
      enabled: false,
      kp: 80,
      kd: 6,
      targets: {},
    };

    // Define parameters
    this.params = {
      scene: null,
      paused: true,
      help: false,
      ctrlnoiserate: 0.0,
      ctrlnoisestd: 0.0,
      keyframeNumber: 0,
    };
    this.mujoco_time = 0.0;
    this.bodies = {};
    this.lights = [];
    this.tmpVec = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();

    // Joint Management
    this.jointMap = {};
    this.lastSync = 0;
    this.lastFrameTime = 0;

    this.container = document.createElement("div");
    document.body.appendChild(this.container);

    this.scene = new THREE.Scene();
    this.scene.name = "scene";

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.001,
      100
    );
    this.camera.name = "PerspectiveCamera";
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.theme = {
      sceneBg: "#fef4da",
      floor: "#fcf4dc",
      ambient: "#fcf4dc",
      hemi: "#fcf4dc",
    };
    this.scene.background = new THREE.Color(this.theme.sceneBg);

    this._createFillLights();

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "default",
    });

    const MAX_PIXEL_RATIO = 1.5;
    this.renderer.setPixelRatio(Math.min(MAX_PIXEL_RATIO, window.devicePixelRatio));

    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.setAnimationLoop(this.render.bind(this));
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener("resize", this.onWindowResize.bind(this));

    this.dragStateManager = new DragStateManager(
      this.scene,
      this.renderer,
      this.camera,
      this.container.parentElement,
      this.controls
    );

    this.jointDragManager = new JointDragManager(
      this.scene,
      this.renderer,
      this.camera,
      this.container.parentElement,
      this.controls,
      null
    );

    this.jointDragManager.enable();
    this.dragStateManager.disable();

    this.hoverRaycaster = new THREE.Raycaster();
    this.hoveredBody = null;
    this.mousePos = new THREE.Vector2();
    this.originalMaterials = new Map();
    this.highlightColor = new THREE.Color(0xfbe651);

    this.renderer.domElement.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.renderer.domElement.addEventListener("mouseleave", this.onMouseLeave.bind(this));
  }

  // =========================
  // RL helpers (ENV obs/step)
  // =========================

  _rlRebuildIndex() {
    const order = this.rl.jointOrder || [];
    this.rl.qposAdr = new Array(order.length);
    this.rl.dofAdr = new Array(order.length);

    for (let i = 0; i < order.length; i++) {
      const name = order[i];
      const j = this.jointMap?.[name];
      if (!j) {
        this.rl.qposAdr[i] = -1;
        this.rl.dofAdr[i] = -1;
        continue;
      }
      this.rl.qposAdr[i] = j.qposAddr;
      this.rl.dofAdr[i] = j.dofAddr;
    }
  }

  _findBodyIdByHint(hint) {
    if (!this.model) return -1;
    const nbody = getNbodyRobust(this.model);
    const h = String(hint ?? "").toLowerCase();

    let best = -1;
    for (let i = 0; i < nbody; i++) {
      let name = "";
      try {
        name = getBodyNameRobust(this.model, this.mujoco, i);
      } catch {
        /* ignore */
      }

      const n = name.toLowerCase();
      if (!n) continue;

      if (n === h) return i;
      if (best < 0 && n.includes(h)) best = i;
    }
    return best;
  }

  _rlResolveFeetBodies() {
    if (!this.rl.withFeet || !this.model) return;

    const L = this._findBodyIdByHint(this.rl.foot.leftBodyHint);
    const R = this._findBodyIdByHint(this.rl.foot.rightBodyHint);

    this.rl.foot.leftBodyId = L;
    this.rl.foot.rightBodyId = R;

    console.log("[RL] Feet resolved:", {
      leftHint: this.rl.foot.leftBodyHint,
      leftBodyId: L,
      rightHint: this.rl.foot.rightBodyHint,
      rightBodyId: R,
    });

    if (L < 0 || R < 0) {
      console.warn(
        "[RL] Foot bodyId resolve failed. Provide foot body hints via ENV_CONFIG payload.foot = {leftBodyHint, rightBodyHint}"
      );
    }
  }

  _rlGetBodyLinVelWorld(bodyId) {
    const sim = this.simulation;
    if (!sim || bodyId < 0) return [0, 0, 0];

    if (sim.cvel && sim.cvel.length >= bodyId * 6 + 6) {
      const base = bodyId * 6;
      return [sim.cvel[base + 3], sim.cvel[base + 4], sim.cvel[base + 5]];
    }

    if (sim.xvelp && sim.xvelp.length >= bodyId * 3 + 3) {
      const base = bodyId * 3;
      return [sim.xvelp[base + 0], sim.xvelp[base + 1], sim.xvelp[base + 2]];
    }

    return [0, 0, 0];
  }

  _rlEstimateFootContact(bodyId) {
    const sim = this.simulation;
    if (!sim || bodyId < 0) return 0;

    if (!sim.xpos || sim.xpos.length < bodyId * 3 + 3) return 0;
    const z = sim.xpos[bodyId * 3 + 2];

    const [, , vz] = this._rlGetBodyLinVelWorld(bodyId);

    const nearGround = z < this.rl.foot.contactZ;
    const slowVert = Math.abs(vz) < this.rl.foot.contactVz;

    return nearGround && slowVert ? 1 : 0;
  }

  _debugListBodiesContains(substr = "foot") {
    if (!this.model) {
      console.log("[DEBUG] no model loaded");
      return;
    }
    const nbody = getNbodyRobust(this.model);
    const s = substr.toLowerCase();
    console.log("[DEBUG] bodies containing:", s);
    for (let i = 0; i < nbody; i++) {
      let name = "";
      try {
        name = getBodyNameRobust(this.model, this.mujoco, i);
      } catch {
        /* */
      }
      if (name.toLowerCase().includes(s)) console.log("  ", i, name);
    }
  }

  _rlBuildObs(phase = 0) {
    const n = this.rl.jointOrder.length;
    const extraFeet = this.rl.withFeet ? 2 + 4 : 0;
    const obs = new Float32Array(3 + 4 + 3 + 3 + n + n + 1 + extraFeet);
    let s = 0;

    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;

    // root pos (3)
    obs[s++] = qpos[0];
    obs[s++] = qpos[1];
    obs[s++] = qpos[2];
    // root quat (4)
    obs[s++] = qpos[3];
    obs[s++] = qpos[4];
    obs[s++] = qpos[5];
    obs[s++] = qpos[6];
    // root linvel (3)
    obs[s++] = qvel[0];
    obs[s++] = qvel[1];
    obs[s++] = qvel[2];
    // root angvel (3)
    obs[s++] = qvel[3];
    obs[s++] = qvel[4];
    obs[s++] = qvel[5];

    // joint qpos (n)
    for (let i = 0; i < n; i++) {
      const qa = this.rl.qposAdr[i];
      obs[s + i] = qa >= 0 ? this.simulation.qpos[qa] : 0;
    }
    s += n;

    // joint qvel (n)
    for (let i = 0; i < n; i++) {
      const da = this.rl.dofAdr[i];
      obs[s + i] = da >= 0 ? this.simulation.qvel[da] : 0;
    }
    s += n;

    // phase (1)
    obs[s++] = phase ?? 0;

    // feet obs (6)
    if (this.rl.withFeet) {
      const L = this.rl.foot.leftBodyId;
      const R = this.rl.foot.rightBodyId;

      const cL = this._rlEstimateFootContact(L);
      const cR = this._rlEstimateFootContact(R);

      const [vLx, vLy] = this._rlGetBodyLinVelWorld(L);
      const [vRx, vRy] = this._rlGetBodyLinVelWorld(R);

      obs[s++] = cL;
      obs[s++] = cR;
      obs[s++] = vLx;
      obs[s++] = vLy;
      obs[s++] = vRx;
      obs[s++] = vRy;
    }

    return obs;
  }

  _rlResetToReference(qRef, qdRef) {
    this.simulation.resetData();

    const n = this.rl.jointOrder.length;
    for (let i = 0; i < n; i++) {
      const qa = this.rl.qposAdr[i];
      const da = this.rl.dofAdr[i];
      if (qa >= 0) this.simulation.qpos[qa] = qRef[i] ?? 0;
      if (da >= 0) this.simulation.qvel[da] = qdRef ? qdRef[i] ?? 0 : 0;
    }

    this.simulation.forward();
  }

  _rlApplyPD(qTarget) {
    const n = this.rl.jointOrder.length;
    const kp = this.rl.kp;
    const kd = this.rl.kd;

    for (let i = 0; i < this.simulation.qfrc_applied.length; i++) {
      this.simulation.qfrc_applied[i] = 0.0;
    }

    for (let i = 0; i < n; i++) {
      const qa = this.rl.qposAdr[i];
      const da = this.rl.dofAdr[i];
      if (qa < 0 || da < 0) continue;

      const q = this.simulation.qpos[qa];
      const qd = this.simulation.qvel[da];
      const qt = qTarget[i];

      const tau = kp * (qt - q) - kd * qd;
      this.simulation.qfrc_applied[da] += tau;
    }
  }

  _createFillLights() {
    if (this.ambientLight) this.scene.remove(this.ambientLight);
    if (this.hemiLight) this.scene.remove(this.hemiLight);

    this.ambientLight = new THREE.AmbientLight(new THREE.Color(this.theme.ambient), 0.2);
    this.ambientLight.name = "AmbientLight";
    this.scene.add(this.ambientLight);

    this.hemiLight = new THREE.HemisphereLight(
      new THREE.Color(this.theme.hemi),
      new THREE.Color(this.theme.hemi),
      0.1
    );
    this.hemiLight.position.set(0, 1, 0);
    this.hemiLight.name = "HemisphereLight";
    this.scene.add(this.hemiLight);
  }

  onMouseMove(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mousePos.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mousePos.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.hoverRaycaster.setFromCamera(this.mousePos, this.camera);
    const intersects = this.hoverRaycaster.intersectObjects(this.scene.children, true);

    let newHoveredBody = null;

    for (let i = 0; i < intersects.length; i++) {
      let obj = intersects[i].object;
      if (obj.bodyID !== undefined && obj.bodyID > 0) {
        const bodyGroup = this.bodies[obj.bodyID];
        if (bodyGroup && bodyGroup.name) {
          newHoveredBody = bodyGroup;
          break;
        }
      }
    }

    if (newHoveredBody !== this.hoveredBody) {
      if (this.hoveredBody) {
        this.removeBodyHighlight(this.hoveredBody);
        window.parent.postMessage({ type: "BODY_MOUSEOUT", bodyName: this.hoveredBody.name }, "*");
      }

      this.hoveredBody = newHoveredBody;

      if (this.hoveredBody) {
        this.applyBodyHighlight(this.hoveredBody);
        window.parent.postMessage({ type: "BODY_MOUSEOVER", bodyName: this.hoveredBody.name }, "*");
      }
    }
  }

  applyBodyHighlight(bodyGroup) {
    bodyGroup.traverse((child) => {
      if (child.isMesh && child.material) {
        if (!child.material.emissive) return;

        if (!this.originalMaterials.has(child.uuid)) {
          this.originalMaterials.set(child.uuid, {
            emissive: child.material.emissive.clone(),
            emissiveIntensity: child.material.emissiveIntensity || 0,
          });
        }

        child.material.emissive.copy(this.highlightColor);
        child.material.emissiveIntensity = 0.3;
      }
    });
  }

  removeBodyHighlight(bodyGroup) {
    bodyGroup.traverse((child) => {
      if (
        child.isMesh &&
        child.material &&
        child.material.emissive &&
        this.originalMaterials.has(child.uuid)
      ) {
        const original = this.originalMaterials.get(child.uuid);
        child.material.emissive.copy(original.emissive);
        child.material.emissiveIntensity = original.emissiveIntensity;
        this.originalMaterials.delete(child.uuid);
      }
    });
  }

  onMouseLeave() {
    if (this.hoveredBody) {
      this.removeBodyHighlight(this.hoveredBody);
      window.parent.postMessage({ type: "BODY_MOUSEOUT", bodyName: this.hoveredBody.name }, "*");
      this.hoveredBody = null;
    }
  }

  updateDragMode() {
    if (this.rl.enabled) {
      this.dragStateManager.disable();
      this.jointDragManager.disable();
      return;
    }

    if (this.params["paused"]) {
      this.dragStateManager.disable();
      this.jointDragManager.enable();
    } else {
      this.jointDragManager.disable();
      this.dragStateManager.enable();
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    const MAX_PIXEL_RATIO = 1.5;
    this.renderer.setPixelRatio(Math.min(MAX_PIXEL_RATIO, window.devicePixelRatio));
  }

  // ─── 수정된 syncToReact: torso 데이터 추가 ───
  syncToReact(timeMS) {
    if (!this.jointMap || Object.keys(this.jointMap).length === 0) return;

    const jointData = {};
    for (const [name, joint] of Object.entries(this.jointMap)) {
      jointData[name] = { id: joint.id, val: joint.value, min: joint.min, max: joint.max };
    }

    const delta = timeMS - (this.lastFrameTime || timeMS - 16);
    const fps = delta > 0 ? Math.round(1000 / delta) : 60;

    // ─── torso body 데이터 추출 ───
    let torso = null;
    if (this.simulation && this.simulation.xpos && this.simulation.xquat) {
      const TORSO_BODY_ID = this._findBodyIdByHint("torso");
      const bid = TORSO_BODY_ID >= 0 ? TORSO_BODY_ID : 1; // fallback to body 1

      const tx = this.simulation.xpos[bid * 3 + 0];
      const ty = this.simulation.xpos[bid * 3 + 1];
      const tz = this.simulation.xpos[bid * 3 + 2];

      const qw = this.simulation.xquat[bid * 4 + 0];
      const qx = this.simulation.xquat[bid * 4 + 1];
      const qy = this.simulation.xquat[bid * 4 + 2];
      const qz = this.simulation.xquat[bid * 4 + 3];

      const pitch = Math.atan2(
        2 * (qw * qy - qz * qx),
        1 - 2 * (qx * qx + qy * qy)
      );
      const roll = Math.atan2(
        2 * (qw * qx + qy * qz),
        1 - 2 * (qx * qx + qy * qy)
      );

      torso = {
        height: tz,
        pitch: pitch,
        roll: roll,
        x: tx,
        y: ty,
      };
    }

    window.parent.postMessage(
      {
        type: "SYNC_STATE",
        payload: {
          joints: jointData,
          time: this.simulation ? this.simulation.time : 0,
          fps: fps,
          torso: torso,
        },
      },
      "*"
    );
    this.lastFrameTime = timeMS;
  }

  render(timeMS) {
    if (!this.model || !this.simulation) {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      return;
    }
    this.controls.update();

    if (timeMS - this.lastSync > 100) {
      this.syncToReact(timeMS);
      this.lastSync = timeMS;
    }

    if (!this.rl.enabled) {
      if (!this.params["paused"]) {
        let timestep = this.model.getOptions().timestep;
        if (timeMS - this.mujoco_time > 35.0) {
          this.mujoco_time = timeMS;
        }
        while (this.mujoco_time < timeMS) {
          if (this.params["ctrlnoisestd"] > 0.0) {
            let rate = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
            let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
            let currentCtrl = this.simulation.ctrl;
            for (let i = 0; i < currentCtrl.length; i++) {
              currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
              this.params["Actuator " + i] = currentCtrl[i];
            }
          }

          for (let i = 0; i < this.simulation.qfrc_applied.length; i++) {
            this.simulation.qfrc_applied[i] = 0.0;
          }

          let dragged = this.dragStateManager.physicsObject;
          if (dragged && dragged.bodyID) {
            const nbody = getNbodyRobust(this.model);
            for (let b = 0; b < nbody; b++) {
              if (this.bodies[b]) {
                getPosition(this.simulation.xpos, b, this.bodies[b].position);
                getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
                this.bodies[b].updateWorldMatrix();
              }
            }

            let bodyID = dragged.bodyID;
            this.dragStateManager.update();
            let force = toMujocoPos(
              this.dragStateManager.currentWorld
                .clone()
                .sub(this.dragStateManager.worldHit)
                .multiplyScalar(this.model.body_mass[bodyID] * 250)
            );
            let point = toMujocoPos(this.dragStateManager.worldHit.clone());
            this.simulation.applyForce(
              force.x,
              force.y,
              force.z,
              0,
              0,
              0,
              point.x,
              point.y,
              point.z,
              bodyID
            );
          }

          if (this.pdDrive?.enabled) {
            const { kp, kd, targets } = this.pdDrive;

            for (const [jointName, qTarget] of Object.entries(targets)) {
              const joint = this.jointMap?.[jointName];
              if (!joint) continue;

              const qa = joint.qposAddr;
              const da = joint.dofAddr;
              if (qa < 0 || da < 0) continue;

              const q = this.simulation.qpos[qa];
              const qd = this.simulation.qvel[da];

              let err = qTarget - q;
              while (err > Math.PI) err -= 2 * Math.PI;
              while (err < -Math.PI) err += 2 * Math.PI;

              const tau = kp * err - kd * qd;
              this.simulation.qfrc_applied[da] += tau;
            }
          }

          this.simulation.step();
          this.mujoco_time += timestep * 1000.0;
        }
      } else {
        if (!this.jointDragManager.active) {
          this.simulation.forward();
        }
      }
    } else {
      if (!this.jointDragManager.active) this.simulation.forward();
    }

    const nbody = getNbodyRobust(this.model);
    for (let b = 0; b < nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.simulation.xpos, b, this.bodies[b].position);
        getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    const nlight = getNlightRobust(this.model);
    for (let l = 0; l < nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.simulation.light_xpos, l, this.lights[l].position);
        getPosition(this.simulation.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

let viewer = new Mujoco();
window.parent.postMessage({ type: "IFRAME_READY" }, "*");

// =========================
// RPC reply helpers (ENV_*)
// =========================
function rpcReply(type, requestId, payload) {
  window.parent.postMessage({ type, requestId, payload }, "*");
}
function rpcError(requestId, error) {
  window.parent.postMessage({ type: "RPC_ERROR", requestId, error: String(error) }, "*");
}

function isFloat32Array(x) {
  return x && x.constructor && x.constructor.name === "Float32Array";
}

const MJJNT_FREE = 0;
const MJJNT_BALL = 1;
const MJJNT_SLIDE = 2;
const MJJNT_HINGE = 3;

window.addEventListener("message", async (event) => {
  try {
    const msg = event.data || {};
    const { type, requestId, payload } = msg;

    // =========================
    // RL ENV RPC
    // =========================
    if (type === "ENV_CONFIG") {
      if (!requestId) return;

      if (!viewer?.model || !viewer?.simulation) {
        throw new Error("ENV_CONFIG: scene not loaded");
      }

      const jointOrder = payload?.jointOrder ?? [];
      viewer.rl.jointOrder = jointOrder;
      viewer.rl.substeps = payload?.substeps ?? viewer.rl.substeps;
      viewer.rl.kp = payload?.kp ?? viewer.rl.kp;
      viewer.rl.kd = payload?.kd ?? viewer.rl.kd;

      viewer.rl.obsLayout = payload?.obsLayout ?? "ROOT+Q+QD+PHASE";
      viewer.rl.withFeet = String(viewer.rl.obsLayout).includes("FEET");

      if (payload?.foot) {
        if (payload.foot.leftBodyHint) viewer.rl.foot.leftBodyHint = payload.foot.leftBodyHint;
        if (payload.foot.rightBodyHint) viewer.rl.foot.rightBodyHint = payload.foot.rightBodyHint;
        if (payload.foot.contactZ != null) viewer.rl.foot.contactZ = payload.foot.contactZ;
        if (payload.foot.contactVz != null) viewer.rl.foot.contactVz = payload.foot.contactVz;
      }

      viewer.rl.enabled = true;

      viewer.params.paused = true;
      viewer.updateDragMode();

      viewer._rlRebuildIndex();
      viewer._rlResolveFeetBodies();

      rpcReply("ENV_CONFIG_RESULT", requestId, { ok: true });
      return;
    }

    if (type === "ENV_DISABLE") {
      if (!requestId) return;

      viewer.rl.enabled = false;
      viewer.updateDragMode();

      rpcReply("ENV_DISABLE_RESULT", requestId, { ok: true });
      return;
    }

    if (type === "ENV_RESET") {
      if (!requestId) return;

      if (!viewer?.model || !viewer?.simulation) throw new Error("ENV_RESET: scene not loaded");
      if (!viewer.rl.enabled) throw new Error("ENV_RESET: RL not enabled (call ENV_CONFIG first)");

      const qRef = payload?.qRef;
      const qdRef = payload?.qdRef;
      const phase = payload?.phase ?? 0;

      if (!isFloat32Array(qRef)) throw new Error("ENV_RESET: payload.qRef must be Float32Array");

      viewer._rlResetToReference(qRef, qdRef);

      const obs = viewer._rlBuildObs(phase);
      rpcReply("ENV_RESET_RESULT", requestId, { obs, simTime: viewer.simulation.time });
      return;
    }

    if (type === "ENV_STEP") {
      if (!requestId) return;

      if (!viewer?.model || !viewer?.simulation) throw new Error("ENV_STEP: scene not loaded");
      if (!viewer.rl.enabled) throw new Error("ENV_STEP: RL not enabled (call ENV_CONFIG first)");

      const qTarget = payload?.qTarget;
      const phase = payload?.phase ?? 0;

      if (!isFloat32Array(qTarget)) throw new Error("ENV_STEP: payload.qTarget must be Float32Array");

      for (let i = 0; i < viewer.rl.substeps; i++) {
        viewer._rlApplyPD(qTarget);
        viewer.simulation.step();
      }

      const obs = viewer._rlBuildObs(phase);
      rpcReply("ENV_STEP_RESULT", requestId, { obs, simTime: viewer.simulation.time });
      return;
    }

    // =========================
    // Existing messages
    // =========================
    switch (type) {
      case "RESET_POSE":
        if (viewer?.model && viewer?.simulation) {
          resetToTPose(viewer.model, viewer.simulation, viewer.mujoco);
        }
        break;

      case "PAUSE_SIMULATION":
        if (viewer?.params) {
          viewer.params.paused = true;
          viewer.updateDragMode();
        }
        break;

      case "RESUME_SIMULATION":
        if (viewer?.rl?.enabled) break;

        if (viewer?.params) {
          viewer.params.paused = false;
          viewer.updateDragMode();
        }
        break;

      case "LOAD_SCENE": {
        removeAllMujocoRoots(viewer);

        const normalizedRoot = await stageMjcfSceneToVfs(mujoco, msg.root, {
          files: msg.files || [],
          xml: msg.xml || null,
        });

        [viewer.model, viewer.state, viewer.simulation, viewer.bodies, viewer.lights] =
          await loadSceneFromURL(mujoco, normalizedRoot, viewer);

        viewer.jointDragManager.simulation = viewer.simulation;
        viewer.jointDragManager.model = viewer.model;

        viewer.jointMap = {};

        resetToTPose(viewer.model, viewer.simulation, viewer.mujoco);

        if (viewer?.params) {
          viewer.params.paused = true;
          viewer.updateDragMode();
        }

        viewer.rl.enabled = false;
        viewer.rl.jointOrder = [];
        viewer.rl.qposAdr = [];
        viewer.rl.dofAdr = [];
        viewer.rl.withFeet = false;
        viewer.rl.foot.leftBodyId = -1;
        viewer.rl.foot.rightBodyId = -1;

        window.parent.postMessage({ type: "SCENE_LOADED", sceneName: normalizedRoot }, "*");

        viewer._debugListBodiesContains("foot");
        viewer._debugListBodiesContains("ank");
        break;
      }

      case "GET_JOINT_INFO": {
        if (!viewer.model || !viewer.simulation) return;

        viewer.jointMap = {};

        const njnt = getNjntRobust(viewer.model);
        console.log("Model Loaded. NJNT:", njnt);

        for (let i = 0; i < njnt; i++) {
          const jType = getI32(viewer.model.jnt_type, i, viewer.mujoco);
          if (!(jType === MJJNT_SLIDE || jType === MJJNT_HINGE)) continue;

          let name = "";
          try {
            name = getJointNameRobust(viewer.model, viewer.mujoco, i);
          } catch (e) {
            console.warn(`Failed to extract name for joint ${i}`, e);
          }

          const finalName = name && name.length > 0 ? name : `joint_${i}`;
          viewer.jointMap[finalName] = new Joint(viewer.simulation, viewer.model, i, viewer.mujoco);
        }

        viewer.syncToReact(performance.now());
        break;
      }

      case "CONTROL_JOINT": {
        if (viewer?.rl?.enabled) break;

        const { jointName, value } = msg;
        if (viewer.jointMap && viewer.jointMap[jointName]) {
          viewer.jointMap[jointName].value = value;
          if (viewer.params.paused) viewer.simulation.forward();
        }
        break;
      }

      case "SET_JOINT_TARGETS_PD": {
        if (viewer?.rl?.enabled) break;

        const { enabled = true, targets, kp = 80, kd = 6 } = msg;

        if (!enabled || !targets) {
          viewer.pdDrive.enabled = false;
          viewer.pdDrive.targets = {};
          break;
        }

        viewer.pdDrive.enabled = true;
        viewer.pdDrive.targets = targets;
        viewer.pdDrive.kp = kp;
        viewer.pdDrive.kd = kd;
        break;
      }

      default:
        break;
    }
  } catch (error) {
    console.error("❌ Error handling message:", error);

    if (event?.data?.requestId) {
      rpcError(event.data.requestId, error?.message ?? String(error));
      return;
    }

    window.parent.postMessage({ type: "ERROR", error: String(error) }, "*");
  }
});
