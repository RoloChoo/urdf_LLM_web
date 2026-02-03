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
  // fieldOrPtr: either pointer(number) or Int32Array-like
  if (isPtrNumber(fieldOrPtr)) {
    return mujocoInstance.HEAP32[(fieldOrPtr >> 2) + index];
  }
  return fieldOrPtr[index];
}

function getNum(fieldOrPtr, index, mujocoInstance) {
  // mjtNum is typically double => HEAPF64 for pointer case
  // If it's a TypedArray, just index it.
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
  // name_jntadr: int offsets into `names`
  const offset = getI32(model.name_jntadr, jointId, mujocoInstance);

  // names: either pointer(number) to heap, or Uint8Array-like view
  if (isPtrNumber(model.names)) {
    return readCStringFromHeapU8(mujocoInstance.HEAPU8, model.names + offset);
  }
  return readCStringFromU8Array(model.names, offset);
}

function getNjntRobust(model) {
  // Some embind builds expose `njnt` as a number, some as a function/getter wrapper.
  return typeof model.njnt === "function" ? model.njnt() : model.njnt;
}

function getNbodyRobust(model) {
  return typeof model.nbody === "function" ? model.nbody() : model.nbody;
}

function getNlightRobust(model) {
  return typeof model.nlight === "function" ? model.nlight() : model.nlight;
}

// =========================================================
// Joint Class (Wrapper for Joint Control) - robust arrays
// =========================================================
class Joint {
  constructor(simulation, model, jointId, mujocoInstance) {
    this.simulation = simulation;
    this.model = model;
    this.mj = mujocoInstance;
    this.id = jointId;

    // Address in qpos array
    this.qposAddr = getI32(model.jnt_qposadr, jointId, this.mj);

    // Joint Limits (Range) [min, max]
    // jnt_range is a flat array, stride is 2
    this.min = getNum(model.jnt_range, 2 * jointId, this.mj);
    this.max = getNum(model.jnt_range, 2 * jointId + 1, this.mj);

    // If both are 0, it might be an unlimited joint (e.g., continuous).
    // For UI sliders, assign a default visualization range.
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

    // Theme defaults; can be overridden by parent messages
    this.theme = {
      sceneBg: "#fef4da",
      floor: "#fcf4dc",
      ambient: "#fcf4dc",
      hemi: "#fcf4dc",
    };
    this.scene.background = new THREE.Color(this.theme.sceneBg);

    // Centralized fill lights based on default theme
    this._createFillLights();

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "default",
    });

    // Cap device pixel ratio for performance on high-DPI displays
    const MAX_PIXEL_RATIO = 1.5;
    this.renderer.setPixelRatio(
      Math.min(MAX_PIXEL_RATIO, window.devicePixelRatio)
    );

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

    // Joint drag manager for manipulating joints when simulation is paused
    this.jointDragManager = new JointDragManager(
      this.scene,
      this.renderer,
      this.camera,
      this.container.parentElement,
      this.controls,
      null // Will be set when simulation is loaded
    );

    // Initially enable joint drag manager (starts paused) and disable physics drag
    this.jointDragManager.enable();
    this.dragStateManager.disable();

    // Add hover highlighting functionality
    this.hoverRaycaster = new THREE.Raycaster();
    this.hoveredBody = null;
    this.mousePos = new THREE.Vector2();
    this.originalMaterials = new Map(); // Store original materials for restoration
    this.highlightColor = new THREE.Color(0xfbe651);

    this.renderer.domElement.addEventListener(
      "mousemove",
      this.onMouseMove.bind(this)
    );
    this.renderer.domElement.addEventListener(
      "mouseleave",
      this.onMouseLeave.bind(this)
    );
  }

  _createFillLights() {
    if (this.ambientLight) this.scene.remove(this.ambientLight);
    if (this.hemiLight) this.scene.remove(this.hemiLight);

    this.ambientLight = new THREE.AmbientLight(
      new THREE.Color(this.theme.ambient),
      0.2
    );
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
    const intersects = this.hoverRaycaster.intersectObjects(
      this.scene.children,
      true
    );

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
        window.parent.postMessage(
          { type: "BODY_MOUSEOUT", bodyName: this.hoveredBody.name },
          "*"
        );
      }

      this.hoveredBody = newHoveredBody;

      if (this.hoveredBody) {
        this.applyBodyHighlight(this.hoveredBody);
        window.parent.postMessage(
          { type: "BODY_MOUSEOVER", bodyName: this.hoveredBody.name },
          "*"
        );
      }
    }
  }

  applyBodyHighlight(bodyGroup) {
    bodyGroup.traverse((child) => {
      if (child.isMesh && child.material) {
        // Some materials may not have emissive (e.g., MeshBasicMaterial)
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
      window.parent.postMessage(
        { type: "BODY_MOUSEOUT", bodyName: this.hoveredBody.name },
        "*"
      );
      this.hoveredBody = null;
    }
  }

  updateDragMode() {
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
    this.renderer.setPixelRatio(
      Math.min(MAX_PIXEL_RATIO, window.devicePixelRatio)
    );
  }

  // Sync state to Parent (React)
  syncToReact(timeMS) {
    if (!this.jointMap || Object.keys(this.jointMap).length === 0) return;

    const jointData = {};
    for (const [name, joint] of Object.entries(this.jointMap)) {
      jointData[name] = {
        id: joint.id,
        val: joint.value,
        min: joint.min,
        max: joint.max,
      };
    }

    const delta = timeMS - (this.lastFrameTime || timeMS - 16);
    const fps = delta > 0 ? Math.round(1000 / delta) : 60;

    window.parent.postMessage(
      {
        type: "SYNC_STATE",
        payload: {
          joints: jointData,
          time: this.simulation ? this.simulation.time : 0,
          fps: fps,
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

    if (!this.params["paused"]) {
      let timestep = this.model.getOptions().timestep;
      if (timeMS - this.mujoco_time > 35.0) {
        this.mujoco_time = timeMS;
      }
      while (this.mujoco_time < timeMS) {
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate = Math.exp(
            -timestep / Math.max(1e-10, this.params["ctrlnoiserate"])
          );
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

        this.simulation.step();
        this.mujoco_time += timestep * 1000.0;
      }
    } else {
      if (!this.jointDragManager.active) {
        this.simulation.forward();
      }
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

window.addEventListener("message", async (event) => {
  try {
    switch (event.data.type) {
      case "RESET_POSE":
        if (viewer?.simulation) {
          viewer.simulation.resetData();
          viewer.simulation.forward();
        }
        break;

      case "PAUSE_SIMULATION":
        if (viewer?.params) {
          viewer.params.paused = true;
          viewer.updateDragMode();
        }
        break;

      case "RESUME_SIMULATION":
        if (viewer?.params) {
          viewer.params.paused = false;
          viewer.updateDragMode();
        }
        break;

      case "LOAD_SCENE": {
        removeAllMujocoRoots(viewer);

        const normalizedRoot = await stageMjcfSceneToVfs(
          mujoco,
          event.data.root,
          {
            files: event.data.files || [],
            xml: event.data.xml || null,
          }
        );

        [
          viewer.model,
          viewer.state,
          viewer.simulation,
          viewer.bodies,
          viewer.lights,
        ] = await loadSceneFromURL(mujoco, normalizedRoot, viewer);

        viewer.jointDragManager.simulation = viewer.simulation;
        viewer.jointDragManager.model = viewer.model;

        viewer.jointMap = {};

        viewer.simulation.resetData();
        viewer.simulation.forward();

        if (viewer?.params) {
          viewer.params.paused = true;
          viewer.updateDragMode();
        }

        window.parent.postMessage(
          { type: "SCENE_LOADED", sceneName: normalizedRoot },
          "*"
        );
        break;
      }

      // ----------------------------------------------------
      // Get Joint Info (ROBUST: pointer OR TypedArray)
      // ----------------------------------------------------
      case "GET_JOINT_INFO": {
        if (!viewer.model || !viewer.simulation) return;

        viewer.jointMap = {};

        const njnt = getNjntRobust(viewer.model);
        console.log("Model Loaded. NJNT:", njnt);

        // Debug types (optional; helps a lot)
        console.log(
          "name_jntadr:",
          typeof viewer.model.name_jntadr,
          viewer.model.name_jntadr?.constructor?.name
        );
        console.log(
          "names:",
          typeof viewer.model.names,
          viewer.model.names?.constructor?.name
        );
        console.log(
          "jnt_qposadr:",
          typeof viewer.model.jnt_qposadr,
          viewer.model.jnt_qposadr?.constructor?.name
        );
        console.log(
          "jnt_range:",
          typeof viewer.model.jnt_range,
          viewer.model.jnt_range?.constructor?.name
        );

        for (let i = 0; i < njnt; i++) {
          let name = "";
          try {
            name = getJointNameRobust(viewer.model, viewer.mujoco, i);
          } catch (e) {
            console.warn(`Failed to extract name for joint ${i}`, e);
          }

          const finalName = name && name.length > 0 ? name : `joint_${i}`;
          console.log(`Joint[${i}] Mapping: ${finalName}`);

          viewer.jointMap[finalName] = new Joint(
            viewer.simulation,
            viewer.model,
            i,
            viewer.mujoco
          );
        }

        viewer.syncToReact(performance.now());
        break;
      }

      // ----------------------------------------------------
      // Control Joint (User Manipulation)
      // ----------------------------------------------------
      case "CONTROL_JOINT": {
        const { jointName, value } = event.data;
        if (viewer.jointMap && viewer.jointMap[jointName]) {
          viewer.jointMap[jointName].value = value;
          if (viewer.params.paused) {
            viewer.simulation.forward();
          }
        }
        break;
      }

      default:
        console.warn("Unknown message type:", event.data.type);
    }
  } catch (error) {
    console.error("❌ Error handling message:", error);
    window.parent.postMessage(
      { type: "ERROR", error: String(error) },
      "*"
    );
  }
});
