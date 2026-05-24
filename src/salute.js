/**
 * salute.js – Military salute activity for Unitree G1
 *
 * Actuator index map (position controllers, ctrl = target rad):
 *  0  left_hip_pitch       6  right_hip_pitch
 *  1  left_hip_roll        7  right_hip_roll
 *  2  left_hip_yaw         8  right_hip_yaw
 *  3  left_knee            9  right_knee
 *  4  left_ankle_pitch    10  right_ankle_pitch
 *  5  left_ankle_roll     11  right_ankle_roll
 * 12  waist_yaw           13  waist_roll          14  waist_pitch
 * 15  left_shoulder_pitch 16  left_shoulder_roll  17  left_shoulder_yaw  18  left_elbow
 * 19  left_wrist_roll     20  left_wrist_pitch    21  left_wrist_yaw
 * 22–28 left hand fingers
 * 29  right_shoulder_pitch 30 right_shoulder_roll 31 right_shoulder_yaw 32 right_elbow
 * 33  right_wrist_roll     34 right_wrist_pitch   35 right_wrist_yaw
 * 36–42 right hand fingers
 */

// ─── Pose definitions (sparse – only joints that differ from 0) ──────────────

/** Attention stance – stand upright, arms at sides */
const POSE_ATTENTION = {
  // legs straight, slight ankle compensation
  4:  -0.05,   // left  ankle pitch
  10: -0.05,   // right ankle pitch
  // arms hang naturally
  15:  0.05,   // left  shoulder pitch  (tiny forward)
  16: -0.15,   // left  shoulder roll   (arm close to body)
  29:  0.05,   // right shoulder pitch
  30:  0.15,   // right shoulder roll
};

/**
 * Right-hand military salute
 *  – right upper arm raises forward-up to ~90 °
 *  – elbow bends so forearm is diagonal toward the forehead
 *  – wrist flat (palm faces down/forward)
 *  – left arm stays at side
 *  – slight head-level waist pitch (optional subtle lean)
 */
const POSE_SALUTE = {
  // legs unchanged
  4:  -0.05,
  10: -0.05,

  // right arm – raise into salute
  29: -1.05,   // right shoulder pitch  (arm swings forward-up)
  30:  0.20,   // right shoulder roll   (arm slightly out)
  31:  0.30,   // right shoulder yaw    (rotate so palm faces in)
  32:  1.75,   // right elbow           (bend ~100 °, forearm up to forehead)
  33:  0.00,   // right wrist roll      (flat)
  34: -0.20,   // right wrist pitch     (slight downward tilt – fingertips toward brow)
  35:  0.10,   // right wrist yaw

  // right hand – fingers together, flat
  36:  0.10,   // right_hand_thumb_0
  37:  0.10,   // right_hand_thumb_1
  38:  0.10,   // right_hand_thumb_2
  39: -0.10,   // right_hand_middle_0
  40:  0.00,   // right_hand_middle_1
  41: -0.10,   // right_hand_index_0
  42:  0.00,   // right_hand_index_1

  // left arm – relaxed at side
  15:  0.05,
  16: -0.15,
};

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Total number of position actuators on the G1 (43 incl. fingers) */
const NUM_CTRL = 43;

/** Build a full ctrl array from a sparse pose object (all others = 0) */
function poseToCtrl(pose) {
  const ctrl = new Float64Array(NUM_CTRL);
  for (const [idx, val] of Object.entries(pose)) {
    ctrl[Number(idx)] = val;
  }
  return ctrl;
}

/** Linear interpolation between two ctrl arrays */
function lerpCtrl(a, b, t) {
  const out = new Float64Array(NUM_CTRL);
  for (let i = 0; i < NUM_CTRL; i++) {
    out[i] = a[i] + (b[i] - a[i]) * t;
  }
  return out;
}

/** Apply a ctrl array to MuJoCo data */
function applyCtrl(data, ctrl) {
  for (let i = 0; i < NUM_CTRL; i++) {
    data.ctrl[i] = ctrl[i];
  }
}

// ─── Salute Activity class ────────────────────────────────────────────────────

export class SaluteActivity {
  /**
   * @param {object} mujocoData   – live mujoco `data` object
   * @param {object} [options]
   * @param {number} [options.raiseDuration=1.2]   seconds to raise the arm
   * @param {number} [options.holdDuration=2.0]    seconds to hold the salute
   * @param {number} [options.lowerDuration=1.0]   seconds to lower the arm
   */
  constructor(mujocoData, options = {}) {
    this.data          = mujocoData;
    this.raiseDuration = options.raiseDuration ?? 1.2;
    this.holdDuration  = options.holdDuration  ?? 2.0;
    this.lowerDuration = options.lowerDuration ?? 1.0;

    this._ctrlAttention = poseToCtrl(POSE_ATTENTION);
    this._ctrlSalute    = poseToCtrl(POSE_SALUTE);

    this._phase    = 'idle';   // 'idle' | 'raise' | 'hold' | 'lower' | 'done'
    this._elapsed  = 0;
    this._active   = false;
    this._onDoneCb = null;
  }

  /** Start the salute sequence. Calls `onDone()` when finished. */
  start(onDone = null) {
    this._phase   = 'raise';
    this._elapsed = 0;
    this._active  = true;
    this._onDoneCb = onDone;
    console.log('[Salute] sequence started');
  }

  /** Stop immediately and reset to attention pose */
  stop() {
    this._active = false;
    this._phase  = 'idle';
    applyCtrl(this.data, this._ctrlAttention);
    console.log('[Salute] sequence stopped');
  }

  get isActive() { return this._active; }
  get phase()    { return this._phase;  }

  /**
   * Call this every simulation step.
   * @param {number} dt  – seconds elapsed since last call (use mujoco timestep × n_substeps)
   */
  update(dt) {
    if (!this._active) return;

    this._elapsed += dt;

    switch (this._phase) {

      case 'raise': {
        const t = Math.min(this._elapsed / this.raiseDuration, 1);
        applyCtrl(this.data, lerpCtrl(this._ctrlAttention, this._ctrlSalute, easeInOut(t)));
        if (t >= 1) {
          this._phase   = 'hold';
          this._elapsed = 0;
          console.log('[Salute] holding salute');
        }
        break;
      }

      case 'hold': {
        applyCtrl(this.data, this._ctrlSalute);
        if (this._elapsed >= this.holdDuration) {
          this._phase   = 'lower';
          this._elapsed = 0;
          console.log('[Salute] lowering arm');
        }
        break;
      }

      case 'lower': {
        const t = Math.min(this._elapsed / this.lowerDuration, 1);
        applyCtrl(this.data, lerpCtrl(this._ctrlSalute, this._ctrlAttention, easeInOut(t)));
        if (t >= 1) {
          this._phase  = 'done';
          this._active = false;
          applyCtrl(this.data, this._ctrlAttention);
          console.log('[Salute] sequence complete');
          if (this._onDoneCb) this._onDoneCb();
        }
        break;
      }
    }
  }
}

/** Smooth ease-in / ease-out curve (cubic Hermite) */
function easeInOut(t) {
  return t * t * (3 - 2 * t);
}
