/**
 * salute.js – Military salute activity for Unitree G1 (29-DOF + hands)
 *
 * Actuator index map (position controllers, ctrl = target rad):
 *  0  left_hip_pitch        6  right_hip_pitch
 *  1  left_hip_roll         7  right_hip_roll
 *  2  left_hip_yaw          8  right_hip_yaw
 *  3  left_knee             9  right_knee
 *  4  left_ankle_pitch     10  right_ankle_pitch
 *  5  left_ankle_roll      11  right_ankle_roll
 * 12  waist_yaw            13  waist_roll          14  waist_pitch
 * 15  left_shoulder_pitch  16  left_shoulder_roll  17  left_shoulder_yaw  18  left_elbow
 * 19  left_wrist_roll      20  left_wrist_pitch    21  left_wrist_yaw
 * 22  left_hand_thumb_0    23  left_hand_thumb_1   24  left_hand_thumb_2
 * 25  left_hand_middle_0   26  left_hand_middle_1
 * 27  left_hand_index_0    28  left_hand_index_1
 * 29  right_shoulder_pitch 30  right_shoulder_roll 31  right_shoulder_yaw 32  right_elbow
 * 33  right_wrist_roll     34  right_wrist_pitch   35  right_wrist_yaw
 * 36  right_hand_thumb_0   37  right_hand_thumb_1  38  right_hand_thumb_2
 * 39  right_hand_middle_0  40  right_hand_middle_1
 * 41  right_hand_index_0   42  right_hand_index_1
 */

// ─── helpers ──────────────────────────────────────────────────────────────────

/**
 * Build a full ctrl Float32Array (length = nActuator) from a sparse {index: rad} map.
 * Indices not in the map stay at 0.
 */
function buildCtrl(nActuator, sparse) {
  const arr = new Float32Array(nActuator);
  for (const [idx, val] of Object.entries(sparse)) {
    arr[Number(idx)] = val;
  }
  return arr;
}

/**
 * Linear-interpolate between two ctrl arrays element-wise.
 */
function lerpCtrl(a, b, t) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i] + (b[i] - a[i]) * t;
  }
  return out;
}

/**
 * Write a ctrl array into data.ctrl (handles TypedArray views).
 */
function applyCtrl(data, arr) {
  for (let i = 0; i < arr.length; i++) {
    data.ctrl[i] = arr[i];
  }
}

/** Smooth ease-in / ease-out (cubic Hermite) */
function easeInOut(t) {
  return t * t * (3 - 2 * t);
}

// ─── Pose definitions (sparse – only joints that differ from 0) ───────────────

/**
 * ATTENTION – stand upright, arms resting naturally at sides.
 */
const SPARSE_ATTENTION = {
  // subtle ankle pitch to keep CoM stable
  4:  -0.04,   // left  ankle pitch
  10: -0.04,   // right ankle pitch

  // left arm hangs close to body
  15:  0.08,   // left  shoulder pitch  (slight forward)
  16: -0.18,   // left  shoulder roll   (arm against torso)

  // right arm hangs close to body
  29:  0.08,   // right shoulder pitch
  30:  0.18,   // right shoulder roll

  // fingers slightly curled (relaxed hand)
  25:  0.10,   // left  middle_0
  26:  0.10,   // left  middle_1
  27:  0.10,   // left  index_0
  28:  0.10,   // left  index_1
  39:  0.10,   // right middle_0
  40:  0.10,   // right middle_1
  41:  0.10,   // right index_0
  42:  0.10,   // right index_1
};

/**
 * SALUTE – classic military right-hand salute.
 *
 * Right arm:
 *   shoulder pitches forward-up (~105°), elbow bends (~100°),
 *   wrist held flat so the palm faces slightly downward toward forehead.
 * Left arm: remains at attention side.
 * Waist: very slight backward pitch for an upright, proud posture.
 */
const SPARSE_SALUTE = {
  // ankles – same as attention
  4:  -0.04,
  10: -0.04,

  // very slight waist pitch – chest up, proud stance
  14: -0.04,   // waist pitch (negative = lean back slightly)

  // ── left arm stays at attention ─────────────────────────────────────────
  15:  0.08,   // left shoulder pitch
  16: -0.18,   // left shoulder roll

  // left fingers relaxed
  25:  0.10,
  26:  0.10,
  27:  0.10,
  28:  0.10,

  // ── right arm – raise into salute ───────────────────────────────────────
  29: -1.10,   // right shoulder pitch  (arm swings forward-upward, ~63° forward)
  30:  0.22,   // right shoulder roll   (arm slightly abducted)
  31:  0.25,   // right shoulder yaw    (rotate so palm faces inward/down)
  32:  1.85,   // right elbow           (forearm bends up ~106°)
  33: -0.20,   // right wrist roll      (flatten the palm)
  34: -0.15,   // right wrist pitch     (tip fingers slightly toward forehead)
  35:  0.10,   // right wrist yaw       (align hand with forearm)

  // right fingers FLAT / extended (military salute = flat hand)
  36:  0.05,   // right thumb_0  (slightly open)
  37:  0.05,   // right thumb_1
  38:  0.00,   // right thumb_2
  39:  0.00,   // right middle_0 – fully extended
  40:  0.00,   // right middle_1
  41:  0.00,   // right index_0  – fully extended
  42:  0.00,   // right index_1

  // left fingers relaxed during salute too
  25:  0.10,
  26:  0.10,
  27:  0.10,
  28:  0.10,
};

// ─── SaluteActivity class ─────────────────────────────────────────────────────

export class SaluteActivity {
  /**
   * @param {object} model   – MuJoCo model object
   * @param {object} data    – MuJoCo data object
   * @param {object} [opts]  – optional timing overrides (seconds)
   *   opts.raiseDuration  default 1.2 s
   *   opts.holdDuration   default 2.0 s
   *   opts.lowerDuration  default 1.0 s
   */
  constructor(model, data, opts = {}) {
    this.model = model;
    this.data  = data;

    this.raiseDuration  = opts.raiseDuration  ?? 1.2;
    this.holdDuration   = opts.holdDuration   ?? 2.0;
    this.lowerDuration  = opts.lowerDuration  ?? 1.0;

    const n = model.nu;   // total number of actuators
    this._ctrlAttention = buildCtrl(n, SPARSE_ATTENTION);
    this._ctrlSalute    = buildCtrl(n, SPARSE_SALUTE);

    this._phase    = 'idle';
    this._elapsed  = 0;
    this._active   = false;
    this._onDoneCb = null;

    // Immediately put robot in attention stance
    applyCtrl(this.data, this._ctrlAttention);
  }

  // ── public API ──────────────────────────────────────────────────────────────

  /** Begin the full salute sequence (raise → hold → lower → done). */
  start(onDone = null) {
    if (this._active) this.stop();
    this._phase    = 'raise';
    this._elapsed  = 0;
    this._active   = true;
    this._onDoneCb = onDone;
    console.log('[Salute] sequence started → raise');
  }

  /** Abort immediately and snap back to attention. */
  stop() {
    this._active = false;
    this._phase  = 'idle';
    applyCtrl(this.data, this._ctrlAttention);
    console.log('[Salute] sequence stopped');
  }

  /** True while any phase is running. */
  get isActive() { return this._active; }

  /** Current phase string: 'idle' | 'raise' | 'hold' | 'lower' | 'done' */
  get phase() { return this._phase; }

  /**
   * Drive the animation – call this every simulation step.
   * @param {number} dt  seconds since last call  (model.opt.timestep works fine)
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
          this._phase   = 'done';
          this._active  = false;
          applyCtrl(this.data, this._ctrlAttention);
          console.log('[Salute] sequence complete');
          if (this._onDoneCb) this._onDoneCb();
        }
        break;
      }
    }
  }
}
