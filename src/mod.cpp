// Skirt Physics - Dusklight native mod
//
// Adds hat-style secondary motion (momentum + wind) to Link's outer skirt, driven through the
// eight skirt bones that already exist in his skeleton:
//
//   joint 27 fskirtL1   28 fskirtL2   29 fskirtR1   30 fskirtR2
//         31 rskirtL1   32 rskirtL2   33 rskirtR1   34 rskirtR2
//
// Vanilla only drives joints 27/29 (leg IK) and otherwise leaves the skirt to baked animation.
// This mod keeps all of that and layers a small spring-damper simulation on top:
//
//   * a per-frame hook after daAlink_c::setHatAngle() advances the simulation, and
//   * a post-hook on daAlink_c::modelCallBack() applies the result to each skirt joint using the
//     same call vanilla uses for the leg-IK skirt rotation (setMatrixWorldAxisRot).
//
// Every panel can only swing OUTWARD from its rest pose (angle >= 0), never inward, so the
// skirt can never fold into the legs or the chainmail layer beneath it.

#include "mods/service.hpp"
#include "mods/svc/hook.hpp"
#include "mods/svc/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// Game includes
#include "d/actor/d_a_alink.h"
#include "d/d_kankyo_wether.h"

DEFINE_MOD();

IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(HookService, svc_hook);

DEFINE_HOOK(&daAlink_c::setHatAngle, LinkSetHatAngle);
DEFINE_HOOK(&daAlink_c::modelCallBack, LinkModelCallBack);
DEFINE_HOOK(&daAlink_c::jointControll, LinkJointControll);

namespace {

// ---------------------------------------------------------------------------------------------
// Tunables. Angles are radians, time is one game frame. Start here when tuning the feel.
// ---------------------------------------------------------------------------------------------
struct Tuning {
    // Rest pose spring and damping, upper (L1) and lower (L2) segment of each panel.
    float spring1 = 0.060f;
    float damp1 = 0.180f;
    float spring2 = 0.080f;
    float damp2 = 0.200f;

    // How strongly each input pushes the skirt outward (radians per frame^2 per unit input).
    float velGain = 0.0011f;    // movement velocity: hem trails behind the direction of travel
    float accGain = 0.0070f;    // acceleration: hem lags on speed-up, swings forward on braking
    float windGain = 0.0017f;   // Link's own wind push (gusts/fans that move him) along the wind direction
    float ambientGain = 0.040f; // ambient breeze (what blows his hat/hair): force per unit wind power
    float fallGain = 0.0060f;   // falling: air lifts the hem on every panel
    float gravGain = 0.0200f;   // body tilt (flips, rolls, leaning): hem hangs along world gravity
    float hemLen = 20.0f;       // length of the tracked hem proxy point below the hip

    // Lower segment follows the upper one's motion with a lag.
    float chainGain = 0.35f;

    // Outward-only limits (radians). Angles never go below 0 = never swing inward.
    float max1 = 0.55f;  // ~31 deg
    float max2 = 0.45f;  // ~26 deg

    // Skip the simulation when speed or wind is tiny so the skirt rests perfectly still.
    float teleportDist = 300.0f;  // a jump larger than this in one frame resets the simulation
};

constexpr Tuning kTune{};

constexpr int kFirstJoint = 27;
constexpr int kPanelCount = 4;     // front-left, front-right, rear-left, rear-right
constexpr int kJointCount = 8;     // two segments per panel

// Joint order in the skeleton: L1, L2 of FL; L1, L2 of FR; L1, L2 of RL; L1, L2 of RR.
// Outward direction of each panel in Link's local frame (x = Link's left, z = forward).
struct PanelDir {
    float nx;
    float nz;
};
constexpr float kInvSqrt2 = 0.70710678f;
constexpr PanelDir kPanel[kPanelCount] = {
    {+kInvSqrt2, +kInvSqrt2},  // fskirtL  (front left)
    {-kInvSqrt2, +kInvSqrt2},  // fskirtR  (front right)
    {+kInvSqrt2, -kInvSqrt2},  // rskirtL  (rear left)
    {-kInvSqrt2, -kInvSqrt2},  // rskirtR  (rear right)
};

// Set to false after calibrating if you want a quiet log.
constexpr bool kDebugLog = true;

struct PanelState {
    float a = 0.0f;   // upper segment outward angle
    float av = 0.0f;  // upper segment angular velocity
    float b = 0.0f;   // lower segment outward angle relative to the upper one
    float bv = 0.0f;
};

PanelState g_panel[kPanelCount];
bool g_havePrev = false;
float g_prevPos[3] = {0, 0, 0};
float g_prevVelL[3] = {0, 0, 0};  // previous local velocity (x, y, z)
int g_logCounter = 0;
unsigned g_cbCalls = 0;      // modelCallBack post-hook hits on skirt joints
unsigned g_jcCalls = 0;      // jointControll post-hook hits on skirt joints
unsigned g_applied = 0;      // rotations actually applied
bool g_jcDone[8] = {};       // jointControll already applied this joint (avoid double apply)

constexpr float kBamPerRad = 32768.0f / 3.14159265f;

void reset_state() {
    for (PanelState& p : g_panel) {
        p = PanelState{};
    }
    g_havePrev = false;
    g_prevVelL[0] = g_prevVelL[1] = g_prevVelL[2] = 0.0f;
}

void log_msg(const char* msg) {
    svc_log->info(mod_ctx, msg);
}

// Advance one panel's two-segment chain by one frame given the pushing force along its outward
// direction. `parentVel` is how fast the upper segment moved this frame (for the lag chain).
void step_panel(PanelState& p, float drive, float fall) {
    const float force = drive + fall;

    // Upper segment.
    p.av += force - kTune.spring1 * p.a - kTune.damp1 * p.av;
    p.a += p.av;
    if (p.a < 0.0f) {  // outward only
        p.a = 0.0f;
        p.av = std::max(p.av, 0.0f);
    } else if (p.a > kTune.max1) {
        p.a = kTune.max1;
        p.av = std::min(p.av, 0.0f);
    }

    // Lower segment: pushed by the same force, dragged along by the upper segment's motion.
    p.bv += 0.8f * force - kTune.chainGain * p.av - kTune.spring2 * p.b - kTune.damp2 * p.bv;
    p.b += p.bv;
    // The panel as a whole (a + b) must also never swing inward.
    const float minB = -p.a;
    const float maxB = kTune.max2;
    if (p.b < minB) {
        p.b = minB;
        p.bv = std::max(p.bv, 0.0f);
    } else if (p.b > maxB) {
        p.b = maxB;
        p.bv = std::min(p.bv, 0.0f);
    }
}

// World position of a joint's origin, from the current animated model matrices.
bool world_joint(daAlink_c* link, int joint, float out[3]) {
    if (link->mpLinkModel == nullptr) {
        return false;
    }
    cXyz local;
    cXyz world;
    mDoMtx_multVecZero(link->mpLinkModel->getAnmMtx(joint), &local);
    mDoMtx_multVec(link->mpLinkModel->getBaseTRMtx(), &local, &world);
    out[0] = world.x;
    out[1] = world.y;
    out[2] = world.z;
    return true;
}

void norm3(float v[3]) {
    const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-5f) {
        v[0] /= l;
        v[1] /= l;
        v[2] /= l;
    }
}

float dot3(const float a[3], const float b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Runs once per game frame (after the vanilla hat/hair update).
void on_set_hat_angle_post(ModContext*, void* args, void*, void*) {
    daAlink_c* link = mods::arg<daAlink_c*>(args, 0);
    if (link == nullptr) {
        return;
    }

    // Body frame from the animated skeleton so flips, rolls and leaning are seen by the simulation:
    // left = right hip -> left hip, up = hips -> waist, forward = left x up (x = left, y = up, z = forward).
    float pw[3], hl[3], hr[3];
    const float yawRad = static_cast<float>(link->shape_angle.y) / kBamPerRad;
    float left[3] = {std::cos(yawRad), 0.0f, -std::sin(yawRad)};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float fwd[3] = {std::sin(yawRad), 0.0f, std::cos(yawRad)};
    float px = link->current.pos.x;
    float py = link->current.pos.y;
    float pz = link->current.pos.z;
    if (!link->checkWolf() && world_joint(link, 16, pw) && world_joint(link, 17, hl) &&
        world_joint(link, 22, hr)) {
        for (int k = 0; k < 3; k++) {
            left[k] = hl[k] - hr[k];
            up[k] = pw[k] - 0.5f * (hl[k] + hr[k]);
        }
        norm3(left);
        norm3(up);
        const float d = dot3(left, up);
        for (int k = 0; k < 3; k++) {
            left[k] -= d * up[k];
        }
        norm3(left);
        fwd[0] = left[1] * up[2] - left[2] * up[1];
        fwd[1] = left[2] * up[0] - left[0] * up[2];
        fwd[2] = left[0] * up[1] - left[1] * up[0];
        // Tracked hem proxy: a point hanging below the hip along the body axis.
        px = pw[0] - up[0] * kTune.hemLen;
        py = pw[1] - up[1] * kTune.hemLen;
        pz = pw[2] - up[2] * kTune.hemLen;
    }

    if (!g_havePrev) {
        g_prevPos[0] = px;
        g_prevPos[1] = py;
        g_prevPos[2] = pz;
        g_havePrev = true;
        return;
    }

    const float wx = px - g_prevPos[0];
    const float wy = py - g_prevPos[1];
    const float wz = pz - g_prevPos[2];
    g_prevPos[0] = px;
    g_prevPos[1] = py;
    g_prevPos[2] = pz;

    if (std::sqrt(wx * wx + wz * wz) > kTune.teleportDist || std::fabs(wy) > kTune.teleportDist) {
        reset_state();
        g_prevPos[0] = px;
        g_prevPos[1] = py;
        g_prevPos[2] = pz;
        g_havePrev = true;
        return;
    }

    // Low-pass the world velocity: the tracked hip point also carries animation wobble (strafing,
    // lock-on footwork), which would otherwise show up as skirt vibration. g_prevVelL holds the
    // filtered WORLD velocity here.
    const float kVelFilter = 0.30f;
    float fv[3], acw[3];
    const float wv[3] = {wx, wy, wz};
    for (int k = 0; k < 3; k++) {
        fv[k] = g_prevVelL[k] + (wv[k] - g_prevVelL[k]) * kVelFilter;
        float a_ = fv[k] - g_prevVelL[k];
        acw[k] = std::clamp(a_, -2.5f, 2.5f);   // no violent spikes from teleports or animation snaps
        g_prevVelL[k] = fv[k];
    }
    const float velLx = dot3(fv, left);
    const float velLy = dot3(fv, up);
    const float velLz = dot3(fv, fwd);
    auto to_local_x = [&](float x, float z) { return x * left[0] + z * left[2]; };
    auto to_local_z = [&](float x, float z) { return x * fwd[0] + z * fwd[2]; };
    const float accLx = dot3(acw, left);
    const float accLz = dot3(acw, fwd);

    // Gravity in the body frame (smoothed): zero when upright, pushes panels outward when Link
    // tilts or flips.
    static float s_gravL = 0.0f, s_gravF = 0.0f;
    s_gravL += (-left[1] - s_gravL) * 0.25f;
    s_gravF += (-fwd[1] - s_gravF) * 0.25f;
    const float gravL = s_gravL;
    const float gravF = s_gravF;

    // Wind. Link's mWindSpeed is only non-zero for strong gusts that physically push him, so the
    // ambient breeze that blows his hat and hair is read straight from the weather system, at his
    // hip, exactly like setHatAngle does. Both become a force along the wind direction.
    const cXyz& wind = link->getWindSpeed();
    cXyz ambPos(px, py + 60.0f, pz);
    cXyz ambDir(0.0f, 0.0f, 0.0f);
    f32 ambPow = 0.0f;
    dKyw_get_AllWind_vec(&ambPos, &ambDir, &ambPow);
    if (!(ambPow > 0.0f)) {
        ambPow = 0.0f;
    }
    ambPow = std::min(ambPow, 4.0f);  // strong gale cap so the skirt never goes wild
    // Steady wind alone only leans the skirt a few degrees; real breezes flutter, so modulate it.
    static unsigned s_tick = 0;
    s_tick++;
    const float t = static_cast<float>(s_tick);
    const float flutter = 1.0f + 0.35f * std::sin(t * 0.11f) + 0.20f * std::sin(t * 0.27f + 1.0f);
    const float ambK = ambPow * kTune.ambientGain * std::max(flutter, 0.2f);
    // dKyw_get_AllWind_vec points where the wind comes FROM, so the push is along the negated vector.
    const float forceX = wind.x * kTune.windGain - ambDir.x * ambK;
    const float forceZ = wind.z * kTune.windGain - ambDir.z * ambK;
    const float windLx = to_local_x(forceX, forceZ);
    const float windLz = to_local_z(forceX, forceZ);

    // Falling air lift (only when moving downward fast).
    const float fall = kTune.fallGain * std::clamp(-wy - 2.0f, 0.0f, 20.0f) * 0.1f;

    for (int i = 0; i < kPanelCount; i++) {
        const float nx = kPanel[i].nx;
        const float nz = kPanel[i].nz;

        const float velAlong = velLx * nx + velLz * nz;
        const float accAlong = accLx * nx + accLz * nz;
        const float windAlong = windLx * nx + windLz * nz;

        const float drive = -kTune.velGain * velAlong - kTune.accGain * accAlong +
                            windAlong + kTune.gravGain * (gravL * nx + gravF * nz);

        // Dead zone: ignore tiny drives so the skirt rests perfectly still instead of buzzing.
        const float kDead = 0.0012f;
        const float driveDz = drive > kDead ? drive - kDead : (drive < -kDead ? drive + kDead : 0.0f);
        step_panel(g_panel[i], driveDz, fall);
    }

    if constexpr (kDebugLog) {
        const float speed = std::sqrt(velLx * velLx + velLz * velLz);
        const float windMag = std::sqrt(wind.x * wind.x + wind.z * wind.z);
        if ((speed > 1.0f || windMag > 1.0f || ambPow > 0.05f) && ++g_logCounter >= 45) {
            g_logCounter = 0;
            char buf[260];
            std::snprintf(buf, sizeof(buf),
                "speed=%.2f wind=%.2f amb=%.2f angles(deg) FL=%.1f FR=%.1f RL=%.1f RR=%.1f hits cb=%u jc=%u applied=%u", speed,
                windMag, ambPow, g_panel[0].a * 57.2958f, g_panel[1].a * 57.2958f,
                g_panel[2].a * 57.2958f, g_panel[3].a * 57.2958f, g_cbCalls, g_jcCalls, g_applied);
            log_msg(buf);
        }
    }
}

// Applies the simulated angle to one skirt joint.
void apply_skirt_joint(daAlink_c* link, int joint) {
    if (link == nullptr || joint < kFirstJoint || joint >= kFirstJoint + kJointCount) {
        return;
    }
    if (link->checkWolf() || link->mpLinkModel == nullptr) {
        return;
    }

    const int idx = joint - kFirstJoint;  // 0..7
    const int panel = idx / 2;
    const bool lower = (idx % 2) != 0;
    const PanelState& st = g_panel[panel];
    const float angle = lower ? st.b : st.a;
    if (angle == 0.0f) {
        return;
    }

    // Tilt the hem toward the panel's outward direction n = (nx, nz): for a hem hanging below
    // the joint, rotating by theta about the vector (-nz, 0, nx) moves it along +n.
    const float theta = angle * kBamPerRad;
    const s16 rotX = static_cast<s16>(-kPanel[panel].nz * theta);
    const s16 rotZ = static_cast<s16>(kPanel[panel].nx * theta);

    link->setMatrixWorldAxisRot(link->mpLinkModel->getAnmMtx(joint), rotX, 0, rotZ, TRUE, nullptr);
    g_applied++;
}

// jointControll runs inside modelCallBack for every joint (vanilla's own 27/29 leg-IK skirt
// rotation lives there). Hooking it is the reliable path if modelCallBack itself is inlined.
void on_joint_controll_post(ModContext*, void* args, void*, void*) {
    daAlink_c* link = mods::arg<daAlink_c*>(args, 0);
    const int joint = mods::arg<int>(args, 1);
    if (joint < kFirstJoint || joint >= kFirstJoint + kJointCount) {
        return;
    }
    g_jcCalls++;
    apply_skirt_joint(link, joint);
    g_jcDone[joint - kFirstJoint] = true;
}

// Fallback: if jointControll already handled this joint in the same call, do nothing.
void on_model_callback_post(ModContext*, void* args, void*, void*) {
    daAlink_c* link = mods::arg<daAlink_c*>(args, 0);
    const int joint = mods::arg<int>(args, 1);
    if (joint < kFirstJoint || joint >= kFirstJoint + kJointCount) {
        return;
    }
    g_cbCalls++;
    bool& done = g_jcDone[joint - kFirstJoint];
    if (done) {
        done = false;
        return;
    }
    apply_skirt_joint(link, joint);
}

}  // namespace

extern "C" {
MOD_EXPORT ModResult mod_initialize(ModError*) {
    reset_state();

    ModResult result = mods::hook::add_post<LinkSetHatAngle>(on_set_hat_angle_post);
    if (result != MOD_OK) {
        svc_log->error(mod_ctx, "skirt_physics: failed to hook daAlink_c::setHatAngle");
        return result;
    }

    result = mods::hook::add_post<LinkJointControll>(on_joint_controll_post);
    if (result != MOD_OK) {
        svc_log->error(mod_ctx, "skirt_physics: failed to hook daAlink_c::jointControll");
        return result;
    }

    result = mods::hook::add_post<LinkModelCallBack>(on_model_callback_post);
    if (result != MOD_OK) {
        svc_log->error(mod_ctx, "skirt_physics: failed to hook daAlink_c::modelCallBack");
        return result;
    }

    log_msg("skirt_physics initialized");
    return MOD_OK;
}

MOD_EXPORT ModResult mod_update(ModError*) {
    return MOD_OK;
}

MOD_EXPORT ModResult mod_shutdown(ModError*) {
    reset_state();
    return MOD_OK;
}
}
