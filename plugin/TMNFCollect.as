// TMNFCollect - streams game frames, inputs and car telemetry to a Python
// controller over a TCP socket.
//
// Sampling is clocked on physics ticks (OnRunStep runs once per 10 ms of race
// time) rather than on rendered frames. Screenshots are only legal inside
// Render(), so a sample point in OnRunStep stashes the tick's telemetry and
// the game's next natural Render() captures it. Above 1x, a sample state is
// held until that callback, then simulation resumes. This keeps the regular
// camera/render path while physics runs quickly between samples.
//
// Everything on the wire is little-endian.

const uint PROTO_VERSION = 5;

// plugin -> controller
const uint8 MSG_HELLO = 0x01;
const uint8 MSG_SAMPLE = 0x02;
const uint8 MSG_EVENT = 0x03;
// One per physics tick: inputs at 100Hz, without a frame attached.
const uint8 MSG_TICK = 0x06;

// controller -> plugin
const uint8 CMD_COMMAND = 0x10;
const uint8 CMD_CONFIG = 0x11;
const uint8 CMD_FOCUS = 0x14;

// event kinds
const uint8 EV_RUN_START = 1;
const uint8 EV_CHECKPOINT = 2;
const uint8 EV_FINISH = 3;
const uint8 EV_GAMESTATE = 4;
const uint8 EV_RUN_RESET = 5;

Net::Socket@ g_sock = null;
bool g_connected = false;
uint64 g_nextConnectAttempt = 0;

uint16 g_port = 8477;
int g_instanceId = 0;
string g_token = "";

// Runtime config; the controller overrides these with CMD_CONFIG.
bool g_collecting = false;
int g_periodMs = 50; // 20 Hz in race time
int g_capW = 320;
int g_capH = 240;
// Training frames want the bare game view, without the speedometer,
// clock and checkpoint widgets drawn over it.
bool g_hideUi = true;

uint g_seq = 0;
uint g_dropped = 0;      // ticks whose frame never got rendered
int g_lastTickTime = 0;  // most recently simulated tick, for alignment
int g_lastSampleTime = -1000000;
int g_prevRaceTime = -1000000;
bool g_prevFinished = false;

// Telemetry for the sample Render() is about to capture.
bool g_pending = false;
// Above 1x, several physics ticks can run inside one game-loop iteration. At a
// sample tick, hold this exact state until the next regular Render() callback.
// The frame is therefore natural (including its camera update), but simulation
// is free to run quickly between sample points.
bool g_frameBarrier = false;
bool g_frameHeld = false;
// The hold is a speed of 0, the way Linesight pauses the game, rather than
// Running=false: after Running=false the game still ran one tick, which then
// had to be rewound, and re-applying inputs around that rewind was not exact.
// The run's speed comes from the controller so it can be put back.
float g_speed = 1.0f;
// Ticks that ran anyway while held: should stay at zero.
int g_heldTicks = 0;
int p_raceTime = 0;
uint p_displaySpeed = 0;
float p_velX = 0, p_velY = 0, p_velZ = 0;
float p_posX = 0, p_posY = 0, p_posZ = 0;
float p_yaw = 0, p_pitch = 0, p_roll = 0;
float p_localX = 0, p_localY = 0, p_localZ = 0;
float p_gasF = 0, p_brakeF = 0, p_steerF = 0;
bool p_up = false, p_down = false, p_left = false, p_right = false;
int p_gas = 0, p_steer = 0;
uint p_checkpoints = 0;
bool p_finished = false;
bool p_sliding = false;
int p_gearbox = 0;
float p_camX = 0, p_camY = 0, p_camZ = 0;
float p_camYaw = 0, p_camPitch = 0, p_camRoll = 0;
float p_camFov = 0;

PluginInfo@ GetPluginInfo()
{
    PluginInfo@ info = PluginInfo();
    info.Name = "TMNFCollect";
    info.Author = "tmnf-collect";
    info.Version = "0.1.0";
    info.Description = "Streams 20 Hz frames, inputs and telemetry to a data collection controller.";
    return info;
}

void Main()
{
    ParseCommandLine();
    log("TMNFCollect: instance " + g_instanceId + " -> 127.0.0.1:" + g_port, Severity::Info);
}

void OnDisabled()
{
    if (g_frameHeld) { GetSimulationManager().SetSpeed(g_speed); g_frameHeld = false; }
    Disconnect();
}

// --------------------------------------------------------------- arguments

// TMLoader passes its whole trailing argument string to the game as a single
// quoted argument, so the switches have to be looked up inside the joined
// command line rather than matched against individual argv entries.
string ArgValue(const string&in commandLine, const string&in key)
{
    int at = commandLine.FindFirst(key);
    if (at < 0) return "";
    uint start = uint(at) + key.Length;
    int end = commandLine.FindFirstOf(" \t\"", start);
    return end < 0
        ? commandLine.Substr(start)
        : commandLine.Substr(start, end - int(start));
}

void ParseCommandLine()
{
    const array<string>@ args = IO::GetCommandLineArgs();
    string commandLine = "";
    for (uint i = 0; i < args.Length; i++) {
        if (i > 0) commandLine += " ";
        commandLine += args[i];
    }

    string port = ArgValue(commandLine, "/tmnfml_port=");
    if (port != "") g_port = uint16(Text::ParseUInt(port));

    string id = ArgValue(commandLine, "/tmnfml_id=");
    if (id != "") g_instanceId = int(Text::ParseInt(id));

    g_token = ArgValue(commandLine, "/tmnfml_token=");
}

// -------------------------------------------------------------- connection

bool EnsureConnected()
{
    if (g_connected) return true;

    uint64 now = Time::Now;
    if (now < g_nextConnectAttempt) return false;
    g_nextConnectAttempt = now + 1000;

    // TMInterface 2.2.1 reports false from Connect even when the TCP connection
    // is established, so the handshake write below is what actually decides
    // whether we are connected.
    Net::Socket@ sock = Net::Socket();
    sock.Connect("127.0.0.1", g_port, 500);

    @g_sock = sock;
    g_connected = true;
    g_seq = 0;
    SendHello();
    return g_connected;
}

void Disconnect()
{
    @g_sock = null;
    g_connected = false;
    g_pending = false;
}

void Fail(const string&in why)
{
    // Losing the controller is routine (it exits between batches) and the
    // connect retry runs once a second, so this stays quiet on purpose.
    Disconnect();
}

void WriteString(const string&in s)
{
    if (!g_connected) return;
    if (!g_sock.Write(uint16(s.Length))) { Fail("string length"); return; }
    if (s.Length > 0 && !g_sock.Write(s)) { Fail("string body"); }
}

void SendHello()
{
    bool ok = g_sock.Write(MSG_HELLO)
        && g_sock.Write(uint(PROTO_VERSION))
        && g_sock.Write(int(g_instanceId))
        && g_sock.Write(uint(IO::GetCurrentProcessId()));
    if (!ok) { Disconnect(); return; }
    WriteString(g_token);

    // Report the state the game is already in. OnGameStateChanged only fires
    // on a transition, and the game often reaches the menu before the plugin
    // has connected, so a controller waiting for that transition would
    // otherwise wait for one that already happened.
    SendEvent(EV_GAMESTATE, 0, int(GetCurrentGameState()));
}

// ----------------------------------------------------------------- inbound

void PollCommands()
{
    if (!g_connected) return;
    while (g_connected && g_sock.Available >= 3) {
        uint8 kind = g_sock.ReadUint8();
        uint16 length = g_sock.ReadUint16();
        string payload = length > 0 ? g_sock.ReadString(length) : "";

        if (kind == CMD_COMMAND) {
            ExecuteCommand(payload);
        } else if (kind == CMD_FOCUS) {
            Graphics::FocusGameWindow();
        } else if (kind == CMD_CONFIG) {
            ApplyConfig(payload);
        }
    }
}

// Config arrives as "key=value;key=value" so it stays readable in logs.
void ApplyConfig(const string&in payload)
{
    uint start = 0;
    while (start <= payload.Length) {
        int end = payload.FindFirst(";", start);
        string part = end < 0 ? payload.Substr(start) : payload.Substr(start, end - int(start));
        ApplyConfigEntry(part);
        if (end < 0) break;
        start = uint(end) + 1;
    }
}

void ApplyConfigEntry(const string&in entry)
{
    int eq = entry.FindFirst("=");
    if (eq < 0) return;
    string key = entry.Substr(0, eq);
    string value = entry.Substr(uint(eq) + 1);

    if (key == "collect") {
        if (g_collecting && value != "1") log("TMNFCollect: ticks run while held: " + g_heldTicks);
        g_heldTicks = 0;
        g_collecting = (value == "1");
        g_lastSampleTime = -1000000;
        ApplyRaceInterface();
        // Per run, not per game session: the controller reads this back as
        // "how many sample points did *this* run lose".
        if (g_collecting) {
            g_dropped = 0;
            g_pending = false;
        }
    } else if (key == "period") {
        g_periodMs = Math::Max(10, int(Text::ParseInt(value)));
    } else if (key == "width") {
        g_capW = int(Text::ParseInt(value));
    } else if (key == "height") {
        g_capH = int(Text::ParseInt(value));
    } else if (key == "frame_barrier") {
        g_frameBarrier = (value == "1");
    } else if (key == "speed") {
        g_speed = Text::ParseFloat(value);
    } else if (key == "hide_ui") {
        g_hideUi = (value == "1");
        ApplyRaceInterface();
    }
}

// The race interface comes back on its own across map loads, so this is
// re-applied whenever collection is armed and again once a run starts.
void ApplyRaceInterface()
{
    ToggleRaceInterface(!(g_hideUi && g_collecting));
}

// ---------------------------------------------------------------- outbound

void SendEvent(uint8 kind, int raceTime, int arg)
{
    if (!g_connected) return;
    if (!g_sock.Write(MSG_EVENT)) { Fail("event"); return; }
    g_sock.Write(kind);
    g_sock.Write(raceTime);
    g_sock.Write(arg);
}

// Inputs change far faster than 20Hz: a keyboard tap can be a single 10ms tick,
// which the sampled stream either misses or lands on the wrong frame. Ticks are
// sent for every step of the simulation and carry no image, so a whole run of
// them costs less than one frame.
void SendTick(SimulationManager@ sim, int raceTime)
{
    if (!g_connected) return;
    InputState inputs = sim.GetInputState();

    uint8 keys = 0;
    if (inputs.Up) keys |= 1;
    if (inputs.Down) keys |= 2;
    if (inputs.Left) keys |= 4;
    if (inputs.Right) keys |= 8;

    // Two writes, not eight. Every failed write disconnects the plugin for
    // good, so the fewer of them per tick the better -- and at 100Hz this runs
    // 800 times a second per instance if it is not kept lean. The analog
    // values and speed are in the 20Hz samples already; what is only available
    // here is which keys were down.
    if (!g_sock.Write(MSG_TICK)) { Fail("tick"); return; }
    g_sock.Write(raceTime);
    g_sock.Write(keys);
}

void SendPendingSample(array<uint8>@ pixels, int width, int height)
{
    if (!g_connected) return;
    if (!g_sock.Write(MSG_SAMPLE)) { Fail("sample"); return; }
    g_sock.Write(uint(g_seq));
    g_sock.Write(int(p_raceTime));
    g_sock.Write(uint(p_displaySpeed));

    g_sock.Write(p_velX); g_sock.Write(p_velY); g_sock.Write(p_velZ);
    g_sock.Write(p_posX); g_sock.Write(p_posY); g_sock.Write(p_posZ);
    g_sock.Write(p_yaw); g_sock.Write(p_pitch); g_sock.Write(p_roll);
    g_sock.Write(p_localX); g_sock.Write(p_localY); g_sock.Write(p_localZ);

    g_sock.Write(uint8(p_up ? 1 : 0));
    g_sock.Write(uint8(p_down ? 1 : 0));
    g_sock.Write(uint8(p_left ? 1 : 0));
    g_sock.Write(uint8(p_right ? 1 : 0));
    g_sock.Write(int(p_gas));
    g_sock.Write(int(p_steer));
    g_sock.Write(p_gasF); g_sock.Write(p_brakeF); g_sock.Write(p_steerF);

    g_sock.Write(uint(p_checkpoints));
    g_sock.Write(uint8(p_finished ? 1 : 0));
    g_sock.Write(uint8(p_sliding ? 1 : 0));
    g_sock.Write(int(p_gearbox));

    g_sock.Write(p_camX); g_sock.Write(p_camY); g_sock.Write(p_camZ);
    g_sock.Write(p_camYaw); g_sock.Write(p_camPitch); g_sock.Write(p_camRoll);
    g_sock.Write(p_camFov);

    // The tick the telemetry came from, and the tick the game had reached
    // when this frame was actually drawn. With natural rendering those can
    // differ, so the gap is recorded instead of assumed to be zero.
    g_sock.Write(int(g_lastTickTime));
    g_sock.Write(uint(g_dropped));

    g_sock.Write(uint16(width));
    g_sock.Write(uint16(height));
    g_sock.Write(uint(pixels.Length));
    if (!g_sock.Write(pixels)) { Fail("pixels"); return; }
    g_seq++;
}

// ------------------------------------------------------------ race sampling

void StashTelemetry(SimulationManager@ sim, int raceTime)
{
    p_raceTime = raceTime;

    TM::PlayerInfo@ player = sim.PlayerInfo;
    p_displaySpeed = player.DisplaySpeed;
    p_checkpoints = player.CurCheckpointCount;
    p_finished = player.RaceFinished;

    vec3 velocity = sim.Dyna.CurrentState.LinearSpeed;
    p_velX = velocity.x; p_velY = velocity.y; p_velZ = velocity.z;
    vec3 position = sim.Dyna.CurrentState.Location.Position;
    p_posX = position.x; p_posY = position.y; p_posZ = position.z;
    sim.Dyna.CurrentState.Location.Rotation.GetYawPitchRoll(p_yaw, p_pitch, p_roll);

    TM::SceneVehicleCar@ car = sim.SceneVehicleCar;
    vec3 local = car.CurrentLocalSpeed;
    p_localX = local.x; p_localY = local.y; p_localZ = local.z;
    p_gasF = car.InputGas;
    p_brakeF = car.InputBrake;
    p_steerF = car.InputSteer;
    p_sliding = car.IsSliding;
    p_gearbox = car.GearboxState;

    // The chase camera has its own frame-paced smoothing, so record where it
    // actually was rather than assuming it follows the car exactly.
    TM::GameCamera@ camera = GetCurrentCamera();
    if (camera !is null) {
        vec3 cameraPos = camera.Location.Position;
        p_camX = cameraPos.x; p_camY = cameraPos.y; p_camZ = cameraPos.z;
        camera.Location.Rotation.GetYawPitchRoll(p_camYaw, p_camPitch, p_camRoll);
        p_camFov = camera.Fov;
    }

    InputState inputs = sim.GetInputState();
    p_up = inputs.Up; p_down = inputs.Down;
    p_left = inputs.Left; p_right = inputs.Right;
    p_gas = inputs.Gas; p_steer = inputs.Steer;
}

void OnRunStep(SimulationManager@ sim)
{
    if (g_frameHeld && g_pending) g_heldTicks++;
    if (!EnsureConnected()) return;

    int raceTime = sim.RaceTime;
    g_lastTickTime = raceTime;

    // A restart rewinds race time; treat that as the end of the previous run.
    if (raceTime < g_prevRaceTime) {
        SendEvent(EV_RUN_RESET, raceTime, 0);
        g_lastSampleTime = -1000000;
        g_prevFinished = false;
    }
    if (g_prevRaceTime < 0 && raceTime >= 0) {
        SendEvent(EV_RUN_START, raceTime, 0);
        ApplyRaceInterface();
    }
    g_prevRaceTime = raceTime;

    bool finished = sim.PlayerInfo.RaceFinished;
    if (finished && !g_prevFinished) {
        SendEvent(EV_FINISH, raceTime, int(sim.PlayerInfo.CurCheckpointCount));
    }
    g_prevFinished = finished;

    if (!g_collecting || raceTime < 0) return;
    // Every tick carries inputs; only every g_periodMs-th one carries a frame.
    SendTick(sim, raceTime);

    if (raceTime % g_periodMs != 0) return;
    if (raceTime == g_lastSampleTime) return;
    g_lastSampleTime = raceTime;

    // The game decides when it draws; if it has not drawn since the previous
    // sample point, that sample never became a frame.
    if (g_pending) g_dropped++;

    // Every sample point also snaps the chase camera to the car. The game
    // smooths it on wall-clock time, so without this the frame for a given
    // car state depends on frame pacing and is not reproducible.
    if (g_frameBarrier) {
        // A rewind to the state the game is already in changes no physics.
        // What it does is drop the ticks the loop had still queued for this
        // iteration, so that speed 0 takes effect now rather than a few
        // ticks late (Linesight pauses the same way). A rewind also drops
        // the input state, so this tick's own inputs go back in; those are
        // the values already in effect, not a shifted transition.
        InputState inputs = sim.GetInputState();
        sim.RewindToState(sim.SaveState(), true);
        sim.SetInputState(InputType::Gas, inputs.Gas);
        sim.SetInputState(InputType::Steer, inputs.Steer);
        sim.SetInputState(InputType::Up, inputs.Up ? 1 : 0);
        sim.SetInputState(InputType::Down, inputs.Down ? 1 : 0);
        sim.SetInputState(InputType::Left, inputs.Left ? 1 : 0);
        sim.SetInputState(InputType::Right, inputs.Right ? 1 : 0);
        sim.SetSpeed(0.0f);
        g_frameHeld = true;
    } else {
        sim.ResetCamera();
    }
    StashTelemetry(sim, raceTime);
    g_pending = true;
}

int g_lastCpCount = -1;
int g_lastCpTime = -1;

void OnCheckpointCountChanged(SimulationManager@ sim, int current, int target)
{
    // While a replay is being validated this callback keeps firing with the
    // final checkpoint, so only report genuine changes.
    int raceTime = sim.RaceTime;
    if (current == g_lastCpCount && raceTime == g_lastCpTime) return;
    g_lastCpCount = current;
    g_lastCpTime = raceTime;
    SendEvent(EV_CHECKPOINT, raceTime, current);
}

void OnGameStateChanged(TM::GameState state)
{
    SendEvent(EV_GAMESTATE, 0, int(state));
}

void Render()
{
    if (g_frameHeld) {
        GetSimulationManager().SetSpeed(g_speed);
        g_frameHeld = false;
    }
    // Render() is the only callback that keeps running in the menus, so this is
    // where the connection gets (re-)established between races.
    if (!EnsureConnected()) return;
    PollCommands();

    if (!g_pending) return;
    g_pending = false;
    if (!g_connected) return;

    TM::GameCamera@ shotCamera = GetCurrentCamera();
    if (shotCamera !is null) {
        vec3 shotPos = shotCamera.Location.Position;
        p_camX = shotPos.x; p_camY = shotPos.y; p_camZ = shotPos.z;
        shotCamera.Location.Rotation.GetYawPitchRoll(p_camYaw, p_camPitch, p_camRoll);
        p_camFov = shotCamera.Fov;
    }

    vec2 size(float(g_capW), float(g_capH));
    array<uint8>@ pixels = Graphics::CaptureScreenshot(size);
    if (pixels is null) return;
    SendPendingSample(pixels, int(size.x), int(size.y));
}
