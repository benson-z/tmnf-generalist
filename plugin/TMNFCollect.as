// TMNFCollect - streams game frames, inputs and car telemetry to a Python
// controller over a TCP socket.
//
// Sampling is clocked on physics ticks (OnRunStep runs once per 10 ms of race
// time) rather than on rendered frames, so raising the game speed does not
// change what gets recorded. Screenshots are only legal inside Render(), so a
// sample point in OnRunStep stashes the tick's telemetry and calls
// Graphics::ForceGameRender(), which synchronously runs Render(), where the
// capture happens.
//
// Everything on the wire is little-endian.

const uint PROTO_VERSION = 3;

// plugin -> controller
const uint8 MSG_HELLO = 0x01;
const uint8 MSG_SAMPLE = 0x02;
const uint8 MSG_EVENT = 0x03;
const uint8 MSG_LOG = 0x04;
const uint8 MSG_PONG = 0x05;

// controller -> plugin
const uint8 CMD_COMMAND = 0x10;
const uint8 CMD_CONFIG = 0x11;
const uint8 CMD_PING = 0x12;
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
bool g_forceRender = false;
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
        } else if (kind == CMD_PING) {
            if (!g_sock.Write(MSG_PONG)) { Fail("pong"); return; }
            WriteString(payload);
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
    } else if (key == "force_render") {
        g_forceRender = (value == "1");
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
    if (raceTime % g_periodMs != 0) return;
    if (raceTime == g_lastSampleTime) return;
    g_lastSampleTime = raceTime;

    // Without forced rendering the game decides when it draws; if it has not
    // drawn since the previous sample point, that sample never became a frame.
    if (g_pending) g_dropped++;

    StashTelemetry(sim, raceTime);
    g_pending = true;
    if (g_forceRender) {
        // Synchronously runs Render(), where CaptureScreenshot is legal.
        Graphics::ForceGameRender();
    }
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
    // Render() is the only callback that keeps running in the menus, so this is
    // where the connection gets (re-)established between races.
    if (!EnsureConnected()) return;
    PollCommands();

    if (!g_pending) return;
    g_pending = false;
    if (!g_connected) return;

    vec2 size(float(g_capW), float(g_capH));
    array<uint8>@ pixels = Graphics::CaptureScreenshot(size);
    if (pixels is null) return;
    SendPendingSample(pixels, int(size.x), int(size.y));
}
