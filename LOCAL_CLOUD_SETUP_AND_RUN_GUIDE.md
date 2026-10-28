# FLZAT Local–Cloud Setup and Run Guide

This guide covers two use cases:

- **FFmpeg/FFplay:** display the Cloud desktop running Gazebo, Isaac Sim, or another GUI application on the Local Ubuntu machine.
- **Tailscale + ROS 2:** run the camera, microphone, and speaker on Local while STT, RAG, and Qwen3-VL run on Vast.ai Cloud.

## 1. View the Cloud desktop with FFplay

FFplay receives the Cloud desktop through SSH and operates independently of ROS 2.

### 1.1. Log in to Vast.ai

1. Open the [Vast.ai instances page](https://cloud.vast.ai/instances/).
2. Start the Cloud instance and wait until its status is **Running**.
3. Select **Connect**, copy the current SSH command, and run it in a Local terminal:

```bash
ssh -p 55640 root@202.215.0.218 -L 8080:localhost:8080
```

The initial Cloud SSH session uses the `root` account. Switch to the `user` account before starting graphical applications:

```bash
su - user
```

If the image asks for the `user` password, enter:

```text
1
```

Normally, `root` can run `su - user` without a password. Password `1` is only for this experimental Cloud image and must not be used in production.

All FLZAT application and ROS 2 commands on Cloud use the Conda environment
`py312`. After switching to `user`, initialize the Cloud environment as
follows:

```bash
cd <path>/Flzat_High_System_ws

export FLZAT_WS="$(pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /opt/ros/jazzy/setup.bash
source "$FLZAT_WS/install/setup.bash"
```

Run this initialization in every new Cloud terminal. System administration
commands such as `apt` and `tailscale` do not require the Conda environment.

### 1.2. Install the required packages

On Local:

```bash
sudo apt update
sudo apt install -y ffmpeg openssh-client
```

On Cloud:

```bash
sudo apt update
sudo apt install -y ffmpeg x11-utils x11-apps mesa-utils
```

### 1.3. Check the Cloud desktop

After switching to `user`, run:

```bash
export DISPLAY=:0
export XAUTHORITY=/home/user/.Xauthority-flzat
xdpyinfo | grep dimensions
```

Record the displayed resolution, for example `1024x768`.

### 1.4. Stream the desktop to Local

Open two terminals:

- **Terminal 1 — Local:** receives and displays the Cloud desktop with FFplay.
- **Terminal 2 — Cloud:** switches to `user` and then starts Gazebo or Isaac Sim. This can be another SSH session or the Vast.ai web terminal.

#### Terminal 1 — Local

The current Cloud SSH endpoint is `202.215.0.218:55640`. Run this command on Local:

```bash
ssh -T -o Compression=no -p 55640 root@202.215.0.218 \
  'runuser -u user -- env \
    HOME=/home/user \
    DISPLAY=:0 \
    XAUTHORITY=/home/user/.Xauthority-flzat \
    ffmpeg -nostdin -hide_banner -loglevel error \
      -f x11grab -video_size 1024x768 -framerate 30 -i :0.0 \
      -an -c:v h264_nvenc -preset p1 -tune ll \
      -b:v 4M -maxrate 4M -bufsize 1M -g 30 -bf 0 \
      -f mpegts -' \
| ffplay -hide_banner -fflags nobuffer -flags low_delay \
    -framedrop -an -i pipe:0
```

Keep this terminal running so FFplay continues displaying the desktop. Replace `1024x768` with the result from `xdpyinfo`. Press `Ctrl+C` to stop the stream.

> **Note:** Run the stream command once in Terminal 1 on Local and leave it running. Start any application you want to display on Cloud; every window opened on desktop `:0` will appear in FFplay. You do not need to restart FFplay for each Gazebo, Isaac Sim, or other application session.

#### Terminal 2 — Cloud

If the terminal is running as `root`, switch to `user`, configure the desktop environment, and start the application:

```bash
su - user
# If prompted for a password, enter: 1

cd <path>/Flzat_High_System_ws
export FLZAT_WS="$(pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /opt/ros/jazzy/setup.bash
source "$FLZAT_WS/install/setup.bash"

export DISPLAY=:0
export XAUTHORITY=/home/user/.Xauthority-flzat
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus

# Choose one command:
xclock
gz sim -v 4 shapes.sdf

# Or start Isaac Sim:
# cd <path>/isaacsim
# ./isaac-sim.sh
```

The Cloud window will appear in FFplay on Local. FFplay only streams the image; Gazebo and Isaac Sim commands still run in Terminal 2 on Cloud.

## 2. Set up and connect Tailscale

```text
Ubuntu Local                         Vast.ai Cloud
camera_node ─ /vision/get_frame_near ─► Cognitive Manager
vad_node ──── /audio_events ──────────► STT → RAG → Qwen3-VL
tts_node ◄─── /tts_action ───────────── JSON / speech
```

1. Open the [Tailscale Quickstart](https://tailscale.com/docs/how-to/quickstart) and sign in to create a tailnet.
2. Install Tailscale on **both Local and Cloud**:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

3. Connect both machines using the same Tailscale account:

```bash
# Local
sudo tailscale up --hostname=flzat-local

# Cloud
sudo tailscale up --hostname=flzat-vast-cloud
```

Open the printed URL to authenticate, and then check the connection:

```bash
tailscale status
tailscale ip -4
```

| Machine | Tailscale IP |
|---|---|
| Ubuntu Local | `<LOCAL_TAILSCALE_IP>` |
| Vast.ai Cloud | `<CLOUD_TAILSCALE_IP>` |

Run this command on each machine to obtain its current IP:

```bash
tailscale ip -4
```

Replace the placeholders below with those values. Do not use the public Vast.ai SSH address as `ROS_STATIC_PEERS`.

Test connectivity in both directions:

```bash
# Local
export CLOUD_TS_IP="<CLOUD_TAILSCALE_IP>"
tailscale ping "$CLOUD_TS_IP"

# Cloud
export LOCAL_TS_IP="<LOCAL_TAILSCALE_IP>"
tailscale ping "$LOCAL_TS_IP"
```

When either machine or the Vast.ai instance changes, run `tailscale ip -4` again and update the corresponding variable below.

## 3. Run camera and voice components on Local

```bash
cd <path>/flzat_high_system_ws

export FLZAT_WS="$(pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /opt/ros/jazzy/setup.bash
source "$FLZAT_WS/install/setup.bash"

unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=10
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export CLOUD_TS_IP="<CLOUD_TAILSCALE_IP>"
export ROS_STATIC_PEERS="$CLOUD_TS_IP"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 daemon stop
ros2 daemon start
```

Start the camera, VAD, and TTS:

```bash
ros2 launch flzat_high_bringup local_io.launch.py \
  run_camera_node:=true \
  run_vad_node:=true \
  run_tts_node:=true \
  camera_index:=0 \
  mic_device:=sysdefault \
  camera_width:=640 \
  camera_height:=480 \
  camera_fps:=15.0 \
  camera_buffer_duration_s:=35.0 \
  debug_fps:=0.0 \
  piper_model_path:="$FLZAT_WS/models/tts/en_US-lessac-medium.onnx"
```

Local now provides `/audio_events`, `/vision/get_frame_near`, and `/tts_action`.

## 4. Run the High System on Cloud

The Cloud runtime uses the Conda environment `py312`. In every new Cloud
terminal, initialize Conda, activate `py312`, and then source ROS 2 and the
workspace in that order. This ensures that ROS 2 nodes use the Python packages
installed in `py312`.

```bash
cd <path>/Flzat_High_System_ws

export FLZAT_WS="$(pwd)"
export FLZAT_MEMORY_ROOT="$FLZAT_WS"
export FLZAT_OUTPUT_ROOT="$FLZAT_WS/runtime_output"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /opt/ros/jazzy/setup.bash
source "$FLZAT_WS/install/setup.bash"

unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=10
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export LOCAL_TS_IP="<LOCAL_TAILSCALE_IP>"
export ROS_STATIC_PEERS="$LOCAL_TS_IP"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 daemon stop
ros2 daemon start
```

Verify that Cloud can discover the Local components:

```bash
ros2 node list
ros2 service list | grep vision
```

The output should include `/camera_node`, `/vad_node`, `/tts_node`, and `/vision/get_frame_near`.

Start STT, RAG, Cognitive Manager, and Qwen3-VL:

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=true \
  use_fake_speech:=false \
  use_fake_vlm:=false \
  use_low_mock:=true \
  run_camera_node:=false \
  run_vad_node:=false \
  run_stt_node:=true \
  run_tts_node:=false \
  frame_source:=camera \
  whisper_model_path:="$FLZAT_WS/models/base.en.pt" \
  model_id:=Qwen/Qwen3-VL-2B-Instruct \
  device:=cuda \
  dtype:=bfloat16 \
  quantization:=none \
  max_new_tokens:=256
```

## 5. Test the pipeline

Speak English into the Local microphone, or publish a text command from either machine:

```bash
ros2 topic pub --once \
  /flzat/high_system/text_command \
  std_msgs/msg/String \
  "{data: 'get the book on the table'}"
```

Display the final JSON command:

```bash
cd "$FLZAT_WS"
python3 tools/pretty_command_listener.py
```

Check the ROS interfaces:

```bash
ros2 action list | grep -E 'stt|tts|vlm'
ros2 topic info /audio_events --verbose
ros2 service type /vision/get_frame_near
```

Both machines must use the same `ROS_DOMAIN_ID`, RMW implementation, and `flzat_high_interfaces` version. Their clocks must also be synchronized; check them with `timedatectl status`.

## 6. Switch from Local–Cloud mode to Local-only mode

You do not need to stop or uninstall Tailscale. Follow these steps so ROS 2 does not continue using the previous Cloud discovery configuration:

1. Press `Ctrl+C` to stop `high_system.launch.py` on Cloud.
2. Press `Ctrl+C` to stop `local_io.launch.py` on Local.
3. Open a new Local terminal and switch ROS discovery to Local-only mode:

```bash
cd <path>/flzat_high_system_ws

export FLZAT_WS="$(pwd)"
export FLZAT_MEMORY_ROOT="$FLZAT_WS"
export FLZAT_OUTPUT_ROOT="$FLZAT_WS/runtime_output"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py312
source /opt/ros/jazzy/setup.bash
source "$FLZAT_WS/install/setup.bash"

unset ROS_LOCALHOST_ONLY
unset ROS_STATIC_PEERS
export ROS_DOMAIN_ID=10
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 daemon stop
ros2 daemon start

echo "ROS_STATIC_PEERS='$ROS_STATIC_PEERS'"
```

`ROS_STATIC_PEERS` must be empty. You can now run the entire pipeline on Local. Use the fake VLM on a machine that cannot run Qwen3-VL:

```bash
ros2 launch flzat_high_bringup high_system.launch.py \
  storage_root:="$FLZAT_MEMORY_ROOT" \
  environment_id:=demo_1 \
  map_id:=demo_1_test \
  output_root:="$FLZAT_OUTPUT_ROOT" \
  embedding_backend:=sentence_transformer \
  enable_text_command:=true \
  use_fake_speech:=false \
  use_fake_vlm:=true \
  use_low_mock:=true \
  run_camera_node:=true \
  run_vad_node:=true \
  run_stt_node:=true \
  run_tts_node:=true \
  frame_source:=camera \
  whisper_model_path:="$FLZAT_WS/models/base.en.pt" \
  piper_model_path:="$FLZAT_WS/models/tts/en_US-lessac-medium.onnx"
```

If Local has enough GPU capacity for the real Qwen model, change `use_fake_vlm:=true` to `false` and provide the model and device parameters. Do not run `local_io.launch.py` at the same time as the command above because that would duplicate the camera, VAD, TTS, service, and action nodes.

To return to Local–Cloud mode, set `ROS_STATIC_PEERS` to the peer machine's current Tailscale IP on both machines and restart the ROS daemon.

## 7. Common issues

- **FFplay does not open:** check `DISPLAY`, `XAUTHORITY`, the desktop resolution, and the `h264_nvenc` encoder.
- **Tailscale ping works but ROS nodes are missing:** check `ROS_DOMAIN_ID`, `ROS_STATIC_PEERS`, and `RMW_IMPLEMENTATION`, and then restart the ROS daemon.
- **Only `/rosout` is visible:** run `unset ROS_LOCALHOST_ONLY` on both machines.
- **No transcript is produced:** check `/audio_events`, `/stt_action`, and `models/base.en.pt` on Cloud.
- **The camera frame is unavailable:** check `/vision/get_frame_near` and clock synchronization.
- **TTS produces no sound:** check `/tts_action` and the Local audio device.

You can run FFplay, `local_io.launch.py`, `high_system.launch.py`, and the JSON listener in separate terminals.
