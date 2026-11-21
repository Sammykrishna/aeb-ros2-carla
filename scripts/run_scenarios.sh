#!/bin/bash
# usage: scripts/run_scenarios.sh CCRs_50 CCRb_50_d6 ...   (env: WATCH=1 for the chase-cam view, RECORD=1 for bags)
cd "$(dirname "$0")/.." || exit 1
source /opt/ros/jazzy/setup.bash && source .venv/bin/activate && source install/setup.bash

record=false; [ -n "$RECORD" ] && record=true
spectator=false; pause_s=0.0; [ -n "$WATCH" ] && { spectator=true; pause_s=5.0; }

# switching maps after yolo is on the gpu briefly holds both maps in vram and has crashed the 6 GB card
python -c "
import carla
c = carla.Client('localhost', 2000); c.set_timeout(120)
if not c.get_world().get_map().name.endswith('Town04'): c.load_world('Town04')
"

for s in "$@"; do
  echo "=== $s"
  timeout 150 ros2 launch aeb_bringup aeb.launch.py scenario:=$s record:=$record spectator:=$spectator end_pause_s:=$pause_s 2>&1 \
    | grep -i -E "carla_scenario\]: .*(PASS|COLLISION|INVALID)|Traceback|\[ERROR\]|RuntimeError"
done
