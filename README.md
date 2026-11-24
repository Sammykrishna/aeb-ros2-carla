# Autonomous emergency braking on ROS 2 and CARLA

This project is an emergency braking system for a car, built from separate ROS 2 nodes and tested in the CARLA simulator with Euro NCAP style scenarios. A YOLO detector finds the car ahead in the camera image, a fusion node measures how far away it is using LiDAR, and a controller decides when to warn and when to brake. Every run can be recorded to a rosbag and checked afterwards.

I started with a CARLA project that worked in one big script and wanted to see what happens when it is split into proper ROS 2 pieces and held to honest measurements. The short version is that the first working version looked fine in a demo and hid several real problems. Most of this README is about finding those problems and what fixed them.

Everything here runs on one laptop with an RTX 4050 and 6 GB of VRAM, Ubuntu 24.04, ROS 2 Jazzy and CARLA 0.9.16.

## See it run

These clips are rebuilt from the camera frames recorded in the rosbags. The green box is YOLO, "AEB sees" is the range the fusion node reports, and the coloured strip and text show what the controller is doing. "True gap" comes from the simulator and is only there to check the AEB against.

In the first run the ego car drives at 50 km/h towards a parked car. The AEB starts braking at about 36 m and stops 11.7 m short.

![Ego car at 50 km/h approaching a stationary car](docs/media/pov_ccrs50.gif)

The second run is the hardest scenario I set up. The lead car is only 12 m ahead and brakes hard at 6 m/s2. The AEB reacts and cuts the impact speed from 50 to 25 km/h, but it does not avoid the crash.

![Lead car brakes hard 12 metres ahead](docs/media/pov_ccrb_d6.gif)

## How it works

```mermaid
flowchart LR
    carla[CARLA 0.9.16] --> bridge[carla_scenario_node]
    bridge -->|camera| detector[yolo_detector]
    bridge -->|LiDAR| fusion[aeb_fusion]
    detector -->|2D boxes| fusion
    fusion -->|3D objects| controller[aeb_controller]
    bridge -->|odometry| controller
    controller -->|brake command| bridge
```

There are four packages.

`aeb_detector` runs YOLO11s at 960 px and publishes 2D boxes. `aeb_fusion` projects the LiDAR points into each box and takes the depth of the nearest cluster, which gives a 3D position for the car. `aeb_controller` is a ROS 2 lifecycle node. It tracks the range over the last second, works out how fast the gap is closing, and decides what to do. `aeb_bringup` holds the launch file, the scenario definitions, the CARLA bridge that talks to the simulator, rosbag recording and the launch tests.

Safety critical commands use a reliable QoS profile and sensor streams use best effort with a queue of one, so a slow consumer never works from stale data. The controller also goes by the timestamp on each message, not the time it arrived, because queueing jitter at 14 m/s looks like metres of range noise.

### How the controller decides

The controller does not use fixed time-to-collision thresholds. Every cycle it asks how hard the car would have to brake right now to stop before the target, allowing for the response delay and a safety margin. It compares that with the most the car can actually deliver, which I measured at 4.5 m/s2 (the simulated brakes give about 0.67 + 3.83 times the brake command).

Below 30 percent of that limit it stays idle. Above it, it warns. From 65 percent it brakes with a pedal level that follows the demand, and from 90 percent, sustained for 0.3 s, it brakes fully. Two things stop it from reacting to noise: a braking state has to hold for a short time before it is entered, and a hard braking lead car only counts once the estimate is confirmed. If the threat goes away for a second the brake releases, and the brake level also decays slowly as the demand falls instead of staying at its peak.

## What the results say

I ran ten scenarios: seven collision scenarios in the style of Euro NCAP car-to-car rear tests, and three false-positive checks where the AEB must stay quiet. The table shows one run of each scenario at its fixed settings. The repeated runs further down are the better evidence, and they changed what I believed about two of these rows.

| Scenario | What happens | Result | First brake | Stops short by |
|---|---|---|---|---|
| Stationary target, 20 km/h | ego approaches a parked car | pass | 8.5 m | 7.0 m |
| Stationary target, 30 km/h | same | pass | 15 m | 8.9 m |
| Stationary target, 40 km/h | same | pass | 23 m | 11.2 m |
| Stationary target, 50 km/h | same | pass | 36 m | 11.7 m |
| Slow lead car | ego at 50, lead at 20 km/h | pass | 16 m | 3.2 m |
| Braking lead car | both at 50, lead brakes at 2 m/s2 | pass | 19 m | 4.6 m |
| Hard braking lead car | both at 50, 12 m gap, lead brakes at 6 m/s2 | **collision at 25 km/h** | 11 m | not avoided |
| Empty road | 12 s at 50 km/h, nothing ahead | pass, no brake | none | not applicable |
| Car in the next lane | ego drives past a parked car 3.5 m to the side | pass, no brake | none | not applicable |
| Close following | lead car 15 m ahead at the same speed | pass, no brake | none | not applicable |

Nine of the ten behave correctly. The false-positive checks are the part I would not skip: it is easy to build something that brakes in every test and call it safe.

### The detector was not the whole problem

My first detector was YOLOv8n at its default 640 px input. It only picked the car up at about 34 m. I benchmarked ten combinations of model and input size on the frames recorded from earlier runs, using the true gap from the simulator to measure how far away each model first saw the car.

![Detection range against inference time for ten detector configurations](docs/media/detector_tradeoff.png)

YOLO11s at 960 px sees the car from about 72 m and costs 7.7 ms per frame, only a couple of milliseconds more than where I started. The larger YOLOv8m reaches about the same range but takes 20 ms. A caveat: the benchmark used three clips of one car on one map, so treat the numbers as a guide and not a ranking.

### Fusion was quietly throwing the detections away

Even with the better detector, the AEB only started tracking the car at around 25 m. YOLO was detecting it in 100 percent of frames at every distance, but the fusion node was dropping most of those detections.

The reason is that at 40 m the car is only about 20 pixels wide, and fusion only used the middle half of the box and demanded at least three LiDAR points inside it. A 32 channel LiDAR often has fewer than that on a distant car. I replayed the recorded LiDAR and boxes through seven variants offline before touching the live system. Just loosening the thresholds made it worse, because stray road and background points pulled the median out by about a metre. Taking the full box and using the nearest LiDAR return as the range fixed both problems.

![Share of frames where fusion produced a range, before and after](docs/media/fusion_by_distance.png)

After the change fusion produces a range in 98 percent of frames at 35 to 45 m, up from 8 percent. Overall the range is there in 96 percent of frames and the average error is 0.04 m. My first version used the median of the nearest cluster for the range, which left an error of about 0.2 m. The median sits behind the bumper because it averages over the whole rear face of the car, and that bias mattered later, when a lead car braking gently was brought into focus by the repeated runs.

### The controller had its own problems

With perception fixed, the controller was the limit. The recorded bags showed that its estimate of the target's deceleration was far noisier than I assumed. When the true value was zero, the estimate had a standard deviation of 0.7 to 1.8 m/s2 and spiked above my 0.5 m/s2 threshold in up to 40 percent of samples. For a slow moving lead car the controller kept deciding the car ahead was braking and started braking itself about 16 m too early, and because the brake was latched at its peak it stayed there until it stopped.

The fixes were the ones described above: confirm the deceleration estimate before using it, require braking states to be sustained, let the brake release and decay, and widen the gap between the partial and full braking thresholds. I screened thresholds in an offline simulation of the real controller before spending simulator time, and then checked every change against CARLA.

![Distance left to the target when the car stopped, before and after the controller fixes](docs/media/stop_gaps.png)

The slow lead car now ends 3.2 m behind its target instead of 20.7 m, and the braking lead 4.6 m instead of 16.6 m.

Here are the two runs from the clips above as time series. On the left the brake comes on at once, then decays as the demand falls, and the final jump is the brake being held after the car has stopped. On the right the lead car starts braking at 1.0 s but the controller cannot see enough closing speed to act until about 1.6 s.

![Speed, gap and brake command over time for two scenarios](docs/media/timelines.png)

### Thirty varied runs per scenario

A single run proves very little, so I added a regression test that runs every scenario thirty times. Each run gets a seed that nudges the setup. The ego and lead speeds move by about 1 km/h, the starting gap by 5 percent, the target's sideways offset by 20 cm, the moment a lead car starts braking by 0.2 s, and the target is one of six car models. The seed makes the setup reproducible, but not the whole run, because the pipeline has its own timing noise.

![Stopping distance for thirty varied runs of each scenario](docs/media/regression_gaps.png)

| Scenario | Runs | Pass | Pass rate | 95% interval | Min gap, mean (range) | First brake, mean | Worst impact |
|---|---|---|---|---|---|---|---|
| CCRb_50_d2 | 30 | 30 | 100% | 89% to 100% | 9.2 m (3.3 to 12.2) | 23.0 m | none |
| CCRb_50_d6 | 30 | 0 | 0% | 0% to 11% | -0.0 m (-0.1 to 0.0) | 9.7 m | 33 km/h |
| CCRm_50_20 | 30 | 30 | 100% | 89% to 100% | 5.3 m (3.4 to 6.6) | 14.0 m | none |
| CCRs_20 | 30 | 30 | 100% | 89% to 100% | 6.8 m (4.4 to 7.5) | 7.9 m | none |
| CCRs_30 | 30 | 30 | 100% | 89% to 100% | 9.4 m (7.8 to 10.1) | 14.2 m | none |
| CCRs_40 | 30 | 30 | 100% | 89% to 100% | 7.6 m (4.2 to 11.4) | 22.5 m | none |
| CCRs_50 | 30 | 30 | 100% | 89% to 100% | 9.9 m (8.4 to 13.2) | 33.4 m | none |
| FP_adjacent | 30 | 30 | 100% | 89% to 100% | none | no brake | none |
| FP_follow | 30 | 30 | 100% | 89% to 100% | none | no brake | none |
| FP_none | 30 | 30 | 100% | 89% to 100% | none | no brake | none |

All 300 runs finished and every scenario met its expectation. Nine scenarios passed all thirty runs, including every false-positive check: 90 runs with no brake command on the empty road, in the next lane or while following closely. Thirty clean runs only show that the pass rate is at least about 89 percent, so a clean streak is good evidence and not a guarantee.

The hard braking lead is the exception, and the table shows 0 percent because the ego car hits the lead every time. Its goal was mitigation, an impact at or below 35 km/h, and it met that in all thirty runs. Impacts ranged from 23 to 34 km/h with a median of 29. The first brake comes at about 10 m with the ego car still at 50 km/h, because the tracker needs a moment of range data before it can tell the lead is slowing hard. It lowers the impact speed and does not avoid the crash.

The braking lead at 2 m/s2 used to be a coin flip, and it no longer is. In the earlier ten run batch it passed 9 of 10 and stopped anywhere from 0.9 m to 12 m short. Two small things stacked up. A lead braking at 2.0 m/s2 sat exactly on my 2.0 m/s2 threshold for believing the lead was braking, and the fused range came in bursts that made the braking estimate jump. Moving the range to the nearest LiDAR return and the threshold down to 1.5 m/s2 fixed it. All thirty runs now stop clear with a gap of 8 to 12 m, and the first brake comes at 21 to 24 m in all but one. The outlier was seed 21, where the first brake came late at 16 m and the car still stopped 3.3 m short of the lead. That is the closest call in the whole batch, and it is where I would look first if the pass rate ever slips.

Two next-lane runs crashed the harness before they began, because a Nissan Micra settled 0.3 to 0.4 m away from where I placed it and my start geometry check rejected it. That says nothing about the AEB. The placement now corrects itself by the measured error, and both seeds then ran and passed. CARLA also segfaulted once partway through the batch, and the runner resumed cleanly after a restart.

## What is still not good

The hard braking lead car still collides, at half the original speed. That one is limited by how quickly the tracker can see a lead car start to brake, so improving it needs earlier cues than range alone, such as brake lights or radar velocity.

The stationary target runs stop 7 to 12 m short. That is a deliberate trade. A car that brakes at 4.5 m/s2 needs about 21 m to stop from 50 km/h before any margin, and my offline simulation showed that braking any later collided under plausible model errors such as weaker brakes or extra delay. Real cars brake much harder, so a real system could wait longer.

Other limits worth knowing. Thirty runs per scenario still leaves an interval of about 11 percentage points. It is all simulation, with one car model, one map and no weather. False positives were only tested in three situations, and I have not tried pedestrians, curves, cut-ins, shadows or oncoming traffic. My offline simulation agrees with CARLA on when braking starts but is optimistic about stopping distance, so I only used it to screen options.

## Things I got wrong along the way

A few of the mistakes changed the results, so they are worth being open about.

One scenario reported a pass while the ground-truth gap was negative. The target was not actually in the ego car's lane, so the car drove past it. I replaced the guessed car length with a gap measured from the bounding boxes, placed the target along the ego heading, and made the harness fail loudly when the start geometry is off or when two cars overlap without a collision event.

Another early collision turned out to be a start-up race. YOLO takes about three seconds to produce its first result, and the scenario had already started. The detector now warms up before it subscribes, and the scenario waits until every node is ready.

I also believed for a while that the partial brake stage was over-braking, based on a deceleration fit that included the spikes at impact. Fitting again with medians showed the brake map is linear and the old mapping was about right. And I once narrowed the gap between the partial and full thresholds because my simulation said it was safe, which made the real controller jump to full braking within a fraction of a second on every stationary run.

The first batch of repeated runs found three problems that the single runs had hidden, and I fixed them before the final batch.

A Toyota Prius parked 83 m away made the fused range jump from 75 m to 68 m at long range, where the LiDAR is sparse. The tracker read that as a closing speed of 30 m/s, and the AEB braked at full strength 67 m from a stationary car. A target cannot close faster than the ego car is driving, so the controller now caps the closing speed at the ego speed plus one metre per second.

Two failures in the next-lane scenario were not the AEB at all. The ego car has no steering, and I was re-aiming its velocity along its current heading every tick, so a small yaw grew into a drift of nearly two metres. In one run it clipped the parked car at 49 km/h without braking. The ego now holds a fixed line, and any run where it drifts more than half a metre is marked invalid so it cannot be graded by accident. Across the final 300 runs the largest drift was 1 cm.

I also assumed that a failing seed would fail the same way again. Those three did, because geometry caused them, but the braking lead showed that the same seed can pass on one run and collide on another.

## Running it

Set up once.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/projects/aeb_ws
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-*.whl
python -m colcon build --symlink-install
source install/setup.bash
```

Build with `python -m colcon` so the entry points use the venv interpreter. The system Python cannot import `carla`.

Start CARLA, then run scenarios. Keep the quality on Low, because a 6 GB card is tight.

```bash
~/CARLA_0.9.16/CarlaUE4.sh -quality-level=Low -RenderOffScreen -nosound &
scripts/run_scenarios.sh CCRs_50 CCRb_50_d6 FP_none
```

`WATCH=1` adds a chase camera with a live readout and works with a windowed CARLA (`-windowed -ResX=1280 -ResY=720`). `RECORD=1` writes a rosbag for each run into `bags/`. The script loads Town04 before starting the pipeline, because switching maps with YOLO already on the GPU crashed the 6 GB card once.

Check the results from the bags and rebuild the charts and clips.

```bash
python scripts/analyze_bags.py bags
python scripts/make_pov_gif.py
/usr/bin/python3 scripts/make_charts.py     # matplotlib lives in the system Python
```

One chart compares the final run against an older one, so `make_charts.py` expects that older run in a local `bags_v7/` folder. Bags are not committed because they are about 3 GB per run.

The regression test needs a running CARLA server.

```bash
python -m aeb_bringup.regression --runs 30                 # about 100 minutes, writes docs/regression_report.md
AEB_REGRESSION=1 AEB_REGRESSION_RUNS=3 python -m pytest src/aeb_bringup/test/test_regression_sim.py
```

The runner exits with an error if any scenario misses its expectation in `config/regression.yaml`, which makes it usable as a CI gate. It saves each run as a small JSON file, resumes after a crash, retries an infrastructure error once, and cleans up any nodes left over from a launch that timed out. The pytest wrapper skips itself unless `AEB_REGRESSION=1` is set and CARLA is reachable.

The tests need no simulator.

```bash
python -m pytest src/aeb_controller/test/test_state_machine.py src/aeb_controller/test/test_tracking.py \
  src/aeb_fusion/test/test_geometry.py src/aeb_bringup/test/test_aeb_lifecycle.py \
  src/aeb_bringup/test/test_scenario_math.py src/aeb_bringup/test/test_regression_stats.py
```

That is 63 tests. They cover the controller logic, the tracker, the fusion geometry, the regression statistics and the controller's lifecycle transitions through `launch_testing`. The auto-generated flake8 and pep257 stubs that `ros2 pkg create` adds enforce ROS style and are not part of this list.

## What is in the repository

```
src/aeb_detector/     YOLO node
src/aeb_fusion/       fusion node and geometry.py (nearest cluster depth)
src/aeb_controller/   lifecycle node, tracking.py, state_machine.py
src/aeb_bringup/      launch file, scenarios.yaml, params, CARLA bridge, regression runner and statistics, tests
scripts/              run_scenarios.sh, analyze_bags.py, controller_sim.py, make_charts.py, make_pov_gif.py
benchmarks/           detector benchmark, offline fusion test, and their CSV results
docs/                 regression report and data, plus media/ with the charts and clips used here
```

`controller_sim.py` is the offline controller simulation I used to screen thresholds. It uses the measured brake map and the measured estimate noise. Run it with `validate` to compare it against the last recorded CARLA results.

## What comes next

The regression test exists and now runs thirty seeds per scenario, so the next jobs are the ones it pointed at. First is a faster reaction to a hard braking lead car, since that is the one scenario that still collides. After that I want to widen what is tested, with pedestrians, curves, cut-ins and weather, and a sweep over QoS settings and DDS implementations to see whether the middleware configuration changes the safety outcome. That last one is the part I think is most worth writing up properly.

Licensed under Apache 2.0.
