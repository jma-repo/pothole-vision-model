# Research notes

Directions parked for later, roughly in order of payoff. Each one lists what is already known and a first experiment small enough to finish in a weekend.

## 1. Severity from the car's accelerometer

**Why it matters.** Depth drives vehicle damage, and a single camera cannot see depth. Size and speed are the only severity inputs to the cost model today, and the damage rate per hit is a guess.

**What we know.** Tesla's SEI metadata carries `linear_acceleration_mps2_x/y/z` for every frame (see `dashcam.proto` in teslamotors/dashcam), and the z value changes on almost every frame at 36 fps. In clip `2026-02-28_11-34-29`, track 8 is a confirmed pothole in the right wheel path. About 0.1–0.3 s around when the car reached it, z swung between −1.8 and +1.9 m/s². The clip-wide standard deviation of z is 0.37 m/s².

**First experiment.**
- For every confirmed pothole in a wheel path, take the window from the moment it leaves the frame until the car's front axle reaches it. The axle reaches it after (distance at last sighting − camera-to-axle offset) / speed, about 0.2 s at 60 km/h.
- In that window, compare peak |Δz| with the same statistic at random moments in the same clip.
- If the confirmed-pothole peaks stand out, regress peak |Δz| on estimated size and speed.

**Pitfalls.** Expansion joints, manhole covers and rail crossings also jolt the car, and a hit with only one wheel shows up mostly in roll. Is gravity already removed from z? Check the value at standstill: it was about 0 in the clips seen so far.

## 2. Compare against the City of Calgary's own records

**Why it matters.** This is the business case. How many of these potholes had nobody reported, and how many days earlier did the car see the reported ones?

**What we know.** The City publishes a pothole repair status map (maps.calgary.ca/potholes), and 311 service requests are on the open data portal. 311 pothole requests rose from 2,680 in 2022 to 12,917 in 2024.

**First experiment.** Download the 311 pothole requests for the dates and areas driven. Match them to detections within about 30 m. Then report three numbers: the share of our potholes with no 311 request, the median lead time, and the 311 requests on driven roads that we missed.

**Pitfalls.** Check whether the 311 data carries coordinates or only an address or ward. Keep the downloaded data local, because matching it against GPS tracks reveals where the car was driven.

## 3. Real traffic volumes instead of guessing road type from speed

**Why it matters.** Traffic is the largest multiplier in the damage estimate. Speed is a poor proxy for road type in congestion, where a freeway looks like a local road.

**First experiment.** Get Calgary's open traffic-volume (average weekday traffic) and street-centreline data. Snap each pothole to the nearest centreline segment in its direction of travel. Then replace `LANE_VEHICLES_PER_DAY` with the segment's volume divided by its lane count.

## 4. Measure recall, not just precision

**Why it matters.** The review only covers what the model flagged. Missed potholes are invisible.

**First experiment.** Sample about 300 random frames from driving clips, stratified by night, wet and snow conditions. Label every pothole in them by hand. Then count how many the pipeline would have reported, with tracking and the three-hit rule applied.

## 5. One record per pothole, tracked over time

**Why it matters.** A city cares more about which pothole is getting worse fastest than about which one is largest. Repeat drives give a time series for free.

**First experiments.**
- Locate the pothole itself rather than the car. Project the box's bottom edge onto the road with the camera geometry already in `build_map.py`, and take the box from the closest frame, not the best-scoring one (`Track.box_by_frame` already keeps every frame).
- Give each pothole a persistent ID across drives, and plot estimated width against date.

## 6. Hard negatives as their own classes

**Why it matters.** The review surfaced four look-alikes:
- snow (pale)
- manhole covers (a clean oval outline)
- wet-pavement reflections (they move with the viewpoint)
- oil stains and crack filler (dark, no shadow)

Naming them as classes, instead of leaving them as background, gives the model a direct signal. Manhole and patch locations are also useful asset data for a city.

**First experiment.** Add `manhole` and `patch_or_sealant` classes. Label them in the frames already exported by `build_finetune.py` and compare the false-alarm rate with the background-only fine-tune.

## 7. Use the motion in the video

**Why it matters.** Reflections and glare do not move like the road surface, while potholes do. The car's speed is known for every frame.

**First experiment.** For each track, predict where its box should be in the next frame if it lies on a flat road. Use the speed, the heading and the camera geometry. Measure how far the detected box drifts from that prediction, and use the drift as a feature to reject reflections.

The back camera sees each pothole again after the car passes, which gives a second, independent look.

## 8. Spring collection protocol

Freeze-thaw potholes peak in March and April. Drive the same two or three fixed routes once a week through that window. Copy RecentClips after every drive, because Tesla keeps only about the last hour. That gives the time series for direction 5 and the seasonal recall for direction 4.

## 9. The next fine-tune

**What we know.** See "Retraining on the review labels" in the README. Two fine-tunes removed 8 of the 14 known false alarms on held-out drives. They also raised 43 and 48 new detections that nobody has reviewed. Going by the snapshots, most of these are false alarms. Used as a second opinion, the fine-tuned model did no better than a confidence threshold of 0.45.

**First experiment.** Review fine-tune 2's 48 extra detections on the review page (`build_review.py` on its run). This replaces the guess about its precision with a measurement. It also adds 48 labels at the exact spots where the two models disagree. Then retrain and score it on drives from days it has not seen.

**Pitfalls.**
- Split by day, not by drive. Drives on the same day share weather, light and roads. 37 of the 51 training potholes and 28 of the 31 test potholes came from 17 November 2026, and so did 35 of the 48 extra detections.
- Mix in all of RDD2022, and keep the learning rate low with no warm-up. The first fine-tune, on a 4,000-image sample with default warm-up, lost 0.12 of pothole mAP50 on the RDD2022 test split.
