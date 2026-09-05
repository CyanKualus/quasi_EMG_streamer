# BrainAmp MR online monitoring

The **Online Monitoring** tab receives the amplifier's data from **BrainVision
Recorder's Remote Data Access (RDA) server** over TCP. Its two black plots,
white traces, grid, LEFT/RIGHT arrangement and amplitude indicators follow
the El Artem application at `Exo quasi/EMG_proc/El Artem Exo`.

## Connect

1. Connect BrainAmp MR to the acquisition computer and open its normal
   BrainVision Recorder workspace. Use the uncorrected native signal, normally
   **5000 Hz**, as in the comparison recordings. Record the source in Recorder.
2. In Recorder, open **Preferences → Remote Data Access**, enable Remote Data
   Access, and start **monitoring**. The RDA client receives data during
   monitoring; saving a recording is controlled separately in Recorder.
3. Open this app's **Online Monitoring** tab. In **Stream options**, to the
   right of the graphs, enter the Recorder computer's hostname/IP.
   Use the arrow on the far right to fold or reopen these options.
   Use `127.0.0.1` when both programs run on the same computer.
   Use **51244** for float32, or **51234** for signed int16. If using two
   computers, allow the selected TCP port through the acquisition computer's
   firewall and use a stable local connection.
4. Enter each hand's **positive, negative** channel names exactly as Recorder
   labels them. Default RIGHT is `F1, F2`; blank LEFT disables that trace.
   Missing-channel errors list the channel names offered by Recorder.
5. Keep **Cycle / TR = 2.5 s** and **Filter buffer = 10 s** for the reviewed
   recordings, then press **Connect** below the bottom graph's options in the
   left column. The progress bar underneath counts received data
   through the approximately 30-second startup. No ECG channel is needed.

Recorder is the supported MR acquisition path: Brain Products explicitly says
its direct BrainAmp LSL connector is unsuitable for MR recordings. See the
[manufacturer's software downloads](https://www.brainproducts.com/downloads/more-software/)
and [Recorder/RDA downloads](https://www.brainproducts.com/downloads/recorder-1/).
RDA connection settings and message layouts are documented in the
[manufacturer's Recorder manual, Remote Data Access chapter](https://www.nmr.mgh.harvard.edu/~tatiana/BrainVisionManuals/Recorder_and_VideoRecorder/20201109_Recorder.pdf).

Disconnecting or closing this app closes its client connection. It does not
send commands to the amplifier or stop Recorder's acquisition or recording.
There is no automatic reconnection: fix the reported problem, then Connect.
Connection and display preferences are saved independently of offline settings.
File processing and figure export are disabled during monitoring to reserve
processing time for the stream.

## Display and timing

Both graphs always show the **TKEO envelope**: the existing 100-ms energy
average of the bipolar 20–95 Hz EMG, every 20 ms, in **mV²**. A previously saved
Filtered EMG preference no longer changes the displayed signal.

Each graph has its own **window options on its left**. Stream and connection
settings occupy a folding panel to the right of both graphs. Folding it with
the far-right arrow gives the graphs more width; its open/closed state is saved.
**Connect**, **Disconnect** and the stream progress bar stay visible on the
left, below the bottom graph's window options. LEFT and RIGHT graph settings
are independent and saved between sessions:

- **Time window**: 5–120 seconds of recent corrected data. The axis is elapsed
  time from the start of this monitoring segment, not a wall-clock timestamp.
- **Ymin**, **Ymax**, and **Magnitude N**: scaling is always manual. Valid
  edits immediately rescale the graph, before another sample or display timer
  tick, including before connecting and while frozen. Each axis value
  represents `value × 10ᴺ mV²`. Ymin and Ymax are integers; N is a signed
  exponent. For example, Ymin = 0,
  Ymax = 5 and N = −3 displays 0–0.005 mV², with ticks labeled 0–5. The axis
  shows its multiplier. Crossing one limit moves the other just enough to
  keep Ymin below Ymax by at least one integer step.
  Changing N rescales the display without changing stored TKEO values. Incoming
  samples do not adjust the limits, and old automatic-scale preferences are
  ignored.
- **Threshold (mV²)** is that hand's manual TKEO amplitude threshold, always in
  physical mV² regardless of N. Its dashed line uses the graph's scale. The
  green/red indicator below each graph compares the latest envelope value to
  that hand's threshold. It does not classify signal quality or determine
  whether artifacts have been removed.
- **Freeze display** holds that graph's data while acquisition, processing and
  the other graph continue. Its indicator turns gray and shows “frozen”. Scale,
  time-window and threshold edits still apply to the held data immediately;
  unfreezing shows recent data. Freeze is
  cleared when connecting and is not saved between sessions.

Existing shared time-window, TKEO Y-range and threshold
preferences initialize both hands when individual preferences have not yet
been saved. Saved EMG Y limits are replaced with the TKEO defaults because
their units differ.

Saved magnitudes from the former `10⁻ᴺ` convention are negated once during
migration, preserving the physical scale. Old fractional Y limits are rounded
outward to integers.

Selecting **Online Monitoring** opens the main window full screen. Press
**Esc** or select another tab to restore the previous window size (including
maximized state). Closing while monitoring is full screen saves the prior
window geometry, so the Start tab does not reopen full screen.

At the defaults, eight *past* complete cycles supply a 20-second template.
The first corrected target becomes available at 22.5 seconds. Four corrected
cycles fill the 10-second conditioning buffer at **30 seconds**; the app then
emits the samples from **25 to 27.5 seconds**. It commits another complete
2.5-second chunk every 2.5 seconds. It never revises an emitted sample.

Thus, when a chunk arrives, its samples are approximately **2.5–5 seconds old**,
plus computation and transport. The newest sample ages from about 2.5 to
5 seconds between updates. Envelope centering adds approximately 50–70 ms;
the right-neighbor dependency is retained across chunk boundaries.

The status estimates newest-TKEO age from the first received block and the
receiver's monotonic clock. **RDA does not provide amplifier wall-clock
timestamps**, so this estimate cannot measure buffering before the first
block arrived. The estimate includes subsequent backlog; it is not a calibrated
end-to-end latency measurement.

With a 2.5-second cycle, the selectable buffers are:

| Conditioning buffer | First output after | Delay of emitted samples |
|---|---:|---:|
| 5 s | 25 s | 2.5–5 s + computation |
| 7.5 s | 27.5 s | 2.5–5 s + computation |
| **10 s (default)** | **30 s** | **2.5–5 s + computation** |
| 20 s | 40 s | 2.5–5 s + computation |

The default starts a fixed cycle grid at the first received sample. Optionally,
set **Start marker** to a Recorder marker description to discard samples until
that marker and start the grid at its sample position. For example, `S 3`
matches `S  3`; runs of whitespace are normalized. This is a **one-time start
alignment**, not continuous trigger tracking. The last received marker is
shown in the status line.

AAS assumes a stable repeating gradient waveform and an appropriate repetition
period. Changing Cycle/TR changes template duration, buffering and latency;
only the **2.5-second grid** has been evaluated against these recordings. The
app does not infer scanner cycles, adjust clock drift or repair irregular
trigger timing. Restart monitoring and rebuild the template after changing
the scanner sequence. Correction can leave residual artifacts and introduce
small burst copies at neighboring scanner cycles.

## Processing and interruptions

1. Decode the RDA header, sample blocks and markers. Convert both integer and
   float32 wire values using each channel's header resolution: `V = value ×
   resolution_in_microvolts × 1e-6`. Float32 data also need this scaling, as in
   the [LSL RDA implementation](https://github.com/labstreaminglayer/App-BrainProducts/blob/master/BrainVisionRDA/rda_message.h).
2. Select only the requested EMG monopoles. At native rate, call MNE
   `GradientRemover` with **eight past neighbors and no future neighbors**;
   exclude the target and use complete cycles.
3. Resample the trailing AAS buffer to 250 Hz with MNE's polyphase resampler.
   Apply the tested 0.5–100 Hz zero-phase FIR and 50 Hz notch, then emit the
   penultimate complete cycle.
4. Form each bipolar difference; preserve the causal 50 Hz notch and fourth-order
   20–95 Hz Butterworth filter states between chunks. Preserve TKEO neighbors
   and the envelope window grid between chunks as well.

The required MNE version remains pinned in `requirements.txt`. Processing runs
outside the GUI thread. Native history, filter buffers, the display queue and
plot history are bounded. Incomplete final cycles and the last held-back cycle
are not flushed with invented future data.

Missing/duplicate/out-of-order blocks, malformed frames, non-finite samples,
excessive backlog and connection failures stop the client and gray the
indicators. Block-counter wraparound is supported. Samples are never silently
joined across a detected gap. A Recorder Stop clears the display and waits for
a new Start, which rebuilds the template and filter states. Starting a recording
while monitoring continues does not reset the template merely because Recorder
sends a state notification. Partial TCP packets and short socket timeouts do
not discard bytes. A stream stall is shown after one second without samples;
ten seconds without samples/complete messages stops the connection.

This tab monitors only; it does not write a second recording or trial scores.

The client and processing pipeline have been checked using local TCP replay.
An actual BrainAmp MR / BrainVision Recorder connection has not been tested
here; verify the acquisition computer's RDA settings and hardware connection.
