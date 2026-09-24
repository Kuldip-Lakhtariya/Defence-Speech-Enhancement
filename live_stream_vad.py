"""
live_stream_vad.py

Real-time mic -> VAD-gated DeepFilterNet3 (ONNX) enhancement -> speaker.

Difference from live_stream_onnx.py: that script streams continuously with
no voice-activity gating. This script only buffers/processes/outputs audio
while the user is actually speaking (detected via Silero VAD), and computes
sub-chunks progressively DURING speech rather than waiting for the full
utterance to end - so enhanced audio starts appearing mid-utterance, not
only after the user stops talking. Accepted tradeoff: same chunk-boundary
quality dip you already accepted for the non-VAD chunked path, now also at
speech-segment sub-chunk boundaries.

>>> DEVICE / MODEL SAMPLE-RATE MISMATCH FIX (Sep 2026) <<<
Your Bluetooth headset's WDM-KS HFP profile (devices 29/30) only runs at
16kHz natively - it will not open at 48kHz (PaErrorCode -9996). The old
version of this script assumed the *device* ran at MIC_SR (48000), which
only worked by accident on devices whose driver silently resamples (e.g.
MME). WDM-KS does not resample for you, so we now capture/play at the
device's real rate (DEVICE_SR) and resample up to the model's expected
rate (MIC_SR) only around enhance_chunked(), not around VAD (which already
wants 16kHz, so DEVICE_SR == VAD_SR removes a resample step that used to
exist here).

>>> SILERO VAD CONTEXT-FRAME FIX (Sep 2026) <<<
The v5+ Silero combined-state ONNX export expects each 512-sample frame to
be prefixed with the last VAD_CONTEXT_SIZE (64 @ 16kHz) samples of the
PREVIOUS frame - not a bare, independent 512-sample window. The ONNX input
shape ([None, None]) happily accepts a bare 512-sample frame with no error,
but silently produces near-zero probabilities across real speech, because
the model is missing timing context it was trained to expect. Confirmed via
side-by-side test against the official silero-vad pip package on identical
audio: bare 512-sample frames scored ~0.001-0.003 throughout a 5s speech
clip (should peak >0.5), while prefixing the 64-sample context produced
matching scores to the official package (~0.55 during voiced frames). Fix:
SileroVAD now keeps a self._context buffer (tail of the previous frame,
zeros at start / after reset_state()) and prepends it to every 512-sample
frame before calling the session, updating it after each call.

>>> SEGMENTATION / PLAYBACK NOTES (Sep 2026) <<<
Short natural pauses mid-sentence were triggering "speech end" too early
(--hangover-ms was 400), causing a sentence to get split into two
independently-processed fragments (separate GRU-state resets) that could
sound like a slight echo/double-voice on playback. Default --hangover-ms
raised to 1000 so brief pauses don't split a sentence into multiple
segments.

>>> LIVE CHUNKS vs FULL-SEGMENT RECORDING (Sep 2026) <<<
chunk_seconds controls BOTH how much audio must accumulate before live
playback can start AND the size of the saved .wav pieces - it is not just
a "recording length" knob, since the exact same enhanced_device_sr that
gets written to disk is also what's pushed to out_q for live playback.
Smaller chunks = lower live latency but more GRU-state resets (more
chunk-boundary quality dips); larger chunks = smoother quality but longer
dead air before you hear anything live.

To let quality be judged without that tradeoff, live chunked
streaming/playback is left completely unchanged (still gated by
--chunk-seconds, still low-latency), but a SEPARATE full_buf now
accumulates the entire utterance from "speech start" to "speech end". On
"speech end", that whole buffer is run through enhance_chunked() in ONE
pass (chunk_seconds set to the segment's own actual duration, so there is
only the one unavoidable reset at the very start of the segment - none of
the mid-utterance boundary resets the live progressive chunks have) and
saved as its own {ts}_full_raw.wav / {ts}_full_enhanced.wav pair. This is
for offline A/B listening only - it does not feed out_q and has no effect
on live playback.

>>> ASSUMPTIONS TO VERIFY / ADJUST AGAINST YOUR ACTUAL CODEBASE <<<
1. enhance_chunked() is called with the real src/df_onnx_dsp.py signature:
   enhance_chunked(audio, enc_session, erb_dec_session, df_dec_session,
   erb_inv_fb, chunk_seconds=..., sr=...). ONNX sessions/erb_inv_fb are
   loaded ONCE at startup via load_sessions()/build_erb_inv_fb(ERB_WIDTHS),
   the same pattern pipeline.py's _get_onnx_sessions() uses. Adjust the
   import names and load_sessions()/build_erb_inv_fb() call below if your
   actual function names differ. This is the single-mic path (no NLMS/
   reference mic), matching live_demo.py --mic and live_stream_onnx.py.
2. Silero VAD ONNX model confirmed via --introspect-vad-only against a v5+
   export: combined state tensor, not separate h/c. Confirmed I/O:
       inputs:  input float32 [None, None] @16kHz,
                state float32 [2, None, 128],
                sr int64 scalar
       outputs: output float32 [None, 1], stateN float32 [2, None, None]
   If you re-introspect against a different model version and the names/
   shapes differ, adjust VAD_INPUT_NAMES / VAD_OUTPUT_NAMES / VAD_STATE_SHAPE
   below to match. Also re-check VAD_CONTEXT_SIZE (64 @16kHz, would be 32
   @8kHz) if you switch model variants.
   Download (no torch needed, just the .onnx file):
       https://github.com/snakers4/silero-vad ->
       src/silero_vad/data/silero_vad.onnx
3. Confirmed device rates via sounddevice.query_devices() on this machine:
       device 29 (WDM-KS, HFP output) -> default_samplerate 16000.0
       device 30 (WDM-KS, HFP input)  -> default_samplerate 16000.0 (paired mic)
   If you switch headsets/drivers, re-run:
       python -c "import sounddevice as sd; print(sd.query_devices(<id>))"
   and update DEVICE_SR below if it differs from 16000.
4. HFP (Bluetooth hands-free) is a mono, narrowband voice profile - any
   dual-mic beamforming your earbuds do happens inside the earbuds' own
   chip before audio reaches Windows, so device 30 almost certainly only
   exposes a single mono channel here regardless of the earbuds' physical
   mic count. Confirm with:
       python -c "import sounddevice as sd; print(sd.query_devices(30))"
   and check max_input_channels.
"""

import argparse
import os
import queue
import threading
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample_poly

# --- adjust these imports to match your actual project structure ---
from src.df_onnx_dsp import (  # noqa: E402
    enhance_chunked,
    load_sessions,
    build_erb_inv_fb,
    ERB_WIDTHS,
)

DEVICE_SR = 16000          # actual hardware rate for devices 29/30 (WDM-KS HFP) - NOT 48000
MIC_SR = 48000             # rate enhance_chunked()/the model expects internally - unchanged
VAD_SR = 16000              # Silero VAD's expected rate - now equals DEVICE_SR, so the mic
                              # block VAD sees no longer needs downsampling (dev/prototype
                              # simplification specific to this headset; if you later add a
                              # device that captures at 48kHz, VAD will need its own resample
                              # path again - don't assume DEVICE_SR == VAD_SR in general)
VAD_FRAME_SAMPLES = 512    # 32ms @ 16kHz, Silero fixed frame size
VAD_CONTEXT_SIZE = 64      # required prefix samples for the v5+ combined-state export @16kHz
                           # (would be 32 @ 8kHz) - see CONTEXT-FRAME FIX note above

VAD_INPUT_NAMES = {"input": "input", "sr": "sr", "state": "state"}
VAD_OUTPUT_NAMES = {"prob": "output", "state": "stateN"}
VAD_STATE_SHAPE = (2, 1, 128)  # combined h/c state, v5+ Silero ONNX export

CAPTURE_DIR = "live_captures"  # raw+enhanced .wav pairs (chunk + full-segment) saved here


def print_model_io(session: ort.InferenceSession) -> None:
    print("VAD model inputs:")
    for i in session.get_inputs():
        print(f"  {i.name}  shape={i.shape}  dtype={i.type}")
    print("VAD model outputs:")
    for o in session.get_outputs():
        print(f"  {o.name}  shape={o.shape}  dtype={o.type}")


def upsample_device_to_model(audio_block: np.ndarray) -> np.ndarray:
    """DEVICE_SR (16kHz) -> MIC_SR (48kHz), ratio 1:3.

    Uses polyphase resampling (resample_poly), not naive repeat/slice -
    naive decimation/upsampling aliases badly on speech and would degrade
    exactly what you're trying to enhance. Safe to call per-chunk (stateless
    across calls) because chunk_samples is fixed and boundaries land on
    whole chunks, not arbitrary sample counts.
    """
    return resample_poly(audio_block, up=3, down=1).astype(np.float32)


def downsample_model_to_device(audio_block: np.ndarray) -> np.ndarray:
    """MIC_SR (48kHz) -> DEVICE_SR (16kHz), ratio 1:3, inverse of above."""
    return resample_poly(audio_block, up=1, down=3).astype(np.float32)


class SileroVAD:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self._resid = np.zeros(0, dtype=np.float32)  # leftover samples < one VAD frame
        self._context = np.zeros(VAD_CONTEXT_SIZE, dtype=np.float32)  # tail of previous frame

    def reset_state(self) -> None:
        self.state[:] = 0
        self._context[:] = 0  # new speech segment - context shouldn't bleed across segments

    def frame_probs(self, mic_frame_16k: np.ndarray) -> list[float]:
        """Feed a block of 16kHz mic audio (device-native rate - no resample
        needed here anymore, see DEVICE_SR/VAD_SR note above), return list
        of speech probs for every complete 512-sample @16kHz VAD frame
        contained in it."""
        buf = np.concatenate([self._resid, mic_frame_16k])
        n_frames = len(buf) // VAD_FRAME_SAMPLES
        probs = []
        for i in range(n_frames):
            frame = buf[i * VAD_FRAME_SAMPLES:(i + 1) * VAD_FRAME_SAMPLES]
            frame_with_context = np.concatenate([self._context, frame])  # 576 samples
            out = self.session.run(
                [VAD_OUTPUT_NAMES["prob"], VAD_OUTPUT_NAMES["state"]],
                {
                    VAD_INPUT_NAMES["input"]: frame_with_context.reshape(1, -1).astype(np.float32),
                    VAD_INPUT_NAMES["sr"]: np.array(VAD_SR, dtype=np.int64),
                    VAD_INPUT_NAMES["state"]: self.state,
                },
            )
            prob, self.state = out
            self._context = frame[-VAD_CONTEXT_SIZE:]  # tail of this frame -> next frame's context
            probs.append(float(prob.squeeze()))
        self._resid = buf[n_frames * VAD_FRAME_SAMPLES:]
        return probs


class SpeechGate:
    """Simple hangover-based state machine turning per-frame VAD probs into
    speech-active / speech-ended events."""

    def __init__(self, threshold: float, hangover_frames: int, onset_frames: int):
        self.threshold = threshold
        self.hangover_frames = hangover_frames
        self.onset_frames = onset_frames
        self.active = False
        self._voiced_run = 0
        self._silence_run = 0

    def update(self, prob: float) -> str | None:
        """Returns one of: None, 'start', 'ongoing', 'end'"""
        voiced = prob >= self.threshold
        if not self.active:
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self.onset_frames:
                self.active = True
                self._silence_run = 0
                self._voiced_run = 0
                return "start"
            return None
        else:
            self._silence_run = 0 if voiced else self._silence_run + 1
            if self._silence_run >= self.hangover_frames:
                self.active = False
                self._silence_run = 0
                return "end"
            return "ongoing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="models/onnx_export_v2")
    ap.add_argument("--vad-model", default="models/onnx_export_v2/silero_vad.onnx")
    ap.add_argument("--chunk-seconds", type=float, default=2.0)
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--onset-ms", type=float, default=96, help="voiced time needed to trigger start")
    ap.add_argument("--hangover-ms", type=float, default=1000, help="silence time needed to end segment")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
    ap.add_argument("--introspect-vad-only", action="store_true")
    args = ap.parse_args()

    if args.introspect_vad_only:
        sess = ort.InferenceSession(args.vad_model, providers=["CPUExecutionProvider"])
        print_model_io(sess)
        return

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    vad_ms_per_frame = 1000 * VAD_FRAME_SAMPLES / VAD_SR  # ~32ms
    onset_frames = max(1, round(args.onset_ms / vad_ms_per_frame))
    hangover_frames = max(1, round(args.hangover_ms / vad_ms_per_frame))

    vad = SileroVAD(args.vad_model, threshold=args.vad_threshold)
    gate = SpeechGate(args.vad_threshold, hangover_frames, onset_frames)

    # Load DeepFilterNet3 ONNX sessions ONCE at startup, same pattern as
    # pipeline.py's _get_onnx_sessions(). Adjust the unpacking below if
    # load_sessions() in your codebase returns a different shape/order.
    enc_session, erb_dec_session, df_dec_session = load_sessions(args.onnx_dir)
    erb_inv_fb = build_erb_inv_fb(ERB_WIDTHS)

    # Separate session set for the background full-segment save path - deliberately
    # NOT shared with the live path's sessions above. If the ONNX runtime build in
    # use here isn't safe for two threads to call .run() on the SAME session
    # concurrently, sharing sessions between the live thread and the background
    # thread could hang, throw, or corrupt output - and because that would happen
    # inside a background thread, it could silently stall everything, including the
    # real-time audio callbacks, with no visible error. Loading a second small set
    # of sessions (~2-3MB each) removes that possibility outright.
    bg_enc_session, bg_erb_dec_session, bg_df_dec_session = load_sessions(args.onnx_dir)

    # chunk_samples is in DEVICE_SR (16kHz) units, because speech_buf
    # accumulates mic callback blocks directly at device rate - it only
    # becomes MIC_SR (48kHz) briefly inside process_and_emit(), around the
    # model call. Getting this unit wrong is the single easiest way to
    # silently reintroduce the old bug: chunk_seconds would then no longer
    # match real wall-clock seconds of buffered audio.
    chunk_samples = int(args.chunk_seconds * DEVICE_SR)
    speech_buf = np.zeros(0, dtype=np.float32)   # drained progressively for live chunked playback
    full_buf = np.zeros(0, dtype=np.float32)     # accumulates the WHOLE segment, never drained mid-utterance
    out_buffer = np.zeros(0, dtype=np.float32)   # persistent leftover audio for out_callback, see below

    in_q: "queue.Queue[np.ndarray]" = queue.Queue()
    out_q: "queue.Queue[np.ndarray]" = queue.Queue()
    log_q: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    def mic_callback(indata, frames, time_info, status):
        # no print()/logging here - keep the callback fast, per the earlier
        # overflow/underflow fix. Push status to a queue instead.
        if status:
            log_q.put(str(status))
        in_q.put(indata[:, 0].copy())

    def out_callback(outdata, frames, time_info, status):
        nonlocal out_buffer
        if status:
            log_q.put(str(status))
        # out_q holds whole enhanced chunks (up to chunk_seconds long - tens of
        # thousands of samples), but each callback only needs `frames` samples
        # (the stream's small blocksize, ~128ms). Pulling one queue item per
        # callback and slicing off just block[:frames] - as this used to do -
        # silently threw away almost all of every chunk's audio. Instead, keep
        # a persistent buffer here and only pull a fresh chunk from out_q when
        # the buffer runs low, so every sample eventually gets played across
        # however many callbacks it takes to drain it.
        while len(out_buffer) < frames:
            try:
                chunk = out_q.get_nowait()
            except queue.Empty:
                break
            out_buffer = np.concatenate([out_buffer, chunk])

        if len(out_buffer) >= frames:
            outdata[:, 0] = out_buffer[:frames]
            out_buffer = out_buffer[frames:]
        else:
            # genuinely nothing buffered yet (e.g. not speaking) - play silence
            outdata[:, 0] = np.pad(out_buffer, (0, frames - len(out_buffer)))
            out_buffer = np.zeros(0, dtype=np.float32)

    def logger_thread() -> None:
        while not stop_event.is_set():
            try:
                msg = log_q.get(timeout=0.5)
                print(f"[status] {msg}")
            except queue.Empty:
                continue

    def make_timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"

    def process_and_emit(buf_device_sr: np.ndarray) -> None:
        """Live path - unchanged. Runs on progressive sub-chunks (size =
        --chunk-seconds) while speaking, and on whatever's left over at
        speech end. Feeds out_q for real-time playback. This is the path
        with occasional chunk-boundary quality dips - that's the accepted
        real-time tradeoff, not a bug."""
        buf_model_sr = upsample_device_to_model(buf_device_sr)
        enhanced_model_sr = enhance_chunked(
            buf_model_sr,
            enc_session,
            erb_dec_session,
            df_dec_session,
            erb_inv_fb,
            chunk_seconds=args.chunk_seconds,
            sr=MIC_SR,  # unchanged - model still runs at 48kHz internally
        )
        enhanced_device_sr = downsample_model_to_device(enhanced_model_sr)
        out_q.put(enhanced_device_sr.astype(np.float32))

    def process_full_segment(buf_device_sr: np.ndarray) -> None:
        """Offline-comparison path only - does NOT touch out_q / live
        playback. Runs the WHOLE utterance through enhance_chunked() in one
        pass, so there's only the one unavoidable reset at the very start
        of the segment instead of one every --chunk-seconds. Saved to disk
        for you to listen to afterward on a real speaker."""
        if len(buf_device_sr) == 0:
            return
        segment_seconds = len(buf_device_sr) / DEVICE_SR
        buf_model_sr = upsample_device_to_model(buf_device_sr)
        enhanced_model_sr = enhance_chunked(
            buf_model_sr,
            bg_enc_session,       # separate session set from the live path - see load_sessions() above
            bg_erb_dec_session,
            bg_df_dec_session,
            erb_inv_fb,
            chunk_seconds=segment_seconds,  # whole segment as ONE chunk - no mid-utterance resets
            sr=MIC_SR,
        )
        enhanced_device_sr = downsample_model_to_device(enhanced_model_sr)

        ts = make_timestamp()
        raw_path = os.path.join(CAPTURE_DIR, f"{ts}_full_raw.wav")
        enhanced_path = os.path.join(CAPTURE_DIR, f"{ts}_full_enhanced.wav")
        sf.write(raw_path, buf_device_sr, DEVICE_SR)
        sf.write(enhanced_path, enhanced_device_sr, DEVICE_SR)
        log_q.put(f"saved full segment ({segment_seconds:.2f}s): {raw_path} / {enhanced_path}")

    threading.Thread(target=logger_thread, daemon=True).start()

    # Both streams opened at DEVICE_SR (16kHz) on the SAME host API
    # (WDM-KS, devices 29/30) - opening input on one API (e.g. MME) and
    # output on another (WDM-KS) is what threw PaErrorCode -9996 earlier;
    # WDM-KS is exclusive-mode and picky about that kind of mixing.
    with sd.InputStream(
        samplerate=DEVICE_SR, channels=1, blocksize=2048, latency="high",
        device=args.input_device, callback=mic_callback,
    ), sd.OutputStream(
        samplerate=DEVICE_SR, channels=1, blocksize=2048, latency="high",
        device=args.output_device, callback=out_callback,
    ):
        print("Listening (Ctrl+C to stop)...")
        try:
            while True:
                mic_frame = in_q.get()
                probs = vad.frame_probs(mic_frame)

                if probs:
                    print(f"[vad] max_prob={max(probs):.3f}  mic_level={np.abs(mic_frame).max():.4f}")

                if gate.active or any(p >= args.vad_threshold for p in probs):
                    speech_buf = np.concatenate([speech_buf, mic_frame])
                    full_buf = np.concatenate([full_buf, mic_frame])

                for p in probs:
                    event = gate.update(p)
                    if event == "start":
                        log_q.put("speech start")
                    elif event == "end":
                        log_q.put("speech end")
                        if len(speech_buf) > 0:
                            process_and_emit(speech_buf)          # live path, unchanged
                        # Run the whole-utterance offline save on a background thread -
                        # it re-enhances the FULL segment in one call, which can take
                        # a few seconds for a long sentence. Running it inline here
                        # would block this loop from draining in_q/out_q for that
                        # whole time, silencing live playback until it finished. Pass
                        # a copy since full_buf is about to be reset to empty below.
                        def _run_full_segment_safely(buf: np.ndarray) -> None:
                            # Background threads swallow uncaught exceptions to
                            # stderr, where they're easy to miss in a scrolling
                            # terminal - log them explicitly so a bug here is
                            # visible instead of silently going nowhere.
                            try:
                                process_full_segment(buf)
                            except Exception as exc:  # noqa: BLE001
                                log_q.put(f"[full-segment ERROR] {type(exc).__name__}: {exc}")

                        threading.Thread(
                            target=_run_full_segment_safely,
                            args=(full_buf.copy(),),
                            daemon=True,
                        ).start()
                        speech_buf = np.zeros(0, dtype=np.float32)
                        full_buf = np.zeros(0, dtype=np.float32)
                        vad.reset_state()

                # progressive processing: as soon as we have a full chunk
                # WHILE still speaking, process it now instead of waiting
                # for the segment to end. (Live path only - full_buf is
                # left untouched here, it keeps accumulating.)
                while gate.active and len(speech_buf) >= chunk_samples:
                    sub_chunk, speech_buf = speech_buf[:chunk_samples], speech_buf[chunk_samples:]
                    process_and_emit(sub_chunk)

        except KeyboardInterrupt:
            stop_event.set()
            print("\nStopped.")


if __name__ == "__main__":
    main()