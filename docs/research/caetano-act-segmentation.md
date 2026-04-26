# ACT Segmentation (Caetano, Burred, Rodet — DAFx 2010) — Research Note

**Source:** Caetano, M., Burred, J. J., Rodet, X. *"Automatic Segmentation of the
Temporal Evolution of Isolated Acoustic Musical Instrument Sounds Using
Spectro-Temporal Cues."* Proc. DAFx-10, Graz, Austria, 2010. IRCAM
Analysis/Synthesis Team. PDF: `docs/Caetano Segmentation DAFx2010.pdf`.

This note summarises the paper as background reading for a future redesign of
`amplitude_analyze`. **No plan yet** — see "Why this matters" below for the
hooks back into our existing TODO. A detailed plan will be drafted later.

## Thesis

The classical ADSR amplitude envelope was invented by Moog as a *synthesis*
abstraction. It assumes (a) all sounds evolve through the same four stages and
(b) amplitude alone defines the boundaries. Neither assumption holds for
acoustic instruments — bowed strings have crescendo behaviour, plucked strings
have no sustain, percussive sounds have no decay/sustain distinction, etc.

Caetano et al. propose using the **Amplitude/Centroid Trajectory (ACT)** model
of Hajda (1996), which uses *both* the amplitude envelope and the spectral
centroid, with two methodological improvements:

1. A new amplitude envelope estimator — **True Amplitude Envelope (TAE)**.
2. **Adaptive efforts** — a robust boundary detector adapted from Peeters'
   "method of efforts" (Peeters 2004).

## The five-boundary ACT model

For *sustained* (step-like excitation: blown / bowed) instruments:

```
BN | A | T | S | R | BN
   1   2   3   4   5
```

| # | Name                          | Definition                                                                 |
|---|-------------------------------|----------------------------------------------------------------------------|
| 1 | Onset                         | Detected via Röbel 2003 transient-onset detector                           |
| 2 | End of attack                 | First local minimum of envelope between (1) and (3); centroid slope flips  |
| 3 | Begin of sustain              | Amplitude reaches local maximum; amplitude and centroid become monotonic   |
| 4 | Begin of release/interruption | Both amplitude *and* centroid start decreasing                              |
| 5 | Offset                        | Last point where TAE energy returns to onset energy                        |

For *impulsive* (plucked / struck) instruments the model collapses to
`BN | A | D/R | I | BN` (decay and release merge into one segment;
"interruption" handles e.g. a string being damped by hand).

If the attack is very fast (acoustic guitar example), boundaries (2) and (3)
coincide — the paper accepts this as a feature, not a bug.

## Two excitation classes (Fig. 5 in the paper)

- **Step-like** — energy is supplied for some duration, then interrupted
  (bowed, blown, sustained organ tones). Standing-wave pattern establishes and
  is held. → Sustained model (5 boundaries).
- **Impulse-like** — energy is supplied in a short burst (struck, plucked).
  Decay is constant due to losses; no true sustain. → Percussive model.

Most analog/digital synth patches we care about (Van Halen "Jump" lead, Roland
JUNO/Prophet pads) behave like step-like excitations and should use the
sustained model — *not* the percussive ADSR our current code forces.

## TAE — True Amplitude Envelope

Time-domain dual of Röbel & Rodet's (2005) cepstral true-envelope spectral
estimator. Iteratively cepstrum-smooths the rectified waveform so that:

- The envelope tracks the *peaks* of the waveform (not the RMS average).
- Inter-peak valleys are skipped (no ripples).
- It does not lag rapid amplitude changes the way RMS / low-pass does.

Compared to (Fig. 3):

- **Low-pass filtering (LPF):** ripples or sluggishness depending on cutoff.
- **RMS energy:** the popular default — moving-average behaviour, smooths over
  ripples but lags abrupt changes.
- **Hilbert / analytic signal:** only valid for narrow-band AM signals — not
  generally true for musical instruments.
- **FDLP (Frequency-Domain Linear Prediction):** model order is awkward to
  pick (proportional to F0, not formants).
- **TAE:** best fit; matches peaks closely without ripples for both bass
  clarinet and acoustic guitar test cases.

We currently use RMS in `amplitude_analyze`. TAE is a clear upgrade target.

## Adaptive efforts — boundary detection

Extension of Peeters' (2004) "method of efforts":

1. Divide the rise-time amplitude range into N equal intervals
   (*thresholds*).
2. For each threshold, measure how long the envelope takes to cross it.
3. The chosen boundary is the threshold whose crossing time is smaller than
   M times the mean crossing time (paper recommends M = 3 for the original
   method).

Caetano's *adaptive* extension:

- For end-of-attack (2): scan **forward from onset (1)**, count how many
  efforts are larger than the mean.
- For begin-of-release (4): scan **backward from offset (5)**, same logic.
- Use *different M values* for the two boundaries — empirically attack and
  release slopes differ enough that one threshold doesn't fit both.

Authors prefer this over derivative-based methods (Skowronek 2006, Jensen
1999) because second-derivative-zero-crossing is too sensitive to ripples in
the envelope. The TAE removes most ripples but the effort method gives
additional robustness.

## Test results (Fig. 7 in the paper)

| Instrument      | Outcome                                                                 |
|-----------------|-------------------------------------------------------------------------|
| Bass clarinet   | Clean fit — all 5 boundaries match visual inspection                    |
| Cello           | Bowed crescendo + vibrato confound both cues; (2) hard to localise; baseline (Peeters) "far off" |
| Acoustic guitar | Very fast attack → (2) and (3) coincide; (4)/(5) correct               |
| Marimba         | Model breaks — stick vibration during attack masks structure; only ~2 boundaries detected |

The cello and marimba failures are honest about model limits. Both involve
*excitation that is itself non-stationary* (bow noise during attack, stick
buzz). Synth tones do not have this problem — the excitation is electrical
and clean.

## Why this matters for `amplitude_analyze`

The four problems we documented in `docs/TODO.md` map almost one-to-one onto
the paper's critique of pure-amplitude segmentation:

| Our TODO issue (Van Halen "Jump")                                              | Paper's fix                                                            |
|--------------------------------------------------------------------------------|------------------------------------------------------------------------|
| Attack end taken at RMS peak — inflated; should be slope inflection            | Use first local minimum of envelope between (1) and (3), or centroid slope flip |
| Decay segment forced into the model on synths that don't have one              | Sustained ACT model has no D — only A → T → S → R                      |
| Sustain end detected too late (noise floor sustains envelope past signal drop) | Offset (5) defined where TAE energy returns to onset energy            |
| Release length 2–5× too long (dominated by noise floor)                        | Release is bounded above by offset (5) and detected via centroid drop  |

The biggest architectural addition is **spectral centroid as a second cue**.
Our current code uses amplitude only. The cases where we fail (slow synth
attacks, evolving pads, sounds with significant noise floor) are exactly the
cases where amplitude is ambiguous and centroid disambiguates.

## Open questions for the future plan

1. **Synth-specific tweaks.** The paper targets acoustic instruments. Synth
   tones (especially digital subtractive / FM patches) have cleaner
   excitation — does that simplify TAE / adaptive-effort tuning, or does it
   expose new problems (e.g. modulated centroid from filter envelopes)?
2. **Per-cluster vs per-note.** Our current pipeline runs per-cluster
   (multiple onsets in a chord/cluster). The paper assumes a single isolated
   note. We need to reconcile cluster-level analysis with note-level
   segmentation — likely by isolating the dominant note first
   (`note_isolate`) and segmenting that.
3. **Polyphonic noise floor.** ACT assumes silence before onset and after
   offset. In polyphonic source-separated stems the "background" is residual
   bleed, not silence. The offset criterion ("TAE returns to onset energy")
   may need adjustment for non-zero floors.
4. **TAE implementation.** Röbel & Rodet's cepstral true-envelope is
   well-known; the *time-domain dual* used here is less standard. Need to
   check whether a Python implementation exists (likely IRCAM-internal) or
   we'd reimplement from the cepstral version.
5. **Centroid stability under vibrato / chorus.** The cello result shows
   vibrato breaks centroid monotonicity. Many synth patches use chorus / LFO
   modulation — same risk. May need to smooth centroid over a window matched
   to the LFO rate (which `spectrum_analyze` already estimates).

## References (paper bibliography highlights)

- [10] Peeters, G. *"A large set of audio features for sound description (similarity and classification) in the CUIDADO project"*, 2004 — origin of the method-of-efforts baseline.
- [14] Hajda, J. *"A New Model for Segmenting the Envelope of Musical Signals: The relative Salience of Steady State versus Attack, Revisited"*, JAES, Nov. 1996 — origin of the ACT model.
- [15] Röbel, A., Rodet, X. *"Efficient Spectral Envelope Estimation and its Application to Pitch Shifting And Envelope Preservation"*, DAFx 2005 — the cepstral true-envelope that TAE adapts.
- [18] Skowronek, J., McKinney, M. *"Features for Audio Classification: Percussiveness of Sounds"*, 2006 — derivative-based segmentation baseline.
- [26] Röbel, A. *"A New Approach to Transient Processing in the Phase Vocoder"*, DAFx 2003 — onset detector used for boundary (1).

Authors' code/examples: `http://recherche.ircam.fr/anasyn/caetano/seg.html`
(URL from 2010 — verify before relying on it).