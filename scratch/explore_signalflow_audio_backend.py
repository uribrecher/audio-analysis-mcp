"""Diagnose what audio backend SignalFlow tries to load on offline-render.

Context: PR #15 CI (ubuntu-latest, headless) segfaults inside
``graph.render_to_new_buffer()`` when the audio graph is built with
``output_device=None, start=False``. The expectation was that None ==
"no real device, just offline render" — but the segfault suggests
SignalFlow still touches a backend (PortAudio / ALSA) when constructing
the graph or when rendering.

This script answers two questions and informs the CI fix:

1. Does ``import signalflow`` already require shared libs we don't have
   on the headless runner (libportaudio, libasound)? `ldd` the .so.
2. Does ``AudioGraph(output_device=None, start=False)`` succeed by
   itself, or does the segfault only happen at ``render_to_new_buffer``?
   Bisect the failure to either ctor or render.

We dump:
  - signalflow module path
  - linked shared libs (`ldd` on Linux, `otool -L` on macOS) for the
    main signalflow .so
  - presence of PortAudio / ALSA libs on $LD_LIBRARY_PATH
  - whether AudioGraph constructs cleanly
  - whether a no-op render (1 frame, no nodes) returns

Run: uv run python scratch/explore_signalflow_audio_backend.py
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def _find_signalflow_so(sf_lib_path: str) -> list[Path]:
    pkg_dir = Path(sf_lib_path).parent
    return sorted(pkg_dir.rglob("*.so")) + sorted(pkg_dir.rglob("*.dylib"))


def _ldd(path: Path) -> str:
    if platform.system() == "Darwin":
        cmd = ["otool", "-L", str(path)]
    else:
        cmd = ["ldd", str(path)]
    if shutil.which(cmd[0]) is None:
        return f"  ({cmd[0]} not available)"
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return res.stdout + (res.stderr or "")
    except Exception as exc:
        return f"  ({cmd[0]} failed: {exc})"


def main() -> None:
    print("=" * 72)
    print("SignalFlow audio-backend probe")
    print(f"  python : {sys.version.split()[0]}")
    print(f"  platform: {platform.platform()}")
    print(f"  sys.platform: {sys.platform}")
    print(f"  ext_suffix: {sysconfig.get_config_var('EXT_SUFFIX')}")
    print("=" * 72)

    print("\n[1] importing signalflow...")
    import signalflow as sf_lib

    print(f"  signalflow.__file__: {sf_lib.__file__}")
    print(f"  signalflow.__version__: {getattr(sf_lib, '__version__', '<missing>')}")

    sos = _find_signalflow_so(sf_lib.__file__)
    print(f"\n[2] signalflow shared libs ({len(sos)} found):")
    for p in sos:
        print(f"  - {p}")

    print(f"\n[3] linked deps of signalflow's main .so:")
    for p in sos:
        if "signalflow" in p.name.lower() and (".so" in p.name or ".dylib" in p.name):
            print(f"  --- {p.name} ---")
            print(_ldd(p))

    print(f"\n[4] env vars relevant to audio backends:")
    for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "ALSA_CONFIG_PATH", "JACK_NO_AUDIO_RESERVATION"):
        print(f"  {var}={os.environ.get(var, '<unset>')}")

    print(f"\n[5] AudioGraph(output_device=None, start=False) — does ctor itself segfault?")
    try:
        graph = sf_lib.AudioGraph(output_device=None, start=False)
        print(f"  OK: graph={graph!r}")
    except Exception as exc:
        print(f"  EXC: {type(exc).__name__}: {exc}")
        return

    print(f"\n[6] render_to_new_buffer(num_frames=1) on empty graph — segfault here?")
    print(f"     (running in subprocess so the segfault doesn't kill us)")
    probe = Path(__file__).parent / "_signalflow_render_probe.py"
    probe.write_text(
        "import signalflow as sf_lib\n"
        "g = sf_lib.AudioGraph(output_device=None, start=False)\n"
        "buf = g.render_to_new_buffer(num_frames=1)\n"
        "print('render_to_new_buffer OK, buf=', buf)\n"
        "g.destroy()\n"
        "print('graph destroyed OK')\n"
    )
    res = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True)
    print(f"  exit code: {res.returncode}")
    print(f"  stdout: {res.stdout!r}")
    print(f"  stderr: {res.stderr!r}")
    if res.returncode == -11:
        print("  >>> Confirmed segfault at render_to_new_buffer.")
    elif res.returncode != 0:
        print(f"  >>> Non-zero exit; check stderr above.")
    else:
        print("  >>> Render succeeded — segfault must be node-related.")

    # Cleanup probe.
    try:
        probe.unlink()
    except Exception:
        pass

    print(f"\n[7] inspect AudioGraphConfig attrs at runtime (stubs are 0.5.x;")
    print(f"     installed signalflow may be 0.5.3 with subset of properties).")
    graph.destroy()
    cfg = sf_lib.AudioGraphConfig()
    print(f"     cfg type: {type(cfg).__name__}")
    print(f"     cfg public attrs: {sorted(a for a in dir(cfg) if not a.startswith('_'))}")
    g2 = sf_lib.AudioGraph(config=cfg, start=False)
    print(f"     AudioGraph (default cfg) OK")
    print(f"     g2.get_backend_names(): {g2.get_backend_names()}")
    print(f"     g2 public attrs (filtered for backend/output/device):")
    for a in sorted(dir(g2)):
        if any(t in a.lower() for t in ("backend", "output", "device")):
            print(f"       {a}")
    g2.destroy()

    # Try setting backend_name via available attrs.
    print(f"\n[7b] try backend_name='dummy' on AudioGraphConfig.")
    cfg2 = sf_lib.AudioGraphConfig()
    if hasattr(cfg2, "backend_name"):
        # miniaudio backends: null, coreaudio, alsa, pulseaudio, jack, wasapi, etc.
        # Try each in turn — what works on this machine, plus null which is
        # the universal "no audio device" fallback inside miniaudio.
        import numpy as np
        for name in ("null", "dummy", "coreaudio", "alsa", "pulseaudio", "jack"):
            cfg2 = sf_lib.AudioGraphConfig()
            cfg2.backend_name = name
            try:
                g3 = sf_lib.AudioGraph(config=cfg2, start=False)
            except Exception as exc:
                print(f"     backend_name={name!r}: {type(exc).__name__}: {exc}")
                continue
            try:
                osc = sf_lib.SineOscillator(frequency=440.0)
                env = sf_lib.ADSREnvelope(
                    attack=0.01, decay=0.05, sustain=0.7, release=0.05, gate=1
                )
                out = osc * env
                out.play()
                buf = g3.render_to_new_buffer(num_frames=1024)
                arr = np.asarray(buf.data, dtype=np.float32)
                print(
                    f"     backend_name={name!r}: render OK, shape={arr.shape}, "
                    f"peak={float(np.abs(arr).max()):.4f}"
                )
            finally:
                g3.destroy()
    else:
        print(f"     backend_name attr NOT on AudioGraphConfig in this version.")
        print(f"     Try ENV VAR: SIGNALFLOW_OUTPUT_BACKEND_NAME=dummy (subprocess).")
        probe2 = Path(__file__).parent / "_signalflow_dummy_env_probe.py"
        probe2.write_text(
            "import os, numpy as np\n"
            "import signalflow as sf_lib\n"
            "cfg = sf_lib.AudioGraphConfig()\n"
            "g = sf_lib.AudioGraph(config=cfg, start=False)\n"
            "print('backends:', g.output_backend_names)\n"
            "osc = sf_lib.SineOscillator(frequency=440.0)\n"
            "env = sf_lib.ADSREnvelope(attack=0.01, decay=0.05, sustain=0.7, release=0.05, gate=1)\n"
            "(osc * env).play()\n"
            "buf = g.render_to_new_buffer(num_frames=1024)\n"
            "arr = np.asarray(buf.data)\n"
            "print('render shape:', arr.shape, 'peak:', float(np.abs(arr).max()))\n"
            "g.destroy()\n"
        )
        env_combos = [
            {},
            {"SIGNALFLOW_OUTPUT_BACKEND_NAME": "dummy"},
            {"SIGNALFLOW_BACKEND": "dummy"},
        ]
        for e in env_combos:
            full_env = os.environ.copy()
            full_env.update(e)
            res = subprocess.run(
                [sys.executable, str(probe2)],
                capture_output=True, text=True, env=full_env,
            )
            print(f"     env={e or '<empty>'}: rc={res.returncode}, "
                  f"stdout={res.stdout.strip()!r}")
            if res.stderr.strip():
                print(f"       stderr={res.stderr.strip()!r}")
        try:
            probe2.unlink()
        except Exception:
            pass

    # Also test passing config_name="offline" or "dummy" (string overload).
    print(f"\n[7c] try AudioGraphConfig(arg0='dummy' or 'offline') — string overload.")
    for name in ("dummy", "offline", "headless"):
        try:
            cfg3 = sf_lib.AudioGraphConfig(name)
            print(f"     AudioGraphConfig({name!r}) OK")
        except Exception as exc:
            print(f"     AudioGraphConfig({name!r}) raised: "
                  f"{type(exc).__name__}: {exc}")

    print(f"\n[8] alternative — install apt deps (libportaudio, libasound) on")
    print(f"     ubuntu-latest. May or may not help on a fully-headless runner")
    print(f"     where there's no ALSA config or PulseAudio server. The dummy")
    print(f"     backend bypasses the whole device-init path, so prefer that.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
