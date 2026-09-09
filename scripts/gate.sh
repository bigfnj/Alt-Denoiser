#!/usr/bin/env bash
#
# Full verification gate. Run this before every commit, and CI runs the same
# script, so a local pass and a CI pass mean the same thing.
#
#   scripts/gate.sh [build-dir]
#
# Environment:
#   PLUGINVAL   path to the pluginval binary. If unset the pluginval stage is
#               SKIPPED, and the script says so loudly and marks the run
#               DEGRADED rather than reporting a clean pass.
#   CMAKE       cmake to use. Defaults to whatever is on PATH, which is not
#               always the one that configured the build directory: on this
#               developer's Windows box Git Bash resolves a bundled cmake 3.31
#               that shadows the 4.3 which configured the tree, and 3.31 cannot
#               load its generator. The gate prints the version it used so that
#               kind of divergence is visible rather than mystifying.
#
# Exit codes: 0 all stages passed, 1 a stage failed.
#
# Deliberately not silent about partial runs. A gate that quietly skips a stage
# is worse than no gate, because it produces a green tick that means less than
# the reader assumes.

set -uo pipefail

BUILD_DIR="${1:-build}"
CMAKE="${CMAKE:-cmake}"
FAILURES=0
DEGRADED=0

say()  { printf '\n== %s ==\n' "$1"; }
fail() { printf '   FAIL: %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
skip() { printf '   SKIPPED (DEGRADED): %s\n' "$1"; DEGRADED=$((DEGRADED + 1)); }
ok()   { printf '   ok: %s\n' "$1"; }

#------------------------------------------------------------------------------
say "0. Toolchain"
printf '   cmake:     %s\n' "$("$CMAKE" --version 2>/dev/null | head -1)"
printf '   cmake at:  %s\n' "$(command -v "$CMAKE")"
printf '   build dir: %s\n' "$BUILD_DIR"
printf '   pluginval: %s\n' "${PLUGINVAL:-<unset>}"

say "1. Build (Release)"
if "$CMAKE" --build "$BUILD_DIR" --config Release --parallel 4 > /tmp/gate-build.log 2>&1; then
    ok "compiled"
else
    fail "build failed; last 20 lines follow"
    tail -20 /tmp/gate-build.log
    printf '\nGate aborted: nothing downstream is meaningful without a build.\n'
    exit 1
fi

#------------------------------------------------------------------------------
say "2. Offline harness across the geometry matrix"
HARNESS=$(find "$BUILD_DIR" -type f \( -name AltDenoiserTests -o -name AltDenoiserTests.exe \) | head -n 1)
if [ -z "$HARNESS" ]; then
    fail "AltDenoiserTests binary not found under $BUILD_DIR"
else
    RUN=""
    if [ "${RUNNER_OS:-}" = "Linux" ] || { [ -z "${RUNNER_OS:-}" ] && [ "$(uname -s)" = "Linux" ]; }; then
        RUN="xvfb-run -a"
    fi
    # The geometries that have historically behaved differently: a block size
    # that is an exact multiple of the 480-sample model hop, ones that are not,
    # and rates needing real resampling in both directions.
    for cfg in "480 48000" "512 48000" "1024 48000" "512 44100" "1024 96000"; do
        if $RUN "$HARNESS" $cfg > /tmp/gate-harness.log 2>&1; then
            ok "harness $cfg"
        else
            fail "harness $cfg"
            grep -E '^\[FAIL' /tmp/gate-harness.log || tail -10 /tmp/gate-harness.log
        fi
    done
fi

#------------------------------------------------------------------------------
say "3. pluginval (strictness 5)"
BUNDLE=$(find "$BUILD_DIR" -type d -name '*.vst3' -prune -print | head -n 1)
if [ -z "${PLUGINVAL:-}" ]; then
    skip "PLUGINVAL not set; format validation did not run"
elif [ ! -x "${PLUGINVAL}" ] && [ ! -f "${PLUGINVAL}" ]; then
    fail "PLUGINVAL points at '${PLUGINVAL}' which is not executable"
elif [ -z "$BUNDLE" ]; then
    fail "no .vst3 bundle found to validate"
else
    RUN=""
    if [ "${RUNNER_OS:-}" = "Linux" ] || { [ -z "${RUNNER_OS:-}" ] && [ "$(uname -s)" = "Linux" ]; }; then
        RUN="xvfb-run -a"
    fi
    if $RUN "$PLUGINVAL" --strictness-level 5 --validate-in-process \
            --timeout-ms 300000 --validate "$BUNDLE" > /tmp/gate-pluginval.log 2>&1; then
        ok "pluginval passed"
    else
        fail "pluginval failed"
        tail -20 /tmp/gate-pluginval.log
    fi
fi

#------------------------------------------------------------------------------
say "4. Packaging shape"
COUNT=$(find "$BUILD_DIR" -type d -name '*.vst3' -prune -print | wc -l | tr -d ' ')
if [ "$COUNT" = "1" ]; then
    ok "exactly one .vst3 bundle"
else
    fail "expected exactly one .vst3 bundle, found $COUNT"
fi
if [ -f LICENSE ]; then
    ok "LICENSE present for packaging"
else
    fail "LICENSE missing; the release package would ship an AGPLv3 binary without it"
fi

#------------------------------------------------------------------------------
printf '\n========================================\n'
if [ "$FAILURES" -ne 0 ]; then
    printf 'GATE FAILED: %d stage(s) failed.\n' "$FAILURES"
    exit 1
fi
if [ "$DEGRADED" -ne 0 ]; then
    printf 'GATE PASSED (DEGRADED): %d stage(s) skipped.\n' "$DEGRADED"
    printf 'This is NOT a clean pass. Set PLUGINVAL to run the full gate.\n'
    exit 0
fi
printf 'GATE PASSED: all stages ran and passed.\n'
