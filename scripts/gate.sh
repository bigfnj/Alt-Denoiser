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
# find_release: pick a RELEASE artefact, not merely the first one the filesystem
# enumerates. Stage 1 builds --config Release, but a multi-config generator keeps
# Debug output in the same tree, and a bare `head -n 1` could hand us that. It
# matters more than usual here: the harness deliberately drives paths guarded by
# jassertfalse (negative FIFO counts, refused geometry), so a Debug binary breaks
# into the debugger instead of testing anything.
find_release() {
    _hits=$(find "$BUILD_DIR" "$@" 2>/dev/null)
    _rel=$(printf '%s\n' "$_hits" | grep -i '/release/' | head -n 1)
    if [ -n "$_rel" ]; then printf '%s\n' "$_rel"; else printf '%s\n' "$_hits" | head -n 1; fi
}

HARNESS=$(find_release -type f \( -name AltDenoiserTests -o -name AltDenoiserTests.exe \))
if [ -z "$HARNESS" ]; then
    fail "AltDenoiserTests binary not found under $BUILD_DIR"
else
    printf '   harness:   %s\n' "$HARNESS"
    RUN=""
    if [ "${RUNNER_OS:-}" = "Linux" ] || { [ -z "${RUNNER_OS:-}" ] && [ "$(uname -s)" = "Linux" ]; }; then
        RUN="xvfb-run -a"
    fi
    # A timeout, because the defect class L8 guards is a HANG, not wrong audio:
    # an infinite resample ratio makes the resampler's inner loop never
    # terminate. Without this a regression there stalls the gate until the
    # job-level timeout rather than producing a red stage. 300 s is about 20x the
    # slowest observed geometry.
    TIMEOUT=""
    command -v timeout > /dev/null 2>&1 && TIMEOUT="timeout 300"

    # The geometries that have historically behaved differently: a block size
    # that is an exact multiple of the 480-sample model hop, ones that are not,
    # and rates needing real resampling in both directions.
    for cfg in "480 48000" "512 48000" "1024 48000" "512 44100" "1024 96000"; do
        if $RUN $TIMEOUT "$HARNESS" $cfg > /tmp/gate-harness.log 2>&1; then
            ok "harness $cfg"
        else
            _rc=$?
            if [ "$_rc" = "124" ]; then
                fail "harness $cfg TIMED OUT after 300 s"
            else
                fail "harness $cfg"
            fi
            grep -E '^\[FAIL' /tmp/gate-harness.log || tail -10 /tmp/gate-harness.log
        fi
    done
fi

#------------------------------------------------------------------------------
say "3. pluginval (strictness 5)"
BUNDLE=$(find_release -type d -name '*.vst3' -prune -print)
if [ -z "${PLUGINVAL:-}" ]; then
    skip "PLUGINVAL not set; format validation did not run"
# `||`, not `&&`. With `&&` an existing but non-executable file satisfied -f,
# skipped the branch, and pluginval was run anyway; the only input that reached
# this message was a path that did not exist, for which the message was wrong.
elif [ ! -x "${PLUGINVAL}" ] || [ ! -f "${PLUGINVAL}" ]; then
    fail "PLUGINVAL points at '${PLUGINVAL}', which is not an executable file"
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
# Checks every format this platform is expected to build, not just the VST3.
# Checking one format would silently under-verify macOS and Linux, where the
# workflow now stages two and a missing AU or LV2 would reach a release.
case "${RUNNER_OS:-$(uname -s)}" in
    Darwin|macOS) FORMATS="vst3 component" ;;
    Linux)        FORMATS="vst3 lv2" ;;
    *)            FORMATS="vst3" ;;
esac
for ext in $FORMATS; do
    # Restricted to Release for the same reason stage 2 is: a tree built for both
    # configurations holds two of each bundle and would fail this check for a
    # reason that has nothing to do with packaging.
    COUNT=$(find "$BUILD_DIR" -type d -name "*.${ext}" -prune -print 2>/dev/null \
            | grep -ic '/release/' | tr -d ' ')
    if [ "$COUNT" = "1" ]; then
        ok "exactly one .${ext} bundle"
    else
        fail "expected exactly one .${ext} bundle, found $COUNT"
    fi
done
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
