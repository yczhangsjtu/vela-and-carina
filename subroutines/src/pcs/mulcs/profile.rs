//! Lightweight profiling helper for MulcsPCS.
//!
//! Controlled by env var `MULCS_PROFILE=1`. When disabled, all operations are
//! no-ops. Outputs CSV lines to stderr:
//! `mulcs_internal,<nv>,<N>,<phase>,<elapsed_ms>,<count>,<notes>`

use std::time::Instant;

/// Returns whether MULCS_PROFILE is enabled.
pub(crate) fn profiling_enabled() -> bool {
    std::env::var("MULCS_PROFILE").unwrap_or_default() == "1"
}

/// Emit a profiling CSV row to stderr.
pub(crate) fn emit_csv(nv: usize, n: usize, phase: &str, ms: f64, count: usize, notes: &str) {
    eprintln!("mulcs_internal,{nv},{n},{phase},{ms:.6},{count},{notes}",);
}

/// A scoped timer that records elapsed time for a named phase.
/// Drops emit CSV when MULCS_PROFILE=1.
pub(crate) struct ScopedTimer {
    nv: usize,
    n: usize,
    phase: &'static str,
    count: usize,
    notes: &'static str,
    start: Instant,
    active: bool,
}

impl ScopedTimer {
    pub(crate) fn new(
        nv: usize,
        n: usize,
        phase: &'static str,
        count: usize,
        notes: &'static str,
    ) -> Self {
        let active = profiling_enabled();
        ScopedTimer {
            nv,
            n,
            phase,
            count,
            notes,
            start: if active {
                Instant::now()
            } else {
                Instant::now()
            },
            active,
        }
    }

    pub(crate) fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if self.active {
            emit_csv(
                self.nv,
                self.n,
                self.phase,
                self.start.elapsed().as_secs_f64() * 1000.0,
                self.count,
                self.notes,
            );
        }
    }
}

/// Manual emit — for phases where you want to emit without a scoped guard.
pub(crate) fn emit_manual(nv: usize, n: usize, phase: &str, ms: f64, count: usize, notes: &str) {
    if profiling_enabled() {
        emit_csv(nv, n, phase, ms, count, notes);
    }
}

/// Emit a CSV header line once.
pub(crate) fn emit_header() {
    if profiling_enabled() {
        eprintln!("# source,nv,N,phase,elapsed_ms,count,notes");
    }
}
