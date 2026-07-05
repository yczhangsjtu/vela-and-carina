//! Lightweight profiling helper for MulcsPCS.
//!
//! Controlled by env var `MULCS_PROFILE=1`. When disabled, all operations are
//! near-zero-cost (no Instant::now(), no env reads after first check).
//! Outputs CSV lines to stderr:
//! `mulcs_internal,<nv>,<N>,<phase>,<elapsed_ms>,<count>,<notes>`

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

/// Cached env flag — read once.
static PROFILE_ACTIVE: AtomicBool = AtomicBool::new(false);
static PROFILE_CHECKED: AtomicBool = AtomicBool::new(false);

pub(crate) fn profiling_enabled() -> bool {
    if !PROFILE_CHECKED.load(Ordering::Relaxed) {
        let active = std::env::var("MULCS_PROFILE").unwrap_or_default() == "1";
        PROFILE_ACTIVE.store(active, Ordering::Relaxed);
        PROFILE_CHECKED.store(true, Ordering::Relaxed);
        active
    } else {
        PROFILE_ACTIVE.load(Ordering::Relaxed)
    }
}

static HEADER_EMITTED: AtomicBool = AtomicBool::new(false);

/// Emit a profiling CSV row to stderr.
fn emit_csv(nv: usize, n: usize, phase: &str, ms: f64, count: usize, notes: &str) {
    if !HEADER_EMITTED.swap(true, Ordering::Relaxed) {
        eprintln!("# source,nv,N,phase,elapsed_ms,count,notes");
    }
    eprintln!("mulcs_internal,{nv},{n},{phase},{ms:.6},{count},{notes}");
}

/// A scoped timer. When profiling is disabled, does nothing (no Instant).
pub(crate) struct ScopedTimer {
    nv: usize,
    n: usize,
    phase: &'static str,
    count: usize,
    notes: &'static str,
    start: Option<Instant>,
}

impl ScopedTimer {
    pub(crate) fn new(
        nv: usize,
        n: usize,
        phase: &'static str,
        count: usize,
        notes: &'static str,
    ) -> Self {
        let start = if profiling_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        ScopedTimer {
            nv,
            n,
            phase,
            count,
            notes,
            start,
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            emit_csv(
                self.nv,
                self.n,
                self.phase,
                start.elapsed().as_secs_f64() * 1000.0,
                self.count,
                self.notes,
            );
        }
    }
}

/// Manual emit — for phases where you want to emit without a scoped guard.
#[allow(dead_code)]
pub(crate) fn emit_manual(nv: usize, n: usize, phase: &str, ms: f64, count: usize, notes: &str) {
    if profiling_enabled() {
        emit_csv(nv, n, phase, ms, count, notes);
    }
}
