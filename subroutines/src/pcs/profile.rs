//! Lightweight profiling helper for all PCS backends.
//!
//! Controlled by env var `PCS_PROFILE=1` (or legacy `MULCS_PROFILE=1`).
//! When disabled, all operations are near-zero-cost.
//! Outputs CSV lines to stdout, unified 9-column schema:
//! `source,backend,nv,N,repeat,phase,elapsed_ms,count,notes`

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

static PROFILE_ACTIVE: AtomicBool = AtomicBool::new(false);
static PROFILE_CHECKED: AtomicBool = AtomicBool::new(false);

pub(crate) fn profiling_enabled() -> bool {
    if !PROFILE_CHECKED.load(Ordering::Relaxed) {
        let active = std::env::var("PCS_PROFILE").unwrap_or_default() == "1"
            || std::env::var("MULCS_PROFILE").unwrap_or_default() == "1";
        PROFILE_ACTIVE.store(active, Ordering::Relaxed);
        PROFILE_CHECKED.store(true, Ordering::Relaxed);
        active
    } else {
        PROFILE_ACTIVE.load(Ordering::Relaxed)
    }
}

/// Emit a profiling CSV row to stdout (unified 9-column schema).
fn emit_csv(backend: &str, nv: usize, n: usize, phase: &str, ms: f64, count: usize, notes: &str) {
    println!("pcs_internal,{backend},{nv},{n},0,{phase},{ms:.6},{count},{notes}");
}

#[allow(dead_code)]
pub(crate) fn emit_header_once() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if !DONE.swap(true, Ordering::Relaxed) {
        println!("source,backend,nv,N,repeat,phase,elapsed_ms,count,notes");
    }
}

/// A scoped timer. When profiling is disabled, does nothing.
pub(crate) struct ScopedTimer {
    backend: &'static str,
    nv: usize,
    n: usize,
    phase: &'static str,
    count: usize,
    notes: &'static str,
    start: Option<Instant>,
}

impl ScopedTimer {
    pub(crate) fn new(
        backend: &'static str,
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
            backend,
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
                self.backend,
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
pub(crate) fn emit_manual(
    backend: &str,
    nv: usize,
    n: usize,
    phase: &str,
    ms: f64,
    count: usize,
    notes: &str,
) {
    if profiling_enabled() {
        emit_csv(backend, nv, n, phase, ms, count, notes);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Zero-overhead manual timer (available for fine-grained profiling)
// ═══════════════════════════════════════════════════════════════════

#[allow(dead_code)]
pub(crate) struct MaybeTimer {
    active: bool,
    acc_ns: u128,
}

#[allow(dead_code)]
impl MaybeTimer {
    pub(crate) fn new() -> Self {
        MaybeTimer {
            active: profiling_enabled(),
            acc_ns: 0,
        }
    }
    pub(crate) fn start(&self) -> MaybeTick {
        MaybeTick {
            t0: if self.active {
                Some(Instant::now())
            } else {
                None
            },
        }
    }
    pub(crate) fn add(&mut self, tick: &MaybeTick) {
        if self.active {
            if let Some(t0) = tick.t0 {
                self.acc_ns += t0.elapsed().as_nanos();
            }
        }
    }
    pub(crate) fn ns(&self) -> u128 {
        self.acc_ns
    }
}

#[allow(dead_code)]
pub(crate) struct MaybeTick {
    t0: Option<Instant>,
}
