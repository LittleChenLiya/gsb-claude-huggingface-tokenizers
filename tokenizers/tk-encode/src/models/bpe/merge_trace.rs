//! Optional tracing of the individual merges BPE applies while encoding a pretoken.
//!
//! Tracing is opt-in per encode. The merge engines are generic over [`MergeTracer`]: the hot path
//! is instantiated with [`NoTrace`], whose `record` is a no-op that inlines away, so tracing *off*
//! compiles to exactly the same merge loop as before and adds nothing. [`TraceBuffer`] is the
//! recording impl, appending one [`MergeStep`] per applied merge in application order.
use crate::models::bpe::tables::BpeTables;

/// One applied merge: the two adjacent tokens that merged, and the rank at which they merged.
///
/// Tokens are external vocabulary ids (the same ids the model emits), mapped out of the engines'
/// internal id space through the tables. `rank` is the merge's priority: lower ranks merge first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MergeStep {
    /// External vocabulary id of the left token.
    pub left: u32,
    /// External vocabulary id of the right token.
    pub right: u32,
    /// Rank of the merge (lower = applied earlier).
    pub rank: u32,
}

/// A sink a merge engine reports each applied merge to.
///
/// The two impls are [`NoTrace`] and [`TraceBuffer`]. Engines are generic over this rather than
/// taking an `Option`: with [`NoTrace`] the call compiles to nothing, so the tracing-off path is
/// byte-for-byte the merge loop there was before.
pub(super) trait MergeTracer {
    /// Reports a merge of `left` and `right` (internal symbol ids) at `rank`.
    fn record(&mut self, tables: &BpeTables, left: u32, right: u32, rank: u32);
}

/// Tracing disabled. Used on every hot encode path; `record` is a no-op that inlines away.
pub(super) struct NoTrace;

impl MergeTracer for NoTrace {
    #[inline(always)]
    fn record(&mut self, _tables: &BpeTables, _left: u32, _right: u32, _rank: u32) {}
}

/// Tracing enabled: applied merges are appended to [`Self::steps`], in application order.
pub(super) struct TraceBuffer<'a> {
    pub(super) steps: &'a mut Vec<MergeStep>,
}

impl MergeTracer for TraceBuffer<'_> {
    #[inline(always)]
    fn record(&mut self, tables: &BpeTables, left: u32, right: u32, rank: u32) {
        self.steps.push(MergeStep {
            left: tables.external(left).unwrap_or(left),
            right: tables.external(right).unwrap_or(right),
            rank,
        });
    }
}
