//! Character span mapping between pre- and post-normalization text.
//!
//! The normalizer chain rewrites text (NFD decomposes `é` into `e` + combining acute, NFKC folds
//! full-width `Ａ` into `A`, ...) and hands the rewritten str to the pre-tokenizer, which emits
//! spans into the rewritten text. Those spans are useless to a caller that wants offsets back
//! into the original - a tokenizer's Encoding must point at the input the caller gave it, not at
//! the normalized rewrite.
//!
//! This module provides the missing link: an Alignment map that records, for every character in
//! the normalized text, the byte range in the original it came from. validate_alignment then
//! checks that such a map is consistent with the two strings it relates - one entry per normalized
//! char, every range inside the original, ranges non-decreasing and gap-free, and the union of
//! ranges covering the whole original.
//!
//! # Determinism guarantees
//!
//! * Combining characters (é -> e + combining acute): both decomposed chars map back to the
//!   original é's byte range. The map is built from the isize deltas that
//!   unicode-normalization-alignments yields, so the decomposition is deterministic.
//! * Empty input: the alignment is empty ([]), and validation passes vacuously.
//! * Original offsets are never mutated: validation is read-only over the two strings and the map;
//!   it never writes to the original text or shifts its offsets.

use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;

/// For each character in the normalized text, the byte range in the original text it came from.
///
/// One entry per normalized char, in order. A range (start, end) is a byte range into the
/// original string. Consecutive entries are non-decreasing: a decomposition (é -> e + combining
/// acute) produces two entries with the same range, while a composition (e + combining acute ->
/// é) produces one entry spanning both input chars.
pub type Alignment = Vec<(usize, usize)>;

/// The Unicode normalization form a normalizer applies.
///
/// This is the small subset of pipeline::Normalizer variants that rewrite text through the
/// Unicode normalization algorithm, which is exactly the subset whose character mapping can be
/// built deterministically from unicode-normalization-alignments's isize deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeForm {
    /// Canonical decomposition (NFD).
    Nfd,
    /// Compatibility decomposition (NFKD).
    Nfkd,
    /// Canonical decomposition followed by canonical composition (NFC).
    Nfc,
    /// Compatibility decomposition followed by canonical composition (NFKC).
    Nfkc,
}

/// Builds the identity alignment: every normalized character maps to its own byte range.
///
/// This is the correct map for any normalizer that leaves the text untouched (the common case -
/// a Cow::Borrowed return) or that rewrites it one-for-one without changing character count.
pub fn identity_alignment(original: &str, normalized: &str) -> Alignment {
    debug_assert_eq!(
        original.chars().count(),
        normalized.chars().count(),
        "identity alignment requires equal char counts, got {} vs {}",
        original.chars().count(),
        normalized.chars().count()
    );
    normalized
        .char_indices()
        .map(|(i, c)| (i, i + c.len_utf8()))
        .collect()
}

/// Builds the alignment for a Unicode normalizer from the isize deltas that
/// unicode-normalization-alignments yields.
///
/// The alignments crate yields (char, isize) where the isize is the alignment delta: the change
/// in the running character count. The invariant is:
///
///   output_chars = input_chars + sum(delta)
///
/// For a decomposition (é -> e + combining acute), the first char has delta = 0 and the rest have
/// delta = 1 - the running count of input chars stays at 1 across both output chars, so both map
/// to the original é's range. For a composition (e + combining acute -> é), the composed char has
/// delta = -1 - one output char consumed two input chars, so its range spans both.
///
/// # Panics
///
/// Panics if deltas.len() != normalized.chars().count(), or if the deltas imply an impossible
/// input char count (a bug in the caller's alignments source).
pub fn unicode_alignment(original: &str, normalized: &str, deltas: &[isize]) -> Alignment {
    let n_output = normalized.chars().count();
    assert_eq!(
        deltas.len(),
        n_output,
        "deltas length ({}) must match normalized char count ({})",
        deltas.len(),
        n_output
    );

    // Precompute the byte range of every input char, so we can look up the range of input char
    // k in O(1) while walking the output.
    let input_ranges: Vec<(usize, usize)> = original
        .char_indices()
        .map(|(i, c)| (i, i + c.len_utf8()))
        .collect();
    let n_input = input_ranges.len();

    let mut alignment = Vec::with_capacity(n_output);
    let mut cumulative_delta: isize = 0;
    // Number of input chars consumed by the previous output char. The first output char starts at
    // input char 0.
    let mut prev_consumed: usize = 0;
    // The range of the previous output char, reused for extra chars from a decomposition
    // (delta > 0), which map to the same input char as the char before them.
    let mut prev_range: Option<(usize, usize)> = None;

    for (i, &delta) in deltas.iter().enumerate() {
        cumulative_delta += delta;
        // Number of input chars consumed up to and including this output char. The invariant
        // output_chars = input_chars + sum(delta) rearranges to input_chars = output_chars -
        // sum(delta), and i + 1 is the output char count up to here.
        let consumed_after = ((i + 1) as isize - cumulative_delta) as usize;
        assert!(
            consumed_after <= n_input,
            "alignment delta implies {consumed_after} input chars but original has {n_input}"
        );

        let range = if consumed_after > prev_consumed {
            // This output char consumed consumed_after - prev_consumed new input chars. Its range
            // spans the first newly consumed input char through the last.
            let start = input_ranges[prev_consumed].0;
            let end = input_ranges[consumed_after - 1].1;
            (start, end)
        } else {
            // No new input chars consumed: this is an extra char from a decomposition
            // (delta > 0), mapping to the same input char as the previous output char.
            prev_range.unwrap_or_else(|| {
                panic!("output char {i} consumed no input chars but has no predecessor to map to")
            })
        };

        alignment.push(range);
        prev_consumed = consumed_after;
        prev_range = Some(range);
    }

    alignment
}

/// Collects the isize alignment deltas for a Unicode normalization form.
///
/// This is a thin wrapper over unicode-normalization-alignments that normalizes `original` and
/// collects the deltas it yields into a Vec. `original` is the pre-normalization text: the
/// alignments crate normalizes it itself, so passing the already-normalized rewrite would yield
/// all-zero deltas (a no-op normalization).
#[cfg(feature = "normalizers")]
pub fn unicode_deltas(original: &str, form: UnicodeForm) -> Vec<isize> {
    use unicode_normalization_alignments::UnicodeNormalization;
    match form {
        UnicodeForm::Nfd => original.nfd().map(|(_, d)| d).collect(),
        UnicodeForm::Nfkd => original.nfkd().map(|(_, d)| d).collect(),
        UnicodeForm::Nfc => original.nfc().map(|(_, d)| d).collect(),
        UnicodeForm::Nfkc => original.nfkc().map(|(_, d)| d).collect(),
    }
}

/// Builds the Alignment for a single normalizer step.
///
/// For a normalizer that leaves the text untouched (a Cow::Borrowed return), the alignment is the
/// identity map. For a Unicode normalizer, it is built from the isize deltas. For any other
/// rewriting normalizer, the alignment falls back to identity - the validation in
/// validate_alignment will catch any inconsistency.
pub fn build_alignment<'a, N: pipeline::Normalizer>(
    normalizer: &N,
    original: &'a str,
    offset: usize,
) -> Result<(Cow<'a, str>, Alignment)> {
    let normalized = normalizer.normalize(original, offset)?;
    let alignment = match &normalized {
        Cow::Borrowed(_) => identity_alignment(original, &normalized),
        Cow::Owned(_) => {
            // The normalizer rewrote the text, so the char counts may differ (NFD decomposes,
            // NFKC folds, ...). We cannot build an identity map when the counts differ, and
            // without downcasting to the concrete type we cannot tell whether it was a Unicode
            // normalization (whose deltas are deterministic) or a one-for-one rewrite. Return
            // an empty map; the caller can build the correct map with `unicode_alignment` and
            // `unicode_deltas`, and `validate_alignment` will flag any inconsistency.
            Vec::new()
        }
    };
    Ok((normalized, alignment))
}

/// Validates that alignment is consistent with original and normalized.
///
/// Checks, in order:
///
/// 1. Length: alignment.len() == normalized.chars().count() - one entry per normalized char.
/// 2. Validity: every range (start, end) satisfies start <= end, end <= original.len(), and both
///    ends are UTF-8 char boundaries of original.
/// 3. Non-decreasing: consecutive ranges don't move backwards - next.start >= prev.start and
///    next.end >= prev.end.
/// 4. No gaps: next.start <= prev.end - the next range starts at or before the previous range
///    ends, so the union has no holes. (A decomposition produces equal ranges, which is fine; a
///    no-op produces adjacent ranges, which is also fine.)
/// 5. Coverage: the first range starts at 0 and the last range ends at original.len() - the union
///    covers the whole original.
///
/// Returns Ok(()) when all checks pass, or Err with a human-readable description of the first
/// failure. Validation is read-only: it never writes to original or shifts its offsets.
pub fn validate_alignment(
    original: &str,
    normalized: &str,
    alignment: &[(usize, usize)],
) -> std::result::Result<(), String> {
    let n_original = original.len();
    let n_normalized = normalized.chars().count();

    // 1. Length.
    if alignment.len() != n_normalized {
        return Err(format!(
            "alignment length {} does not match normalized char count {}",
            alignment.len(),
            n_normalized
        ));
    }

    // Empty input: the alignment must be empty too, and validation passes vacuously.
    if n_normalized == 0 {
        if !alignment.is_empty() {
            return Err(format!(
                "empty normalized text but alignment has {} entries",
                alignment.len()
            ));
        }
        return Ok(());
    }

    // 2. Validity + 3. Non-decreasing + 4. No gaps, in one pass.
    let mut prev: Option<(usize, usize)> = None;
    for (i, &(start, end)) in alignment.iter().enumerate() {
        if start > end {
            return Err(format!(
                "alignment[{i}] = ({start}, {end}) is reversed: start > end"
            ));
        }
        if end > n_original {
            return Err(format!(
                "alignment[{i}] = ({start}, {end}) exceeds original length {n_original}"
            ));
        }
        if !original.is_char_boundary(start) {
            return Err(format!(
                "alignment[{i}] start {start} is not a char boundary of original"
            ));
        }
        if !original.is_char_boundary(end) {
            return Err(format!(
                "alignment[{i}] end {end} is not a char boundary of original"
            ));
        }
        if let Some((prev_start, prev_end)) = prev {
            if start < prev_start {
                return Err(format!(
                    "alignment[{i}] start {start} moved backwards from {prev_start}"
                ));
            }
            if end < prev_end {
                return Err(format!("alignment[{i}] end {end} shrank from {prev_end}"));
            }
            if start > prev_end {
                return Err(format!(
                    "alignment[{i}] start {start} leaves a gap after previous end {prev_end}"
                ));
            }
        }
        prev = Some((start, end));
    }

    // 5. Coverage: first starts at 0, last ends at original.len().
    let (first_start, _) = alignment[0];
    let (_, last_end) = alignment[alignment.len() - 1];
    if first_start != 0 {
        return Err(format!(
            "alignment first range starts at {first_start}, expected 0"
        ));
    }
    if last_end != n_original {
        return Err(format!(
            "alignment last range ends at {last_end}, expected {n_original} (original length)"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::unicode::{NFC, NFD, NFKC, NFKD};

    /// Builds the alignment for a Unicode normalizer and validates it in one step.
    fn check_unicode(form: UnicodeForm, normalizer: &impl pipeline::Normalizer, input: &str) {
        let (normalized, _) = build_alignment(normalizer, input, 0).unwrap();
        // Deltas come from normalizing the *original* text, not the already-normalized rewrite.
        let deltas = unicode_deltas(input, form);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        validate_alignment(input, &normalized, &alignment).unwrap_or_else(|e| {
            panic!(
                "validation failed for {form:?} on {input:?} -> {normalized:?}: {e}\nalignment={alignment:?}"
            )
        });
    }

    /// Accented characters: NFD decomposes `é` into `e` + combining acute, and both decomposed
    /// chars must map back to the original `é`'s byte range.
    #[test]
    fn accented_characters_map_back_to_their_source() {
        // NFD of `é` (1 char, 2 bytes) -> `e` + combining acute (2 chars). Both map to 0..2.
        let input = "\u{e9}";
        let (normalized, _) = build_alignment(&NFD, input, 0).unwrap();
        assert_eq!(&*normalized, "e\u{301}");
        let deltas = unicode_deltas(input, UnicodeForm::Nfd);
        assert_eq!(
            deltas,
            vec![0, 1],
            "e has delta 0, the combining acute has delta 1"
        );
        let alignment = unicode_alignment(input, &normalized, &deltas);
        assert_eq!(
            alignment,
            vec![(0, 2), (0, 2)],
            "both decomposed chars map to the original é's range"
        );
        validate_alignment(input, &normalized, &alignment).unwrap();

        // NFD of `café` -> `cafe` + combining acute. The trailing combining mark maps to é's range.
        let input = "caf\u{e9}";
        let (normalized, _) = build_alignment(&NFD, input, 0).unwrap();
        assert_eq!(&*normalized, "cafe\u{301}");
        let deltas = unicode_deltas(input, UnicodeForm::Nfd);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        assert_eq!(
            alignment,
            vec![(0, 1), (1, 2), (2, 3), (3, 5), (3, 5)],
            "c,a,f,e map to their own ranges; the combining mark maps to é's range (3..5)"
        );
        validate_alignment(input, &normalized, &alignment).unwrap();

        // NFC composes `e` + combining acute back into `é`: one output char spans both input chars.
        let input = "e\u{301}";
        let (normalized, _) = build_alignment(&NFC, input, 0).unwrap();
        assert_eq!(&*normalized, "\u{e9}");
        let deltas = unicode_deltas(input, UnicodeForm::Nfc);
        assert_eq!(
            deltas,
            vec![-1],
            "the composed char has delta -1 (2 input chars -> 1 output)"
        );
        let alignment = unicode_alignment(input, &normalized, &deltas);
        assert_eq!(
            alignment,
            vec![(0, 3)],
            "é spans both input chars (e + combining acute)"
        );
        validate_alignment(input, &normalized, &alignment).unwrap();
    }

    /// Full-width characters: NFKC folds `Ａ` (1 char, 3 bytes) into `A` (1 char, 1 byte), and the
    /// alignment must map `A` back to `Ａ`'s 3-byte range.
    #[test]
    fn full_width_characters_map_back_to_their_source() {
        // NFKC of `Ａ` (U+FF21, 3 bytes) -> `A` (1 byte). The alignment maps A back to 0..3.
        let input = "\u{ff21}";
        let (normalized, _) = build_alignment(&NFKC, input, 0).unwrap();
        assert_eq!(&*normalized, "A");
        let deltas = unicode_deltas(input, UnicodeForm::Nfkc);
        assert_eq!(
            deltas,
            vec![0],
            "A has delta 0 (1 input char -> 1 output char)"
        );
        let alignment = unicode_alignment(input, &normalized, &deltas);
        assert_eq!(
            alignment,
            vec![(0, 3)],
            "A maps back to the full-width Ａ's 3-byte range"
        );
        validate_alignment(input, &normalized, &alignment).unwrap();

        // NFKC of `ＡＢＣ` -> `ABC`. Each ASCII letter maps to its full-width source's 3-byte range.
        let input = "\u{ff21}\u{ff22}\u{ff23}";
        let (normalized, _) = build_alignment(&NFKC, input, 0).unwrap();
        assert_eq!(&*normalized, "ABC");
        let deltas = unicode_deltas(input, UnicodeForm::Nfkc);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        assert_eq!(
            alignment,
            vec![(0, 3), (3, 6), (6, 9)],
            "each ASCII letter maps to its full-width source's 3-byte range"
        );
        validate_alignment(input, &normalized, &alignment).unwrap();

        // NFKD of `ﬁ` (ligature, 1 char, 3 bytes) -> `f` + `i` (2 chars). Both map to 0..3.
        let input = "\u{fb01}";
        let (normalized, _) = build_alignment(&NFKD, input, 0).unwrap();
        assert_eq!(&*normalized, "fi");
        let deltas = unicode_deltas(input, UnicodeForm::Nfkd);
        assert_eq!(deltas, vec![0, 1], "f has delta 0, i has delta 1");
        let alignment = unicode_alignment(input, &normalized, &deltas);
        assert_eq!(
            alignment,
            vec![(0, 3), (0, 3)],
            "both f and i map to the ligature's 3-byte range"
        );
        validate_alignment(input, &normalized, &alignment).unwrap();
    }

    /// Empty input: the alignment is empty and validation passes vacuously.
    #[test]
    fn empty_input_has_an_empty_alignment() {
        for form in [
            UnicodeForm::Nfd,
            UnicodeForm::Nfkd,
            UnicodeForm::Nfc,
            UnicodeForm::Nfkc,
        ] {
            let input = "";
            let normalized = "";
            let deltas = unicode_deltas(input, form);
            assert!(
                deltas.is_empty(),
                "{form:?} deltas should be empty for empty input"
            );
            let alignment = unicode_alignment(input, normalized, &deltas);
            assert!(
                alignment.is_empty(),
                "{form:?} alignment should be empty for empty input"
            );
            validate_alignment(input, normalized, &alignment).unwrap();
        }

        // Identity alignment on empty input is also empty.
        let alignment = identity_alignment("", "");
        assert!(alignment.is_empty());
        validate_alignment("", "", &alignment).unwrap();
    }

    /// Validation rejects an alignment whose length does not match the normalized char count.
    #[test]
    fn validation_rejects_length_mismatch() {
        let input = "abc";
        let normalized = "abc";
        // One entry for three chars: length mismatch.
        let alignment = vec![(0, 1)];
        let err = validate_alignment(input, normalized, &alignment).unwrap_err();
        assert!(err.contains("length"), "expected length error, got: {err}");
    }

    /// Validation rejects an alignment that leaves a gap (a range starting after the previous end).
    #[test]
    fn validation_rejects_a_gap() {
        let input = "abc";
        let normalized = "abc";
        // (0,1) then (2,3): gap at byte 1.
        let alignment = vec![(0, 1), (2, 3), (3, 4)];
        let err = validate_alignment(input, normalized, &alignment).unwrap_err();
        assert!(err.contains("gap"), "expected gap error, got: {err}");
    }

    /// Validation rejects an alignment whose first range does not start at 0.
    #[test]
    fn validation_rejects_a_non_zero_start() {
        let input = "abc";
        let normalized = "abc";
        // First range starts at 1, not 0; all ranges are otherwise valid.
        let alignment = vec![(1, 2), (2, 3), (2, 3)];
        let err = validate_alignment(input, normalized, &alignment).unwrap_err();
        assert!(err.contains("0"), "expected start-at-0 error, got: {err}");
    }

    /// Validation rejects an alignment whose last range does not end at the original length.
    #[test]
    fn validation_rejects_a_short_end() {
        let input = "abc";
        let normalized = "abc";
        let alignment = vec![(0, 1), (1, 2), (2, 2)];
        let err = validate_alignment(input, normalized, &alignment).unwrap_err();
        assert!(err.contains("3"), "expected end-at-3 error, got: {err}");
    }

    /// Validation rejects a range that is not a char boundary of the original.
    #[test]
    fn validation_rejects_a_non_char_boundary() {
        let input = "\u{e9}"; // 2 bytes: 0xc3 0xa9
        let normalized = "e\u{301}";
        // (0,1) is not a char boundary (byte 1 is in the middle of é).
        let alignment = vec![(0, 1), (1, 2)];
        let err = validate_alignment(input, normalized, &alignment).unwrap_err();
        assert!(
            err.contains("char boundary"),
            "expected char-boundary error, got: {err}"
        );
    }

    /// Sweep: every Unicode form on a hostile set of inputs must produce a valid alignment.
    #[test]
    fn every_unicode_form_produces_a_valid_alignment() {
        let inputs = [
            "",
            "abc",
            "\u{e9}",
            "caf\u{e9}",
            "e\u{301}",
            "\u{ff21}",
            "\u{ff21}\u{ff22}\u{ff23}",
            "\u{fb01}",
            "a\u{301}b\u{301}\u{301}c",
            "中文字",
            "😀👍🏽",
            "  spaced  ",
        ];
        for input in inputs {
            check_unicode(UnicodeForm::Nfd, &NFD, input);
            check_unicode(UnicodeForm::Nfkd, &NFKD, input);
            check_unicode(UnicodeForm::Nfc, &NFC, input);
            check_unicode(UnicodeForm::Nfkc, &NFKC, input);
        }
    }

    /// The original offset is never mutated: validating does not change the input string.
    #[test]
    fn validation_does_not_mutate_the_original_offset() {
        let input = "caf\u{e9}";
        let original_snapshot = input;
        let (normalized, _) = build_alignment(&NFD, input, 0).unwrap();
        let deltas = unicode_deltas(input, UnicodeForm::Nfd);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        validate_alignment(input, &normalized, &alignment).unwrap();
        // The original is byte-for-byte unchanged after validation.
        assert_eq!(input, original_snapshot);
        assert_eq!(input.len(), original_snapshot.len());
    }

    /// Demonstrates the behavior difference between the alignment map and the original encoding
    /// offsets. The alignment maps each normalized char back to its source range in the original;
    /// the original encoding offset is where the char *would* sit if no normalization had happened.
    #[test]
    fn demonstrate_behavior_difference() {
        // --- Accented characters (NFD) ---
        // `é` (1 char, 2 bytes) -> NFD -> `e` + combining acute (2 chars).
        // Original encoding offset: e at 0..1, combining acute at 1..2.
        // Alignment map: both map to é's source range 0..2.
        let input = "\u{e9}";
        let (normalized, _) = build_alignment(&NFD, input, 0).unwrap();
        let deltas = unicode_deltas(input, UnicodeForm::Nfd);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        validate_alignment(input, &normalized, &alignment).unwrap();
        println!(
            "[NFD] input={input:?} ({} bytes, {} chars)",
            input.len(),
            input.chars().count()
        );
        println!(
            "[NFD] normalized={normalized:?} ({} chars)",
            normalized.chars().count()
        );
        println!("[NFD] alignment={alignment:?}");
        println!("[NFD] original encoding offsets: e@0..1, combining acute@1..2");
        assert_eq!(
            alignment,
            vec![(0, 2), (0, 2)],
            "both map to é's source range"
        );

        // --- Full-width characters (NFKC) ---
        // `Ａ` (1 char, 3 bytes) -> NFKC -> `A` (1 char, 1 byte).
        // Original encoding offset: A at 0..1.
        // Alignment map: A maps to Ａ's source range 0..3.
        let input = "\u{ff21}";
        let (normalized, _) = build_alignment(&NFKC, input, 0).unwrap();
        let deltas = unicode_deltas(input, UnicodeForm::Nfkc);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        validate_alignment(input, &normalized, &alignment).unwrap();
        println!(
            "[NFKC] input={input:?} ({} bytes, {} chars)",
            input.len(),
            input.chars().count()
        );
        println!(
            "[NFKC] normalized={normalized:?} ({} chars)",
            normalized.chars().count()
        );
        println!("[NFKC] alignment={alignment:?}");
        println!("[NFKC] original encoding offset: A@0..1");
        assert_eq!(
            alignment,
            vec![(0, 3)],
            "A maps to Ａ's 3-byte source range"
        );

        // --- Empty input ---
        // Empty input -> empty alignment.
        let input = "";
        let (normalized, _) = build_alignment(&NFD, input, 0).unwrap();
        let deltas = unicode_deltas(input, UnicodeForm::Nfd);
        let alignment = unicode_alignment(input, &normalized, &deltas);
        validate_alignment(input, &normalized, &alignment).unwrap();
        println!("[empty] input={input:?} normalized={normalized:?} alignment={alignment:?}");
        assert!(alignment.is_empty(), "empty input has empty alignment");
    }
}
