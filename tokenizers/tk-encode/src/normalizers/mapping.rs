//! Normalization char-span mapping and its verification entry point.
//!
//! The encode path normalizes text *before* it is tokenized (see
//! `tokenizer/pipeline/normalizer.rs`), so every offset an encoded `Encoding` reports has to be
//! mapped back from normalized text to what the caller actually passed in. This module produces
//! that mapping for the table-backed Unicode normalizers ([`NFD`], [`NFC`], [`NFKD`], [`NFKC`])
//! and exposes [`check_normalization`], the single entry point other code -- and the tests that
//! guard offsets -- use to validate it.
//!
//! ## Byte spans, not char indices
//!
//! Encoding offsets are byte offsets in the original sequence, so every span here is a byte range
//! too. One [`CharMapping`] per output character records both where the character sits in the
//! normalized text (`normalized`) and which bytes it came from (`original`). Running the output
//! spans together must tile the normalized string exactly; the original spans must *cover* every
//! original byte, and are allowed to overlap: when one original character fans out into several
//! (NFD of `é` gives `e` + U+0301), each emitted character reports the whole original cluster, so
//! asking for either one's offset returns the full `é`. That covering is what `check_normalization`
//! enforces.
//!
//! The alignment arithmetic (a running byte `offset`, `1` for an inserted character, `-N` when a
//! character also removes `N` following characters) is the one `unicode-normalization-alignments`
//! implements and the released `tokenizers` `NormalizedString::transform_range` used to consume;
//! reproducing it here, independently of the legacy code, is what makes this a *check* rather than
//! just another copy of the same assumption.
//!
//! ## Deterministic edge cases
//!
//! Two cases are pinned by construction and asserted by the tests:
//!
//! * **Combining characters.** A combining mark emitted on its own by a decomposition (e.g.
//!   NFD of `é` emits `e` then U+0301) is an inserted character and shares the original span of
//!   the character immediately before it. It is never reported as an empty or out-of-range span.
//! * **Empty input.** Normalizing `""` returns `""` and an empty mapping list: there is no
//!   character to map, and a zero-length range at byte zero round-trips on both sides.
//!
//! ## The pipeline `offset` argument is an origin, not a shift
//!
//! [`Normalizer::normalize`](crate::tokenizer::pipeline::Normalizer::normalize) takes the byte
//! position where `input` starts inside its sequence. Unicode normalizers ignore it; only
//! position-sensitive steps such as `Metaspace`/`Prepend` read it. [`check_normalization`] passes
//! it through unchanged and records it in the report, while every returned span stays relative to
//! `input` -- the mapping never bakes the origin offset in. "Cannot change the original offset"
//! means exactly that.

use std::borrow::Cow;
use std::ops::Range;

use unicode_normalization::{
    IsNormalized, is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick,
};
use unicode_normalization_alignments::UnicodeNormalization as AlignedUnicodeNormalization;

use crate::tokenizer::pipeline::Normalizer;

/// The Unicode normalization form a mapping is built or checked for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnicodeForm {
    /// Canonical decomposition.
    NFD,
    /// Canonical decomposition, followed by canonical composition.
    NFC,
    /// Compatibility decomposition.
    NFKD,
    /// Compatibility decomposition, followed by canonical composition.
    NFKC,
}

impl UnicodeForm {
    /// Every form, in the order they are tried when the caller does not state which form a
    /// normalizer implements. Already-normalized text often satisfies more than one form ("abc" is
    /// NFD *and* NFC), so [`check_normalization`] accepts the first whose reference normalization
    /// reproduces the normalizer's output.
    pub fn all() -> [UnicodeForm; 4] {
        [Self::NFD, Self::NFC, Self::NFKD, Self::NFKC]
    }

    /// Normalize `input` in this form, borrowing when it already is in the form.
    fn normalize(self, input: &str) -> Cow<'_, str> {
        let already = match self {
            Self::NFD => matches!(is_nfd_quick(input.chars()), IsNormalized::Yes),
            Self::NFC => matches!(is_nfc_quick(input.chars()), IsNormalized::Yes),
            Self::NFKD => matches!(is_nfkd_quick(input.chars()), IsNormalized::Yes),
            Self::NFKC => matches!(is_nfkc_quick(input.chars()), IsNormalized::Yes),
        };
        if already {
            return Cow::Borrowed(input);
        }
        let owned = match self {
            Self::NFD => input.nfd().map(|(c, _)| c).collect(),
            Self::NFC => input.nfc().map(|(c, _)| c).collect(),
            Self::NFKD => input.nfkd().map(|(c, _)| c).collect(),
            Self::NFKC => input.nfkc().map(|(c, _)| c).collect(),
        };
        Cow::Owned(owned)
    }

    /// Iterator of `(output char, change)` carrying the alignment hints; see the module docs.
    fn aligned(self, input: &str) -> Box<dyn Iterator<Item = (char, isize)> + '_> {
        match self {
            Self::NFD => Box::new(input.nfd()),
            Self::NFC => Box::new(input.nfc()),
            Self::NFKD => Box::new(input.nfkd()),
            Self::NFKC => Box::new(input.nfkc()),
        }
    }
}

/// Where one character of the normalized text came from in the original text.
///
/// Both ranges are **byte** ranges, exclusive of their end. `original` covers at least one byte:
/// inserted combining marks share the span of the character in front of them instead of pointing
/// at an empty range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharMapping {
    /// The character as it appears after normalization.
    pub character: char,
    /// Its byte span in the normalized string.
    pub normalized: Range<usize>,
    /// The byte span of the original character(s) it maps back to.
    pub original: Range<usize>,
}

/// What [`check_normalization`] found: the normalized text, its per-character mapping, and the
/// origin offset that was handed to the normalizer (left untouched by the check itself).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingReport {
    /// The text the normalizer produced.
    pub normalized: String,
    /// One entry per character of `normalized`, in order.
    pub mappings: Vec<CharMapping>,
    /// The `offset` the check passed to the normalizer, echoed for assertions comparing two
    /// origins. The spans in `mappings` are never shifted by it.
    pub origin_offset: usize,
}

impl MappingReport {
    /// Map a byte range of the normalized string back to the original string.
    ///
    /// The returned range is the smallest original span covering every mapping the range touches.
    /// A zero-length range at a char boundary round-trips to the corresponding zero-length point.
    /// Returns `None` for a range that is out of bounds or not on a char boundary.
    pub fn map_range_back(&self, range: Range<usize>) -> Option<Range<usize>> {
        let normalized = &self.normalized;
        if range.start > range.end
            || range.end > normalized.len()
            || !normalized.is_char_boundary(range.start)
            || !normalized.is_char_boundary(range.end)
        {
            return None;
        }
        let mut start = usize::MAX;
        let mut end = 0;
        for m in &self.mappings {
            if m.normalized.start < range.end && m.normalized.end > range.start {
                start = start.min(m.original.start);
                end = end.max(m.original.end);
            }
        }
        if start == usize::MAX {
            // Empty range between/around characters: translate the point by the next character,
            // or by the end of the last character when the point is at the very end.
            let point = self
                .mappings
                .iter()
                .find(|m| m.normalized.start >= range.start)
                .map(|m| m.original.start)
                .unwrap_or_else(|| self.mappings.last().map(|m| m.original.end).unwrap_or(0));
            Some(point..point)
        } else {
            Some(start..end)
        }
    }
}

/// Why a normalization mapping failed verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MappingError {
    /// The normalizer itself returned an error.
    Normalizer(String),
    /// The normalizer's output does not match the reference normalization.
    UnexpectedOutput {
        /// What the normalizer produced.
        got: String,
        /// What the reference normalization would have produced.
        expected: String,
    },
    /// The per-character spans do not tile the normalized string byte for byte.
    NonTilingNormalized(Range<usize>),
    /// The original spans do not tile the original string byte for byte.
    NonTilingOriginal(Range<usize>),
    /// A claimed span is not a valid byte range of the string it indexes into.
    OutOfBounds {
        /// The offending span.
        range: Range<usize>,
        /// Length of the string it was indexing into.
        len: usize,
    },
}

impl std::fmt::Display for MappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normalizer(e) => write!(f, "normalizer failed: {e}"),
            Self::UnexpectedOutput { got, expected } => write!(
                f,
                "normalizer output {got:?} does not match the reference normalization {expected:?}"
            ),
            Self::NonTilingNormalized(r) => {
                write!(f, "normalized spans do not tile at bytes {r:?}")
            }
            Self::NonTilingOriginal(r) => write!(f, "original spans do not tile at bytes {r:?}"),
            Self::OutOfBounds { range, len } => {
                write!(f, "span {range:?} is out of bounds for a {len}-byte string")
            }
        }
    }
}

impl std::error::Error for MappingError {}

/// Build the per-character mapping of normalizing `input` in `form`, without running any pipeline
/// normalizer: this is the independent reference path [`check_normalization`] compares against.
///
/// Empty input yields an empty mapping, deterministically.
pub fn char_mappings(form: UnicodeForm, input: &str) -> Vec<CharMapping> {
    if input.is_empty() {
        return Vec::new();
    }

    // Mirror `NormalizedString::transform_range` in the released crate: a running byte offset into
    // the original; `change == 1` marks an inserted character that shares the previous character's
    // alignment, `change <= 0` marks a character replacing the original character at `offset` (and,
    // for a negative value, also consuming the following `-change` characters). The offset moves in
    // UTF-8 byte *lengths*, not byte values.
    let mut original_chars = input.chars();
    let mut offset: usize = 0;
    let mut prev_align: Range<usize> = 0..0;
    let mut normalized_byte = 0;
    let mut mappings = Vec::new();

    for (character, change) in form.aligned(input) {
        let original = if change.is_positive() {
            // Inserted character (typically a combining mark a decomposition introduced): keep
            // the span of the character immediately in front. Pinned combining-mark rule.
            prev_align.clone()
        } else {
            let start = offset;
            let replaced = original_chars
                .next()
                .expect("aligned output with non-positive change replaces an original char");
            offset += replaced.len_utf8();
            for _ in 0..(-change) as usize {
                offset += original_chars
                    .next()
                    .expect("negative change consumes following original chars")
                    .len_utf8();
            }
            start..offset
        };
        let normalized = normalized_byte..normalized_byte + character.len_utf8();
        normalized_byte += character.len_utf8();
        prev_align = original.clone();
        mappings.push(CharMapping {
            character,
            normalized,
            original,
        });
    }
    mappings
}

/// Identity mapping of `input`: every character maps to its own span. This is what offsets look
/// like before normalization, so acceptance comparisons map the normalized spans back and assert
/// they land on these ranges.
pub fn identity_mappings(input: &str) -> Vec<CharMapping> {
    input
        .char_indices()
        .map(|(start, character)| CharMapping {
            character,
            normalized: start..start + character.len_utf8(),
            original: start..start + character.len_utf8(),
        })
        .collect()
}

fn validate_tiling(
    input: &str,
    normalized: &str,
    mappings: &[CharMapping],
) -> Result<(), MappingError> {
    // Every span must be in bounds and on a char boundary of the side it indexes into.
    for m in mappings {
        if m.normalized.end > normalized.len()
            || !normalized.is_char_boundary(m.normalized.start)
            || !normalized.is_char_boundary(m.normalized.end)
        {
            return Err(MappingError::OutOfBounds {
                range: m.normalized.clone(),
                len: normalized.len(),
            });
        }
        if m.original.end > input.len()
            || !input.is_char_boundary(m.original.start)
            || !input.is_char_boundary(m.original.end)
            || m.original.start >= m.original.end
        {
            return Err(MappingError::OutOfBounds {
                range: m.original.clone(),
                len: input.len(),
            });
        }
    }
    // The normalized spans tile the output exactly, in order.
    let mut cursor = 0;
    for m in mappings {
        if m.normalized.start != cursor {
            return Err(MappingError::NonTilingNormalized(
                cursor..m.normalized.start,
            ));
        }
        cursor = m.normalized.end;
    }
    if cursor != normalized.len() {
        return Err(MappingError::NonTilingNormalized(cursor..normalized.len()));
    }
    // The original spans cover every original byte; overlaps are legal, because an original
    // character that fans out into several output characters (NFD of "é" -> "e" + U+0301)
    // contributes the same full span to each of them.
    let mut covered = vec![false; input.len()];
    for m in mappings {
        for b in &mut covered[m.original.clone()] {
            *b = true;
        }
    }
    if let Some(missing) = covered.iter().position(|c| !*c) {
        return Err(MappingError::NonTilingOriginal(missing..missing + 1));
    }
    Ok(())
}

/// The verification entry point: run `normalizer` on `input`, build the reference mapping for
/// `expected` (or auto-detect the Unicode form when `None`), and validate that
///
/// 1. the normalizer's text equals the reference normalization;
/// 2. the emitted per-character spans tile the normalized text exactly and cover every byte of
///    the original text (spans may overlap when one original character expands into several);
/// 3. every span stays relative to `input` -- the returned [`MappingReport`] echoes `offset` but
///    never shifts a span by it, so the original offsets are unchanged.
///
/// Empty input is a supported, deterministic case: the report carries two empty strings and an
/// empty mapping.
pub fn check_normalization<N: Normalizer + ?Sized>(
    normalizer: &N,
    input: &str,
    offset: usize,
    expected: Option<UnicodeForm>,
) -> Result<MappingReport, MappingError> {
    let produced = normalizer
        .normalize(input, offset)
        .map_err(|e| MappingError::Normalizer(e.to_string()))?;

    // The stated form wins; otherwise accept the first form whose reference normalization
    // reproduces exactly what the normalizer produced.
    let form = match expected {
        Some(form) => Some(form),
        None => UnicodeForm::all()
            .into_iter()
            .find(|form| form.normalize(input).as_ref() == &*produced),
    };

    let Some(form) = form else {
        return Err(MappingError::UnexpectedOutput {
            got: produced.into_owned(),
            expected: String::new(),
        });
    };

    let reference = form.normalize(input);
    if reference.as_ref() != &*produced {
        return Err(MappingError::UnexpectedOutput {
            got: produced.into_owned(),
            expected: reference.into_owned(),
        });
    }

    let mappings = char_mappings(form, input);
    validate_tiling(input, &produced, &mappings)?;

    Ok(MappingReport {
        normalized: produced.into_owned(),
        mappings,
        origin_offset: offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::{NFC, NFD, NFKC, NFKD, Nmt};

    /// Byte spans of each original char: the "original encode offsets" a mapping has to land on.
    fn original_ranges(input: &str) -> Vec<Range<usize>> {
        identity_mappings(input)
            .into_iter()
            .map(|m| m.original)
            .collect()
    }

    #[test]
    fn nfd_precomposed_accent_marks_inherit_base_span() {
        // "é" is 2 bytes; NFD splits it into "e" + U+0301. The combining mark is inserted and
        // must share the base's full original span instead of getting an empty/zero span.
        let input = "\u{e9}";
        let report = check_normalization(&NFD, input, 0, Some(UnicodeForm::NFD)).unwrap();
        assert_eq!(report.normalized, "e\u{301}");
        assert_eq!(report.mappings.len(), 2);
        assert_eq!(report.mappings[0].character, 'e');
        assert_eq!(report.mappings[0].normalized, 0..1);
        assert_eq!(report.mappings[1].character, '\u{301}');
        assert_eq!(report.mappings[1].normalized, 1..3);
        // Both output characters point back at the full 2-byte "é".
        assert_eq!(report.mappings[0].original, 0..2);
        assert_eq!(report.mappings[1].original, 0..2);
        // Mapped back, the output covers exactly the original encode offset of "é".
        assert_eq!(report.map_range_back(0..3), Some(0..2));
        assert_eq!(original_ranges(input), vec![0..2]);
    }

    #[test]
    fn nfc_combining_sequence_composes_onto_full_cluster() {
        // "e" + U+0301 composes into "é": one output char covering both original bytes.
        let input = "e\u{301}";
        let report = check_normalization(&NFC, input, 0, Some(UnicodeForm::NFC)).unwrap();
        assert_eq!(report.normalized, "\u{e9}");
        assert_eq!(report.mappings.len(), 1);
        assert_eq!(report.mappings[0].normalized, 0..2);
        assert_eq!(report.mappings[0].original, 0..3);
        assert_eq!(report.map_range_back(0..2), Some(0..3));
        assert_eq!(original_ranges(input), vec![0..1, 1..3]);
    }

    #[test]
    fn nfkc_fullwidth_ascii_folds_to_ascii_same_span() {
        // "Ａ" (U+FF21, 3 bytes) compatibility-folds to "A" (1 byte), one-to-one onto its span.
        let input = "\u{ff21}";
        let report = check_normalization(&NFKC, input, 0, Some(UnicodeForm::NFKC)).unwrap();
        assert_eq!(report.normalized, "A");
        assert_eq!(report.mappings.len(), 1);
        assert_eq!(report.mappings[0].normalized, 0..1);
        assert_eq!(report.mappings[0].original, 0..3);
        assert_eq!(report.map_range_back(0..1), Some(0..3));
        assert_eq!(original_ranges(input), vec![0..3]);
    }

    #[test]
    fn nfkc_fullwidth_digit_in_a_sentence() {
        let input = "x\u{ff10}z"; // x, fullwidth "０", z
        let report = check_normalization(&NFKC, input, 0, Some(UnicodeForm::NFKC)).unwrap();
        assert_eq!(report.normalized, "x0z");
        let originals: Vec<_> = report.mappings.iter().map(|m| m.original.clone()).collect();
        // The 3-byte fullwidth digit sits at bytes 1..4 in both directions.
        assert_eq!(originals, vec![0..1, 1..4, 4..5]);
        assert_eq!(original_ranges(input), vec![0..1, 1..4, 4..5]);
    }

    #[test]
    fn nfkd_ligature_expands_with_inserted_char_sharing_span() {
        // "ﬁ" (U+FB01, 3 bytes) -> "fi": the "i" is inserted and shares "f"'s full span, so both
        // output characters map back onto the whole ligature while every original byte is covered.
        let input = "\u{fb01}";
        let report = check_normalization(&NFKD, input, 0, Some(UnicodeForm::NFKD)).unwrap();
        assert_eq!(report.normalized, "fi");
        assert_eq!(report.mappings[0].original, 0..3);
        assert_eq!(report.mappings[1].original, 0..3);
        assert_eq!(report.map_range_back(0..2), Some(0..3));
        assert_eq!(original_ranges(input), vec![0..3]);
    }

    #[test]
    fn empty_input_has_an_empty_deterministic_mapping() {
        for form in UnicodeForm::all() {
            let report = check_normalization(&NFD, "", 0, Some(form)).unwrap();
            assert_eq!(report.normalized, "");
            assert!(report.mappings.is_empty());
            assert_eq!(report.map_range_back(0..0), Some(0..0));
        }
        assert!(char_mappings(UnicodeForm::NFC, "").is_empty());
        assert!(identity_mappings("").is_empty());
    }

    #[test]
    fn normalization_ignores_origin_offset_so_original_offsets_do_not_move() {
        // A Unicode normalizer never reads `offset`; the mapping must be identical whatever the
        // origin the caller announces, and spans stay relative to `input` in every report.
        let input = "caf\u{e9} \u{ff21}";
        for origin in [0usize, 7, 4096] {
            let report =
                check_normalization(&NFKC, input, origin, Some(UnicodeForm::NFKC)).unwrap();
            assert_eq!(report.origin_offset, origin);
            assert_eq!(report.normalized, "caf\u{e9} A");
            let originals: Vec<_> = report.mappings.iter().map(|m| m.original.clone()).collect();
            // "caf" keep their single bytes, "é" is 3..5, the space 5..6, "Ａ" is 6..9.
            assert_eq!(originals, vec![0..1, 1..2, 2..3, 3..5, 5..6, 6..9]);
        }
    }

    #[test]
    fn already_normalized_text_is_the_identity() {
        let input = "plain ascii";
        let report = check_normalization(&NFC, input, 0, None).unwrap();
        assert_eq!(report.normalized, input);
        let originals: Vec<_> = report.mappings.iter().map(|m| m.original.clone()).collect();
        assert_eq!(originals, original_ranges(input));
    }

    #[test]
    fn form_mismatch_is_rejected() {
        // NFKC folds the fullwidth letter, NFD does not: asking the NFD output to satisfy NFKC
        // must fail rather than return a quietly wrong mapping.
        let err = check_normalization(&NFD, "\u{ff21}", 0, Some(UnicodeForm::NFKC)).unwrap_err();
        assert!(matches!(err, MappingError::UnexpectedOutput { .. }));
    }

    #[test]
    fn nmt_is_rejected_without_a_unicode_form() {
        // Nmt is not a Unicode normalization form, so auto-detection must not endorse its output.
        let err = check_normalization(&Nmt, "a\tb", 0, None).unwrap_err();
        assert!(matches!(err, MappingError::UnexpectedOutput { .. }));
    }
}
