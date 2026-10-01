//! Integration-level acceptance for the normalization char-span mapping check.
//!
//! Where the unit tests in `normalizers/mapping.rs` pin the spans against the reference
//! normalization, these tests compare the mapped offsets with the original encode offsets for the
//! three cases the task calls out: accented characters, full-width characters and empty input.

use std::ops::Range;

use tk_encode::normalizers::{
    CharMapping, NFC, NFD, NFKC, NFKD, UnicodeForm, check_normalization, identity_mappings,
};
use tk_encode::tokenizer::pipeline::Normalizer;

/// Original byte spans of every character: the offsets encoding reports before normalization.
fn original_spans(input: &str) -> Vec<Range<usize>> {
    identity_mappings(input)
        .into_iter()
        .map(|m| m.original)
        .collect()
}

/// The single smallest span all mappings for one normalized character agree on.
fn mapped_spans(report: &tk_encode::normalizers::MappingReport) -> Vec<Range<usize>> {
    report.mappings.iter().map(|m| m.original.clone()).collect()
}

/// Every original byte must be reachable by mapping some normalized span back.
fn mapped_back_covers(input: &str, form: UnicodeForm) {
    let normalizer: &dyn Normalizer = match form {
        UnicodeForm::NFD => &NFD,
        UnicodeForm::NFC => &NFC,
        UnicodeForm::NFKD => &NFKD,
        UnicodeForm::NFKC => &NFKC,
    };
    let report = check_normalization(normalizer, input, 0, Some(form)).unwrap();
    let whole_normalized = 0..report.normalized.len();
    assert_eq!(
        report.map_range_back(whole_normalized),
        Some(0..input.len()),
        "mapping the whole normalized text back must cover the whole input"
    );
}

#[test]
fn accented_characters_nfd_and_nfc_round_trip_to_original_offsets() {
    // NFD: one precomposed 2-byte "é" expands to "e" + U+0301. Both normalized chars map back onto
    // the original 3..5 span; a token cut on either side still resolves to the full original "é".
    let input = "caf\u{e9}";
    let report = check_normalization(&NFD, input, 0, Some(UnicodeForm::NFD)).unwrap();
    assert_eq!(report.normalized, "cafe\u{301}");
    assert_eq!(
        mapped_spans(&report),
        vec![0..1, 1..2, 2..3, 3..5, 3..5],
        "the combining acute inherits the span of its base; nothing points at an empty range"
    );
    assert_eq!(original_spans(input), vec![0..1, 1..2, 2..3, 3..5]);
    // The trailing "e" + combining mark together map back to the original "é" bytes only.
    assert_eq!(report.map_range_back(3..6), Some(3..5));

    // NFC: "e" + U+0301 (1 + 2 bytes) composes into one 2-byte "é" spanning both originals.
    let input = "e\u{301}";
    let report = check_normalization(&NFC, input, 0, Some(UnicodeForm::NFC)).unwrap();
    assert_eq!(report.normalized, "\u{e9}");
    assert_eq!(mapped_spans(&report), vec![0..3]);
    assert_eq!(original_spans(input), vec![0..1, 1..3]);
    assert_eq!(report.map_range_back(0..2), Some(0..3));

    mapped_back_covers("caf\u{e9}", UnicodeForm::NFD);
    mapped_back_covers("e\u{301}", UnicodeForm::NFC);
}

#[test]
fn fullwidth_characters_fold_onto_their_original_three_byte_spans() {
    // A full-width "Ａ" (U+FF21) is 3 bytes; NFKC folds it to 1-byte "A" without moving anything.
    let input = "\u{ff21}";
    let report = check_normalization(&NFKC, input, 0, Some(UnicodeForm::NFKC)).unwrap();
    assert_eq!(report.normalized, "A");
    assert_eq!(mapped_spans(&report), vec![0..3]);
    assert_eq!(original_spans(input), vec![0..3]);

    // In a sentence, neighbors keep their bytes and the full-width digit keeps its own 3 bytes.
    let input = "x\u{ff10}z";
    let report = check_normalization(&NFKC, input, 0, Some(UnicodeForm::NFKC)).unwrap();
    assert_eq!(report.normalized, "x0z");
    assert_eq!(mapped_spans(&report), vec![0..1, 1..4, 4..5]);
    assert_eq!(original_spans(input), vec![0..1, 1..4, 4..5]);
    // Point queries land on the same boundary the original encoding used.
    assert_eq!(report.map_range_back(0..0), Some(0..0));
    assert_eq!(report.map_range_back(1..1), Some(1..1));
    assert_eq!(report.map_range_back(2..3), Some(4..5));
}

#[test]
fn empty_input_is_deterministic_for_every_form() {
    for form in UnicodeForm::all() {
        let normalizer: &dyn Normalizer = match form {
            UnicodeForm::NFD => &NFD,
            UnicodeForm::NFC => &NFC,
            UnicodeForm::NFKD => &NFKD,
            UnicodeForm::NFKC => &NFKC,
        };
        let report = check_normalization(normalizer, "", 0, Some(form)).unwrap();
        assert_eq!(report.normalized, "");
        assert!(report.mappings.is_empty());
        assert_eq!(report.map_range_back(0..0), Some(0..0));
        assert!(identity_mappings("").is_empty());
        let _: Vec<CharMapping> = Vec::new();
    }
}

#[test]
fn a_nonzero_origin_offset_does_not_change_the_mapped_offsets() {
    let input = "caf\u{e9} \u{ff21}";
    let at_zero = check_normalization(&NFKC, input, 0, Some(UnicodeForm::NFKC)).unwrap();
    for origin in [1usize, 7, 65_535] {
        let report = check_normalization(&NFKC, input, origin, Some(UnicodeForm::NFKC)).unwrap();
        assert_eq!(report.normalized, at_zero.normalized);
        assert_eq!(mapped_spans(&report), mapped_spans(&at_zero));
        assert_eq!(report.origin_offset, origin);
        // Every span is relative to `input`, so none starts as high as the announced origin.
        assert!(
            report
                .mappings
                .iter()
                .all(|m| m.original.start < input.len() && m.original.end <= input.len())
        );
    }
    assert_eq!(
        mapped_spans(&at_zero),
        vec![0..1, 1..2, 2..3, 3..5, 5..6, 6..9]
    );
    assert_eq!(
        original_spans(input),
        vec![0..1, 1..2, 2..3, 3..5, 5..6, 6..9]
    );
}

#[test]
fn mappings_tile_the_normalized_text_per_character() {
    for (form, input) in [
        (UnicodeForm::NFD, "caf\u{e9}"),
        (UnicodeForm::NFC, "e\u{301}"),
        (UnicodeForm::NFKC, "\u{ff21}\u{ff10}"),
    ] {
        let normalizer: &dyn Normalizer = match form {
            UnicodeForm::NFD => &NFD,
            UnicodeForm::NFC => &NFC,
            UnicodeForm::NFKD => &NFKD,
            UnicodeForm::NFKC => &NFKC,
        };
        let report = check_normalization(normalizer, input, 0, Some(form)).unwrap();
        let mut cursor = 0;
        for m in &report.mappings {
            assert_eq!(m.normalized.start, cursor);
            cursor = m.normalized.end;
        }
        assert_eq!(cursor, report.normalized.len());
    }
}
