//! Runnable acceptance demo for the normalization char-span mapping check.
//!
//! It shows, on the console, the behavioral difference the mapping pins:
//!
//! ```bash
//! cargo run -p tk-encode --features normalizers \
//!     --example normalization_mapping_check
//! ```
//!
//! For accented characters, full-width characters and the empty input it prints, side by side:
//!
//! * the byte offsets the original text encodes to (`identity_mappings`);
//! * the byte offsets the normalized text encodes to;
//! * where every normalized character maps back in the original text (`check_normalization`).
//!
//! The last block runs the same check with a non-zero origin `offset` and shows the spans do not
//! move: the pipeline offset is an origin the check echoes, never a shift applied to the spans.

use std::ops::Range;

use tk_encode::normalizers::{
    NFC, NFD, NFKC, NFKD, UnicodeForm, check_normalization, identity_mappings,
};
use tk_encode::tokenizer::pipeline::Normalizer;

/// Render `s` with one line per char: char, byte span, and (debug) escape.
fn char_lines(s: &str) -> Vec<String> {
    identity_mappings(s)
        .into_iter()
        .map(|m| format!("    {:>2?} bytes {:?}", m.character.to_string(), m.original))
        .collect()
}

fn mapped_lines(form: UnicodeForm, input: &str) -> Vec<String> {
    // Zero origin here: spans are relative to `input` itself.
    let normalizer: &dyn Normalizer = match form {
        UnicodeForm::NFD => &NFD,
        UnicodeForm::NFC => &NFC,
        UnicodeForm::NFKD => &NFKD,
        UnicodeForm::NFKC => &NFKC,
    };
    let report = check_normalization(normalizer, input, 0, Some(form)).unwrap();
    report
        .mappings
        .iter()
        .map(|m| {
            format!(
                "    {:>2?} normalized {:?} -> original {:?}",
                m.character.to_string(),
                m.normalized,
                m.original
            )
        })
        .collect()
}

fn show(title: &str, form: UnicodeForm, input: &str) {
    println!("=== {title}");
    println!("input:       {input:?} ({} bytes)", input.len());
    println!("  original encode offsets (one line per input char):");
    for line in char_lines(input) {
        println!("{line}");
    }
    let normalized = match form {
        UnicodeForm::NFD => NFD.normalize(input, 0).unwrap(),
        UnicodeForm::NFC => NFC.normalize(input, 0).unwrap(),
        UnicodeForm::NFKD => NFKD.normalize(input, 0).unwrap(),
        UnicodeForm::NFKC => NFKC.normalize(input, 0).unwrap(),
    };
    println!(
        "normalized:  {:?} ({} bytes), form {form:?}",
        normalized.as_ref(),
        normalized.len()
    );
    println!("  normalized offsets and the back-mapping onto the original:");
    for line in mapped_lines(form, input) {
        println!("{line}");
    }
    println!();
}

fn show_offset_invariance(input: &str) {
    println!("=== origin `offset` must not move the spans");
    println!("input: {input:?} ({} bytes)", input.len());
    let mut previous: Option<Vec<Range<usize>>> = None;
    for origin in [0usize, 7, 4096] {
        let report = check_normalization(&NFKC, input, origin, Some(UnicodeForm::NFKC)).unwrap();
        let spans: Vec<Range<usize>> = report.mappings.iter().map(|m| m.original.clone()).collect();
        println!(
            "  offset={origin:>4} -> normalized {:?}, mapped spans {spans:?}",
            report.normalized
        );
        if let Some(prev) = &previous {
            assert_eq!(prev, &spans, "spans shifted when the origin changed");
        }
        previous = Some(spans);
    }
    println!("  identical spans at every origin: original offsets unchanged");
    println!();
}

fn main() {
    // Accent, precomposed: NFD fans one char out into a base + a combining mark.
    show(
        "accent (NFD): precomposed e-acute",
        UnicodeForm::NFD,
        "caf\u{e9}",
    );
    // Accent, decomposed: NFC fuses a base + a combining mark back into one char.
    show(
        "accent (NFC): combining acute",
        UnicodeForm::NFC,
        "e\u{301}",
    );
    // Full-width ASCII: NFKC folds a 3-byte ideographic-width letter onto one ASCII byte.
    show(
        "fullwidth (NFKC): full-width capital A",
        UnicodeForm::NFKC,
        "\u{ff21}",
    );
    show(
        "fullwidth (NFKC): full-width digit in a sentence",
        UnicodeForm::NFKC,
        "x\u{ff10}z",
    );
    // Empty input: deterministic, no chars to map, zero-length point round-trips.
    show("empty input (NFD)", UnicodeForm::NFD, "");
    show("empty input (NFKC)", UnicodeForm::NFKC, "");

    show_offset_invariance("caf\u{e9} \u{ff21}");

    println!("all mapping checks passed");
}
