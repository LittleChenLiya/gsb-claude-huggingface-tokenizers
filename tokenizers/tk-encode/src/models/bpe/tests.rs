//! Tests for [`PipelineBPE`], the only BPE in this crate.
//!
//! Everything here goes through [`PipelineBPE::from_vocab_and_merges`], which is the serde-free door
//! both readers come in by. The config-shaped `BPE` — its builder, its serde, its `read_file` — and
//! the tests that pin the two engines against each other moved to `tk-convert` with the type.
use super::*;
use crate::pipeline;
use crate::tokenizer::Result;

use crate::{pipeline::Model as PipelineModel, utils::byte_level::BYTES_CHAR_LOOKUP};

const HELLO_VOCAB: &[(&str, u32)] = &[
    ("h", 0),
    ("e", 1),
    ("l", 2),
    ("o", 3),
    ("he", 4),
    ("hel", 5),
    ("hell", 6),
    ("hello", 7),
];
const HELLO_MERGES: &[(&str, &str)] = &[("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")];

fn v(pairs: &[(&str, u32)]) -> Vocab {
    pairs.iter().map(|&(s, i)| (s.into(), i)).collect()
}

fn m(pairs: &[(&str, &str)]) -> Merges {
    pairs.iter().map(|&(a, b)| (a.into(), b.into())).collect()
}

/// The `hello` model: `h`/`e`/`l`/`o` plus the four merges that fold them into one id.
fn hello(options: BpeConfig) -> Result<PipelineBPE> {
    PipelineBPE::from_config(BpeConfig {
        vocab: v(HELLO_VOCAB),
        merges: m(HELLO_MERGES),
        ..options
    })
}

fn pipeline_ids(model: &PipelineBPE, sequence: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let mut scratch = model.init_scratch();
    pipeline::Model::tokenize_pipeline(model, sequence, &mut scratch, &mut out).unwrap();
    out.iter().map(|t| t.id()).collect()
}

#[test]
fn applies_merges() {
    let pipeline = hello(BpeConfig::default()).unwrap();
    for (input, want) in [
        ("hello", vec![7]),
        ("hell", vec![6]),
        ("helo", vec![5, 3]),
        ("oleh", vec![3, 2, 1, 0]),
    ] {
        assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
    }
}

#[test]
fn empty_input_yields_no_tokens() {
    let pipeline = hello(BpeConfig::default()).unwrap();
    assert!(pipeline_ids(&pipeline, "").is_empty());
}

// The scratch pool hands the SAME scratch to successive encodes. A bug leaking
// state between calls (an undrained merge queue, a stale word buffer) would
// corrupt every encode after the first. Drive several inputs — including
// repeats and an empty string — through one reused scratch and check each still
// matches what a fresh scratch produces. This is the invariant the pool relies on.
#[test]
fn reused_scratch_matches_fresh() {
    let model = hello(BpeConfig::default()).unwrap();
    let mut scratch = model.init_scratch();
    for input in ["hello", "hell", "helo", "oleh", "hello", "", "hxe"] {
        let mut out = Vec::new();
        pipeline::Model::tokenize_pipeline(&model, input, &mut scratch, &mut out).unwrap();
        let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
        assert_eq!(got, pipeline_ids(&model, input), "{input:?}");
    }
}

// A cache may forget a word, but it must never change one. Run every word twice
// through one scratch, the second time answered from the cache, against a model
// built with no cache at all.
#[test]
fn cached_ids_match_uncached() {
    let cached = hello(BpeConfig::default()).unwrap();
    let uncached = hello(BpeConfig {
        cache_capacity: 0,
        ..Default::default()
    })
    .unwrap();

    let mut scratch = cached.init_scratch();
    assert!(scratch.word_cache.is_some(), "nothing is being cached");
    for _ in 0..2 {
        for word in [
            "hello",
            "hell",
            "o",
            "hellohello",
            "hello-a-word-past-fifteen-bytes",
            "hxe",
        ] {
            let mut out = Vec::new();
            pipeline::Model::tokenize_pipeline(&cached, word, &mut scratch, &mut out).unwrap();
            let got: Vec<u32> = out.iter().map(|t| t.id()).collect();
            assert_eq!(got, pipeline_ids(&uncached, word), "{word:?}");
        }
    }
}

#[test]
fn unknown_char_without_unk_is_dropped() {
    let pipeline = hello(BpeConfig::default()).unwrap();
    // 'x' vanishes, making 'h' and 'e' adjacent, so the (h,e) merge applies.
    assert_eq!(pipeline_ids(&pipeline, "hxe"), vec![4]);
}

#[test]
fn unk_replaces_unknown_chars() {
    let mut vocab = v(HELLO_VOCAB);
    vocab.insert("<unk>".into(), 8);
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab,
        merges: m(HELLO_MERGES),
        ..BpeConfig {
            unk_token: Some("<unk>".into()),
            ..Default::default()
        }
    })
    .unwrap();
    for (input, want) in [
        ("hxe", vec![0, 8, 1]),
        ("xh", vec![8, 0]),
        ("hxxe", vec![0, 8, 8, 1]),
        ("xx", vec![8, 8]),
    ] {
        assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
    }
}

#[test]
fn fused_unk_collapses_runs() {
    let mut vocab = v(HELLO_VOCAB);
    vocab.insert("<unk>".into(), 8);
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab,
        merges: m(HELLO_MERGES),
        ..BpeConfig {
            unk_token: Some("<unk>".into()),
            fuse_unk: true,
            ..Default::default()
        }
    })
    .unwrap();
    for (input, want) in [
        ("hxxe", vec![0, 8, 1]),
        ("xxh", vec![8, 0]),
        ("xxxx", vec![8]),
        ("xhx", vec![8, 0, 8]),
    ] {
        assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
    }
}

fn byte_fallback_vocab() -> Vocab {
    let mut vocab = v(&[("h", 300), ("e", 301), ("<unk>", 400)]);
    vocab.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), u32::from(b))));
    vocab
}

#[test]
fn byte_fallback_encodes_missing_chars_as_byte_tokens() {
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab: byte_fallback_vocab(),
        merges: vec![],
        ..BpeConfig {
            byte_fallback: true,
            ..Default::default()
        }
    })
    .unwrap();
    // 'é' is not in the vocab: falls back to its UTF-8 bytes C3 A9
    assert_eq!(pipeline_ids(&pipeline, "hé"), vec![300, 0xC3, 0xA9]);
    assert_eq!(pipeline_ids(&pipeline, "🤗"), vec![0xF0, 0x9F, 0xA4, 0x97]);
    assert_eq!(pipeline_ids(&pipeline, "he"), vec![300, 301]);
}

#[test]
fn byte_fallback_wins_over_unk() {
    let pipeline = PipelineBPE::from_config(BpeConfig {
        vocab: byte_fallback_vocab(),
        merges: vec![],
        ..BpeConfig {
            byte_fallback: true,
            unk_token: Some("<unk>".into()),
            ..Default::default()
        }
    })
    .unwrap();
    assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
}

#[test]
fn ignore_merges_prefers_whole_word() {
    let pipeline = hello(BpeConfig {
        ignore_merges: true,
        ..Default::default()
    })
    .unwrap();
    // direct vocab hit bypasses the merge loop; a miss falls through to it
    assert_eq!(pipeline_ids(&pipeline, "hello"), vec![7]);
    assert_eq!(pipeline_ids(&pipeline, "helo"), vec![5, 3]);
}

#[test]
fn rejects_unsupported_configs() {
    // no merges: the merge-map derivation underflows on merges whose right token
    // is shorter than continuing_subword_prefix (pre-existing, unrelated)
    let build = |options: BpeConfig| {
        PipelineBPE::from_config(BpeConfig {
            vocab: v(HELLO_VOCAB),
            merges: vec![],
            ..options
        })
    };
    // dropout is a valid config this engine cannot run...
    assert!(
        build(BpeConfig {
            dropout: Some(0.5),
            ..Default::default()
        })
        .is_err()
    );
    // ...while a dropout outside 0..=1 was never a valid config at all. Both are rejected, but
    // the second is the range check that used to live in `BpeBuilder::build`.
    let out_of_range = build(BpeConfig {
        dropout: Some(1.5),
        ..Default::default()
    })
    .err()
    .unwrap();
    assert!(matches!(
        out_of_range.downcast_ref::<Error>(),
        Some(Error::InvalidDropout)
    ));
    // affixes are supported: `convert_affixed` decorates each character before the lookup
    assert!(
        build(BpeConfig {
            continuing_subword_prefix: Some("##".into()),
            ..Default::default()
        })
        .is_ok()
    );
    assert!(
        build(BpeConfig {
            end_of_word_suffix: Some("</w>".into()),
            ..Default::default()
        })
        .is_ok()
    );
    // no-op values must not be rejected: gpt2's tokenizer.json serializes
    // prefix/suffix as "" and the reference treats dropout 0.0 as disabled
    assert!(
        build(BpeConfig {
            continuing_subword_prefix: Some(String::new()),
            end_of_word_suffix: Some(String::new()),
            dropout: Some(0.0),
            ..Default::default()
        })
        .is_ok()
    );
}

#[test]
fn rejects_unk_token_missing_from_vocab() {
    assert!(
        hello(BpeConfig {
            unk_token: Some("<unk>".into()),
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn byte_fallback_with_missing_codes_errors() {
    // Incomplete <0xNN> coverage must be a build error, not a panic.
    assert!(
        hello(BpeConfig {
            byte_fallback: true,
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn rejects_merge_token_out_of_vocabulary() {
    // The merge-map derivation is now part of this constructor, so its errors are its own.
    let err = PipelineBPE::from_config(BpeConfig {
        vocab: v(HELLO_VOCAB),
        merges: m(&[("h", "z")]),
        ..BpeConfig::default()
    })
    .err()
    .unwrap();
    match err.downcast_ref::<Error>() {
        Some(Error::MergeTokenOutOfVocabulary(token)) => assert_eq!(token, "z"),
        other => panic!("expected MergeTokenOutOfVocabulary, got {other:?}"),
    }
}

fn projected(s: &str) -> String {
    s.bytes().map(|b| BYTES_CHAR_LOOKUP[b as usize]).collect()
}

/// A gpt2-shaped miniature: the 256 projected single-byte tokens
/// (id == byte value) plus `extra` tokens and merges, given in raw
/// space and projected here — like a real byte-level tokenizer.json,
/// whose vocab is stored in the projected alphabet.
fn byte_level_bpe(
    extra: &[(&str, u32)],
    merges: &[(&str, &str)],
    ignore_merges: bool,
) -> Result<PipelineBPE> {
    let mut vocab: Vocab = (0..=255u8)
        .map(|b| (BYTES_CHAR_LOOKUP[b as usize].to_string(), u32::from(b)))
        .collect();
    vocab.extend(extra.iter().map(|&(s, i)| (projected(s), i)));
    let merges: Merges = merges
        .iter()
        .map(|&(a, b)| (projected(a), projected(b)))
        .collect();
    PipelineBPE::from_config(BpeConfig {
        vocab,
        merges,
        ..BpeConfig {
            ignore_merges,
            byte_level: true,
            ..Default::default()
        }
    })
}

#[test]
fn byte_level_merges_raw_bytes() {
    let pipeline = byte_level_bpe(
        &[("he", 300), (" he", 301)],
        &[("h", "e"), (" ", "he")],
        false,
    )
    .unwrap();
    assert_eq!(pipeline_ids(&pipeline, " he"), vec![301]);
    // single bytes hit the un-projected single-byte tokens (id == byte value)
    assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
    assert_eq!(pipeline_ids(&pipeline, "\x00\x7f"), vec![0x00, 0x7f]);
    assert_eq!(
        pipeline_ids(&pipeline, "hé llo"),
        vec![
            u32::from(b'h'),
            0xC3,
            0xA9,
            u32::from(b' '),
            u32::from(b'l'),
            u32::from(b'l'),
            u32::from(b'o')
        ]
    );
}

#[test]
fn byte_level_ignore_merges_whole_word() {
    let pipeline = byte_level_bpe(&[(" hello", 300)], &[], true).unwrap();
    assert_eq!(pipeline_ids(&pipeline, " hello"), vec![300]);
    // not in vocab → falls through to single-byte atoms
    assert_eq!(
        pipeline_ids(&pipeline, "zz"),
        vec![u32::from(b'z'), u32::from(b'z')]
    );
}

#[test]
fn byte_level_id_to_token_spells_the_projected_alphabet() {
    let pipeline = byte_level_bpe(&[(" he", 300)], &[], true).unwrap();
    assert_eq!(pipeline.id_to_token(300).as_deref(), Some("Ġhe"));
    assert_eq!(pipeline.id_to_token(0xC3).as_deref(), Some("Ã"));
    assert_eq!(pipeline.id_to_token(301), None);
}

#[test]
fn byte_level_requires_full_byte_coverage() {
    // An ASCII-only vocab covers no control/high bytes: building the
    // byte-level pipeline must be a build error, not a panic.
    assert!(
        hello(BpeConfig {
            byte_level: true,
            ..Default::default()
        })
        .is_err()
    );
}

// --- Merge step tracking ----------------------------------------------------
//
// `from_config` enumerates ranks from the merge list, so every rank it builds is unique. The
// equal-rank fixtures below go through `from_merge_map`, which takes ranks as given, the only door
// by which a rank shared by several pairs can enter the tables.

use crate::vocab::bucket_vocab_store::BucketVocabStore;

/// Builds a model straight from an external-id merge map, so ranks can repeat.
fn model_from_merge_map(
    pairs: &[(&str, u32)],
    merges: Vec<((u32, u32), (u32, u32))>,
) -> Result<PipelineBPE> {
    let vocab: Vocab = pairs.iter().map(|&(s, i)| (s.into(), i)).collect();
    let store = BucketVocabStore::build(
        vocab
            .into_iter()
            .map(|(k, v)| (k.into_bytes(), v))
            .collect(),
    );
    PipelineBPE::from_merge_map(store, merges.into_iter().collect(), BpeConfig::default())
}

fn step(rank: u32, left: u32, right: u32, merged: u32) -> MergeStep {
    MergeStep {
        rank,
        left_id: left,
        right_id: right,
        merged_id: merged,
    }
}

/// Tracking a word whose four merges have distinct ranks: one step per merge, in rank order, and
/// the final ids are the one token everything folds into.
#[test]
fn merge_trace_records_steps_and_tokens() {
    let model = hello(BpeConfig::default()).unwrap();
    let (ids, steps) = model.tokenize_with_merge_trace("hello");
    assert_eq!(ids, vec![7]);
    assert_eq!(
        steps,
        vec![
            step(0, 0, 1, 4), // h + e -> he
            step(1, 4, 2, 5), // he + l -> hel
            step(2, 5, 2, 6), // hel + l -> hell
            step(3, 6, 3, 7), // hell + o -> hello
        ]
    );

    // A word that does not merge the whole way: steps stop where the merges stop, and the ids are
    // the remaining pieces.
    let (ids, steps) = model.tokenize_with_merge_trace("helo");
    assert_eq!(ids, vec![5, 3]);
    assert_eq!(steps, vec![step(0, 0, 1, 4), step(1, 4, 2, 5)]);

    // Nothing to merge: no steps, atoms only.
    let (ids, steps) = model.tokenize_with_merge_trace("oleh");
    assert_eq!(ids, vec![3, 2, 1, 0]);
    assert!(steps.is_empty());

    // The empty word stays a fast no-op with no steps.
    let (ids, steps) = model.tokenize_with_merge_trace("");
    assert!(ids.is_empty());
    assert!(steps.is_empty());
}

/// Tracking must report exactly what untracked encoding outputs, for every word -- including the
/// ones the regular path answers from the fold or the word cache without merging at all.
#[test]
fn merge_trace_tokens_match_regular_encoding() {
    let model = hello(BpeConfig::default()).unwrap();
    for word in [
        "hello",
        "hell",
        "helo",
        "oleh",
        "hxe",
        "hellohello",
        "o",
        "",
    ] {
        let (ids, _steps) = model.tokenize_with_merge_trace(word);
        assert_eq!(ids, pipeline_ids(&model, word), "{word:?}");
    }
}

/// The tie fixture: two disjoint merges assigned the *same* rank. BPE prescribes that a rank tie
/// resolves to the leftmost pair, so the word `abcd` must merge `(a,b)` before `(c,d)`, and doing
/// it twice is identical. The steps and final tokens are the whole acceptance point.
#[test]
fn merge_trace_same_rank_picks_leftmost_deterministically() {
    let model = model_from_merge_map(
        &[("a", 0), ("b", 1), ("c", 2), ("d", 3), ("ab", 4), ("cd", 5)],
        // both pairs deliberately share rank 0
        vec![((0, 1), (0, 4)), ((2, 3), (0, 5))],
    )
    .unwrap();

    let expected_steps = vec![step(0, 0, 1, 4), step(0, 2, 3, 5)];
    for _ in 0..3 {
        let (ids, steps) = model.tokenize_with_merge_trace("abcd");
        assert_eq!(ids, vec![4, 5]);
        assert_eq!(steps, expected_steps);
    }
    // only the left pair present
    let (ids, steps) = model.tokenize_with_merge_trace("ab");
    assert_eq!(ids, vec![4]);
    assert_eq!(steps, vec![step(0, 0, 1, 4)]);
}

/// Adjacent pairs that share a rank overlap on a symbol: `(a,b)` and `(b,c)` cannot both fire.
/// The left one wins, and its product `ab` is then stuck next to `c` (no rank-0 `(ab,c)` pair).
/// This is the overlap case where "smaller packed value" and "leftmost" used to disagree; the
/// answer must be the leftmost one.
#[test]
fn merge_trace_same_rank_overlapping_pairs_keeps_left() {
    let model = model_from_merge_map(
        &[("a", 0), ("b", 1), ("c", 2), ("ab", 3), ("bc", 4)],
        vec![((0, 1), (0, 3)), ((1, 2), (0, 4))],
    )
    .unwrap();
    let (ids, steps) = model.tokenize_with_merge_trace("abc");
    assert_eq!(ids, vec![3, 2]);
    assert_eq!(steps, vec![step(0, 0, 1, 3)]);
}

/// Short words go to multipass and long words to the hot/cold queue, but merge order -- including
/// the leftmost tie-break -- and the final tokens must be identical across the two engines.
#[test]
fn merge_trace_engines_agree_on_same_rank() {
    // Two disjoint rank-0 merges plus a rank-1 chain merge on the left. The padding runs of a
    // symbol with no merge push the word past the queue gate without adding steps.
    let model = model_from_merge_map(
        &[
            ("a", 0),
            ("b", 1),
            ("c", 2),
            ("d", 3),
            ("z", 6),
            ("ab", 4),
            ("cd", 5),
            ("zab", 7),
        ],
        vec![
            ((0, 1), (0, 4)), // (a,b), rank 0
            ((2, 3), (0, 5)), // (c,d), rank 0, ties with (a,b)
            ((6, 4), (1, 7)), // (z,ab), rank 1: only applies when z precedes ab
        ],
    )
    .unwrap();

    let (short_ids, short_steps) = model.tokenize_with_merge_trace("abcd");
    assert_eq!(short_ids, vec![4, 5]);
    assert_eq!(short_steps, vec![step(0, 0, 1, 4), step(0, 2, 3, 5)]);

    // 25 content bytes: past the 24-byte ASCII gate (`sequence.len() > GATE_ASCII`), so the queue
    // engine, rather than multipass, runs it.
    let long_word = format!("{}{}", "z".repeat(21), "abcd");
    let (long_ids, long_steps) = model.tokenize_with_merge_trace(&long_word);
    // the first abcd starts one past the z run: z + ab collapses to zab... the remaining z's and
    // cd are untouched
    let mut expected_ids = vec![6; 20];
    expected_ids.push(7); // zab
    expected_ids.push(5); // cd
    assert_eq!(long_ids, expected_ids);
    assert_eq!(
        long_steps,
        vec![
            step(0, 0, 1, 4), // (a,b) before (c,d) on the rank tie
            step(0, 2, 3, 5),
            step(1, 6, 4, 7), // (z,ab) afterwards, by higher rank
        ]
    );
    // cross-engine agreement on the region both contain
    assert_eq!(&long_steps[..2], &short_steps[..]);

    // And the tracked result is still exactly what regular encoding produces.
    let mut scratch = model.init_scratch();
    let mut out = Vec::new();
    pipeline::Model::tokenize_pipeline(&model, &long_word, &mut scratch, &mut out).unwrap();
    assert_eq!(long_ids, out.iter().map(|t| t.id()).collect::<Vec<_>>());
}
