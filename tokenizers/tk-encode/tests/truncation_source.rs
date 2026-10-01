//! Integration tests for the truncation source marker.
//!
//! These build a tokenizer from controllable parts (a tiny BPE vocabulary, a BERT-style template
//! that adds specials, and an added special token) and check, for single sentences, sentence pairs
//! and special-token fixtures, that the encoding result carries the right marker and the right token
//! count. The failure paths (`only_second` without a pair, `max_length == 0`) are checked too.

use std::collections::BTreeMap;

use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
use tk_encode::tokenizer::pipeline::{
    EncodeOptions, Override, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
    PipelinePreTokenizer, PipelineTokenizer, Template,
};
use tk_encode::tokenizer::{TruncationDirection, TruncationParams, TruncationSource, TruncationStrategy};
use tk_encode::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary};

/// The hello BPE: `h e l l o` merge down to `hello`. Vocab ids 0..=7.
fn hello_bpe() -> PipelineBPE {
    let vocab: Vocab = [
        ("h", 0u32),
        ("e", 1),
        ("l", 2),
        ("o", 3),
        ("he", 4),
        ("hel", 5),
        ("hell", 6),
        ("hello", 7),
    ]
    .into_iter()
    .map(|(s, i)| (s.to_string(), i))
    .collect();
    let merges: Merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
        ("hel".to_string(), "l".to_string()),
        ("hell".to_string(), "o".to_string()),
    ];
    PipelineBPE::from_config(BpeConfig {
        vocab,
        merges,
        ..BpeConfig::default()
    })
    .unwrap()
}

/// `[CLS] $A [SEP]` and `[CLS] $A [SEP] $B:1 [SEP]:1` — the specials count: 2 for a single, 3 for a
/// pair.
fn bert_post_processor() -> PipelinePostProcessor {
    const CLS: u32 = 101;
    const SEP: u32 = 102;
    PipelinePostProcessor {
        single: Template {
            prefix: Box::new([(CLS.into(), 0)]),
            suffix: Box::new([(SEP.into(), 0)]),
            ..Template::default()
        },
        pair: Template {
            prefix: Box::new([(CLS.into(), 0)]),
            infix: Box::new([(SEP.into(), 0)]),
            suffix: Box::new([(SEP.into(), 1)]),
            b_type_id: Some(1),
            ..Template::default()
        },
    }
}

fn truncation(max_length: usize, strategy: TruncationStrategy) -> Option<TruncationParams> {
    Some(TruncationParams {
        max_length,
        strategy,
        ..TruncationParams::default()
    })
}

/// A plain tokenizer: no added specials, no post-processor specials, optional truncation.
fn plain_tokenizer(truncation: Option<TruncationParams>) -> PipelineTokenizer {
    PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        PipelinePreTokenizer::None,
        PipelineModel::BPE(hello_bpe()),
        PipelinePostProcessor::default(),
        None,
        BTreeMap::new(),
        None,
        truncation,
    )
}

/// A BERT-style tokenizer: the template adds specials, optional truncation.
fn bert_tokenizer(truncation: Option<TruncationParams>) -> PipelineTokenizer {
    PipelineTokenizer::from_parts(
        AddedVocabulary::new(),
        Vec::new(),
        PipelinePreTokenizer::None,
        PipelineModel::BPE(hello_bpe()),
        bert_post_processor(),
        None,
        BTreeMap::new(),
        None,
        truncation,
    )
}

/// A tokenizer with one added special token `<s>` (id 8), no template specials, optional
/// truncation.
fn special_tokenizer(truncation: Option<TruncationParams>) -> PipelineTokenizer {
    let mut added = AddedVocabulary::new();
    added
        .add_special_tokens(
            [AddedToken::from("<s>", true)],
            8,
            |_| None,
            None::<&PipelineNormalizer>,
        )
        .unwrap();
    PipelineTokenizer::from_parts(
        added,
        Vec::new(),
        PipelinePreTokenizer::None,
        PipelineModel::BPE(hello_bpe()),
        PipelinePostProcessor::default(),
        None,
        BTreeMap::new(),
        None,
        truncation,
    )
}

fn ids(encoding: &tk_encode::tokenizer::pipeline::Encoding) -> Vec<u32> {
    encoding.ids().iter().copied().map(u32::from).collect()
}

// --- Single sentence -------------------------------------------------------

#[test]
fn single_sentence_fits() {
    let tokenizer = plain_tokenizer(truncation(20, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("hello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [7]);
    assert_eq!(encoding.len(), 1);
    assert_eq!(encoding.truncation_source(), TruncationSource::None);
}

#[test]
fn single_sentence_truncated() {
    // "hhhhhhello" is 6 tokens; max 3 cuts to 3.
    let tokenizer = plain_tokenizer(truncation(3, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("hhhhhhello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [0, 0, 0]);
    assert_eq!(encoding.len(), 3);
    assert_eq!(encoding.truncation_source(), TruncationSource::First);
}

#[test]
fn single_sentence_no_truncation_config() {
    let tokenizer = plain_tokenizer(None);
    let encoding = tokenizer
        .encode("hhhhhhello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [0, 0, 0, 0, 0, 7]);
    assert_eq!(encoding.len(), 6);
    assert_eq!(encoding.truncation_source(), TruncationSource::None);
}

// --- Sentence pair ---------------------------------------------------------

#[test]
fn pair_fits() {
    let tokenizer = bert_tokenizer(truncation(20, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode(("hello", "hello"), &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    // [CLS] hello [SEP] hello [SEP]
    assert_eq!(ids(&encoding), [101, 7, 102, 7, 102]);
    assert_eq!(encoding.len(), 5);
    assert_eq!(encoding.truncation_source(), TruncationSource::None);
}

#[test]
fn pair_truncated_first() {
    // 6 + 2 = 8 sequence tokens, 3 specials, max 8 => 5 kept: 4 from A, 1 from B.
    let tokenizer = bert_tokenizer(truncation(8, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode(("hhhhhhello", "he"), &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [101, 0, 0, 0, 0, 102, 4, 102]);
    assert_eq!(encoding.len(), 8);
    assert_eq!(encoding.truncation_source(), TruncationSource::First);
}

#[test]
fn pair_truncated_second() {
    // 2 + 6 = 8 sequence tokens, 3 specials, max 8 => 5 kept: 1 from A, 4 from B.
    let tokenizer = bert_tokenizer(truncation(8, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode(("he", "hhhhhhello"), &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [101, 4, 102, 0, 0, 0, 0, 102]);
    assert_eq!(encoding.len(), 8);
    assert_eq!(encoding.truncation_source(), TruncationSource::Second);
}

#[test]
fn pair_truncated_both() {
    // 6 + 6 = 12 sequence tokens, 3 specials, max 7 => 4 kept: 2 from A, 2 from B.
    let tokenizer = bert_tokenizer(truncation(7, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode(("hhhhhhello", "hhhhhhello"), &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [101, 0, 0, 102, 0, 0, 102]);
    assert_eq!(encoding.len(), 7);
    assert_eq!(encoding.truncation_source(), TruncationSource::Both);
}

// --- Special token fixture -------------------------------------------------

#[test]
fn special_token_fits() {
    // `<s>` is carved to id 8; `<s>hello` is 2 tokens.
    let tokenizer = special_tokenizer(truncation(10, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("<s>hello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [8, 7]);
    assert_eq!(encoding.len(), 2);
    assert_eq!(encoding.truncation_source(), TruncationSource::None);
}

#[test]
fn special_token_truncated() {
    // `<s>hello` is 2 tokens; max 1 cuts to 1, keeping the first token (`<s>`).
    let tokenizer = special_tokenizer(truncation(1, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("<s>hello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [8]);
    assert_eq!(encoding.len(), 1);
    assert_eq!(encoding.truncation_source(), TruncationSource::First);
}

#[test]
fn special_token_counts_toward_max_length() {
    // `<s>hello` is 2 tokens; max 2 with no specials from the template fits exactly.
    let tokenizer = special_tokenizer(truncation(2, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("<s>hello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [8, 7]);
    assert_eq!(encoding.len(), 2);
    assert_eq!(encoding.truncation_source(), TruncationSource::None);

    // max 1 cuts: the special token still counts, so only one token survives.
    let tokenizer = special_tokenizer(truncation(1, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("<s>hello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [8]);
    assert_eq!(encoding.len(), 1);
    assert_eq!(encoding.truncation_source(), TruncationSource::First);
}

// --- Failure / edge paths ---------------------------------------------------

#[test]
fn only_second_without_pair_is_an_error() {
    let tokenizer = plain_tokenizer(truncation(5, TruncationStrategy::OnlySecond));
    let result = tokenizer
        .encode("hhhhhhello", &EncodeOptions::default())
        .wait();

    assert!(result.is_err());
}

#[test]
fn only_first_too_short_is_an_error() {
    // `he` is 1 token, the pair is 6: max 3 cannot be reached by cutting the first sequence alone
    // (1 <= 7 - 3 = 4 removed), so `only_first` errors rather than touching the pair.
    let tokenizer = plain_tokenizer(truncation(3, TruncationStrategy::OnlyFirst));
    let result = tokenizer
        .encode(("he", "hhhhhhello"), &EncodeOptions::default())
        .wait();

    assert!(result.is_err());
}

#[test]
fn max_length_zero_empties_the_sequence() {
    let tokenizer = plain_tokenizer(truncation(0, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode("hhhhhhello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert!(ids(&encoding).is_empty());
    assert_eq!(encoding.len(), 0);
    assert_eq!(encoding.truncation_source(), TruncationSource::First);
}

#[test]
fn max_length_zero_pair_empties_both() {
    let tokenizer = plain_tokenizer(truncation(0, TruncationStrategy::LongestFirst));
    let encoding = tokenizer
        .encode(("hhhhhhello", "hello"), &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    assert!(ids(&encoding).is_empty());
    assert_eq!(encoding.len(), 0);
    assert_eq!(encoding.truncation_source(), TruncationSource::Both);
}

#[test]
fn truncation_direction_left_keeps_the_tail() {
    let tokenizer = plain_tokenizer(Some(TruncationParams {
        max_length: 3,
        direction: TruncationDirection::Left,
        ..TruncationParams::default()
    }));
    let encoding = tokenizer
        .encode("hhhhhhello", &EncodeOptions::default())
        .wait()
        .unwrap()
        .remove(0);

    // Left truncation keeps the last 3 tokens: h, h, hello.
    assert_eq!(ids(&encoding), [0, 0, 7]);
    assert_eq!(encoding.len(), 3);
    assert_eq!(encoding.truncation_source(), TruncationSource::First);
}

#[test]
fn override_off_turns_truncation_off() {
    let tokenizer = plain_tokenizer(truncation(3, TruncationStrategy::LongestFirst));
    let options = EncodeOptions {
        truncation: Override::Off,
        ..EncodeOptions::default()
    };
    let encoding = tokenizer
        .encode("hhhhhhello", &options)
        .wait()
        .unwrap()
        .remove(0);

    assert_eq!(ids(&encoding), [0, 0, 0, 0, 0, 7]);
    assert_eq!(encoding.len(), 6);
    assert_eq!(encoding.truncation_source(), TruncationSource::None);
}
