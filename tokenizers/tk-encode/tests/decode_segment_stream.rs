//! Integration tests for the streaming decode segment boundary API.
//!
//! These tests feed multi-byte characters, whitespace, and special tokens one at a time
//! through [`DecodeSegmentStream`], verify each segment boundary, and confirm the
//! concatenated result equals the batch decode.

use tk_encode::decoders::byte_fallback::ByteFallback;
use tk_encode::decoders::byte_level::ByteLevelDecoder;
use tk_encode::decoders::DecodeSegmentStream;
use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;

/// Convert a byte slice to its byte-level token string (each byte -> its printable char).
fn bytes_to_token(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| BYTES_CHAR_LOOKUP[b as usize])
        .collect()
}

/// Feed tokens one at a time, recording each segment and whether it was pending.
fn feed_tokens<D: tk_encode::Decoder>(
    stream: &mut DecodeSegmentStream<D>,
    tokens: &[String],
) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    for token in tokens {
        let seg = stream.step(token.clone()).unwrap();
        // A token is "pending" when it did not complete a segment (buffered internally).
        let is_pending = seg.is_none();
        out.push((seg.unwrap_or_default(), is_pending));
    }
    out
}

/// The concatenation of all segments equals the batch decode result.
fn assert_stream_matches_batch<D: tk_encode::Decoder + Clone>(
    decoder: &D,
    tokens: &[String],
) {
    let batch = decoder.decode(tokens.to_vec()).unwrap();
    let mut stream = DecodeSegmentStream::new(decoder.clone());
    let mut streamed = String::new();
    for token in tokens {
        if let Some(seg) = stream.step(token.clone()).unwrap() {
            streamed.push_str(&seg);
        }
    }
    if let Some(seg) = stream.flush().unwrap() {
        streamed.push_str(&seg);
    }
    assert_eq!(streamed, batch, "streamed segments must equal batch decode");
}

#[test]
fn multi_byte_chars_whitespace_and_special_tokens_token_by_token() {
    let decoder = ByteLevelDecoder::new();
    let mut stream = DecodeSegmentStream::new(decoder);

    // Build a token stream that mixes:
    // - ASCII words
    // - whitespace (Ġ = space in byte-level)
    // - multi-byte UTF-8 characters split into individual bytes
    // - special tokens ([PAD], [SEP])
    //
    // The target text is: "Hello 世界 [PAD] friend"
    let mut tokens: Vec<String> = Vec::new();

    // "Hello"
    tokens.push("Hello".to_string());
    // " " (Ġ)
    tokens.push("Ġ".to_string());
    // "世界" = E4 B8 96 E7 95 8C, split into individual bytes
    for &b in &[0xE4, 0xB8, 0x96, 0xE7, 0x95, 0x8C] {
        tokens.push(bytes_to_token(&[b]));
    }
    // " " (Ġ)
    tokens.push("Ġ".to_string());
    // "[PAD]"
    tokens.push("[PAD]".to_string());
    // " " (Ġ)
    tokens.push("Ġ".to_string());
    // "friend"
    tokens.push("friend".to_string());

    // Feed one token at a time and record segments.
    let segments = feed_tokens(&mut stream, &tokens);

    // Verify the segments:
    // - "Hello" -> complete
    // - " " -> complete
    // - E4 -> pending (half of 世)
    // - B8 -> pending (still half of 世)
    // - 96 -> complete "世"
    // - E7 -> pending (half of 界)
    // - 95 -> pending (still half of 界)
    // - 8C -> complete "界"
    // - " " -> complete
    // - "[PAD]" -> complete
    // - " " -> complete
    // - "friend" -> complete
    let expected: Vec<(String, bool)> = vec![
        ("Hello".to_string(), false),
        (" ".to_string(), false),
        (String::new(), true),  // E4: pending
        (String::new(), true),  // B8: pending
        ("世".to_string(), false),
        (String::new(), true),  // E7: pending
        (String::new(), true),  // 95: pending
        ("界".to_string(), false),
        (" ".to_string(), false),
        ("[PAD]".to_string(), false),
        (" ".to_string(), false),
        ("friend".to_string(), false),
    ];

    assert_eq!(segments, expected);

    // Flush and verify the final text.
    assert!(!stream.has_pending());
    assert_eq!(stream.flush().unwrap(), None);

    // The concatenation of all segments equals the batch decode.
    assert_stream_matches_batch(&decoder, &tokens);
}

#[test]
fn batch_decode_equivalence_for_random_mixed_token_streams() {
    let decoder = ByteLevelDecoder::new();

    // Generate several token streams with mixed content and verify batch equivalence.
    let test_cases: Vec<Vec<String>> = vec![
        // All ASCII
        vec!["Hello".to_string(), "Ġ".to_string(), "world".to_string()],
        // Multi-byte chars split into bytes
        {
            let mut t = vec!["Ġ".to_string()];
            for &b in "你好".as_bytes() {
                t.push(bytes_to_token(&[b]));
            }
            t
        },
        // Special tokens mixed with text
        vec![
            "[CLS]".to_string(),
            "Ġ".to_string(),
            "test".to_string(),
            "Ġ".to_string(),
            "[SEP]".to_string(),
        ],
        // Multi-byte chars split into bytes, followed by ASCII
        {
            let mut t = Vec::new();
            for &b in "日本".as_bytes() {
                t.push(bytes_to_token(&[b]));
            }
            t.push("Ġ".to_string());
            t.push("done".to_string());
            t
        },
        // Empty tokens and whitespace
        vec!["".to_string(), "Ġ".to_string(), "".to_string()],
    ];

    for tokens in &test_cases {
        assert_stream_matches_batch(&decoder, tokens);
    }
}

#[test]
fn byte_fallback_streaming_segments_match_batch() {
    let decoder = ByteFallback::new();

    // "叫" = E5 8F AB, split across three byte tokens, followed by ASCII.
    let tokens: Vec<String> = vec![
        "<0xE5>".to_string(),
        "<0x8F>".to_string(),
        "<0xAB>".to_string(),
        "a".to_string(),
        "<0xE6>".to_string(),
        "<0x9C>".to_string(),
        "<0xAC>".to_string(),
    ];

    // Feed one token at a time.
    let mut stream = DecodeSegmentStream::new(decoder.clone());
    let mut segments = Vec::new();
    for token in &tokens {
        if let Some(seg) = stream.step(token.clone()).unwrap() {
            segments.push(seg);
        }
    }
    if let Some(seg) = stream.flush().unwrap() {
        segments.push(seg);
    }

    // "叫" completes after the third byte; "a" is immediate; "本" completes after three bytes.
    assert_eq!(segments, vec!["叫".to_string(), "a".to_string(), "本".to_string()]);

    // The concatenation equals the batch decode.
    assert_stream_matches_batch(&decoder, &tokens);
}

#[test]
fn half_character_is_never_output_as_replacement_during_streaming() {
    let decoder = ByteLevelDecoder::new();
    let mut stream = DecodeSegmentStream::new(decoder);

    // Feed the first two bytes of "世" (E4 B8). Neither should produce output.
    assert_eq!(stream.step(bytes_to_token(&[0xE4])).unwrap(), None);
    assert!(stream.has_pending());
    assert_eq!(stream.step(bytes_to_token(&[0xB8])).unwrap(), None);
    assert!(stream.has_pending());

    // Complete the character with the third byte (96).
    assert_eq!(stream.step(bytes_to_token(&[0x96])).unwrap(), Some("世".to_string()));
    assert!(!stream.has_pending());
}
