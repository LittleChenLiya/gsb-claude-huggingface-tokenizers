pub mod bpe;
pub mod byte_fallback;
pub mod byte_level;
pub mod ctc;
pub mod fuse;
pub mod metaspace;
pub mod replace;
pub mod runtime;
pub mod strip;
pub mod wordpiece;

pub use runtime::DecoderRuntime;

use crate::tokenizer::Result;

/// A `Decoder` changes the raw tokens into its more readable form.
///
/// This lives here, with its implementors, rather than in `tokenizer` alongside the encode-side
/// traits: those were the legacy `NormalizedString` pipeline and are gone, while this one is what
/// `PipelineTokenizer::decode` actually dispatches through.
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let results = self.decode_chain(tokens)?;
        Ok(results.join(""))
    }
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;

    /// Decode a single token to its raw bytes, for streaming segment-boundary-aware decoding.
    ///
    /// This is used by [`DecodeSegmentStream`] to buffer incomplete UTF-8 sequences without
    /// outputting replacement characters. The default implementation decodes the token through
    /// [`decode_chain`](Self::decode_chain) and returns the bytes of the joined result. Decoders
    /// that convert tokens to raw bytes internally (byte-level, byte-fallback) override this to
    /// return the bytes directly, bypassing the lossy conversion that would otherwise emit `�`
    /// for a half-character.
    fn decode_token_bytes(&self, token: &str) -> Result<Vec<u8>> {
        let decoded = self.decode_chain(vec![token.to_string()])?;
        Ok(decoded.join("").into_bytes())
    }
}

/// A streaming decoder that tracks segment boundaries.
///
/// A **segment** is a complete unit of text. When decoding byte-level tokens, a token may contain
/// only part of a multi-byte UTF-8 character (a "half-character"). This stream buffers incomplete
/// byte sequences until they form a complete character, so half-characters are never output as
/// replacement characters (`�`).
///
/// The concatenation of all segments returned by [`step`](Self::step) and [`flush`](Self::flush)
/// equals the result of a batch [`Decoder::decode`] over the same tokens.
///
/// # Example
///
/// ```
/// use tk_encode::decoders::byte_level::ByteLevelDecoder;
/// use tk_encode::decoders::DecodeSegmentStream;
///
/// let decoder = ByteLevelDecoder::new();
/// let mut stream = DecodeSegmentStream::new(decoder);
///
/// // ASCII tokens form complete segments immediately.
/// assert_eq!(stream.step("Hello".to_string()).unwrap(), Some("Hello".to_string()));
///
/// // A multi-byte character split across tokens is buffered until complete.
/// // (The byte-level alphabet maps each byte to a printable char; here we feed the
/// // raw bytes of "世" = E4 B8 96 as three separate byte-level tokens.)
/// ```
pub struct DecodeSegmentStream<D: Decoder> {
    decoder: D,
    buffer: Vec<u8>,
}

impl<D: Decoder> DecodeSegmentStream<D> {
    /// Create a new streaming segment decoder wrapping `decoder`.
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            buffer: Vec::new(),
        }
    }

    /// Push one token and return the complete segment it formed, if any.
    ///
    /// Returns `Some(text)` when the token completed a segment, or `None` while a multi-byte
    /// UTF-8 sequence is still forming (the token's bytes are buffered internally).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying decoder fails to decode the token.
    pub fn step(&mut self, token: String) -> Result<Option<String>> {
        let bytes = self.decoder.decode_token_bytes(&token)?;
        self.buffer.extend_from_slice(&bytes);

        match str::from_utf8(&self.buffer) {
            Ok(s) => {
                let result = s.to_string();
                self.buffer.clear();
                Ok(Some(result))
            }
            Err(e) => {
                if e.error_len().is_none() {
                    // Incomplete UTF-8 sequence: keep buffering, no output yet.
                    Ok(None)
                } else {
                    // Invalid UTF-8 sequence: output the valid prefix and skip the invalid bytes.
                    let valid_up_to = e.valid_up_to();
                    let result = String::from_utf8(self.buffer[..valid_up_to].to_vec()).unwrap();
                    let invalid_len = e.error_len().unwrap();
                    self.buffer = self.buffer[valid_up_to + invalid_len..].to_vec();
                    Ok(Some(result))
                }
            }
        }
    }

    /// Flush any remaining buffered bytes as a final segment.
    ///
    /// Call this after all tokens have been processed. Any bytes that never formed a complete
    /// UTF-8 sequence are output as replacement characters, matching the batch decode behavior.
    pub fn flush(&mut self) -> Result<Option<String>> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        let result = String::from_utf8_lossy(&self.buffer).to_string();
        self.buffer.clear();
        Ok(Some(result))
    }

    /// Returns `true` if the stream has buffered bytes that have not yet formed a complete segment.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::byte_fallback::ByteFallback;
    use crate::decoders::byte_level::ByteLevelDecoder;
    use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

    /// Convert a byte slice to its byte-level token string (each byte -> its printable char).
    fn bytes_to_token(bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|&b| BYTES_CHAR_LOOKUP[b as usize])
            .collect()
    }

    /// Collect all segments from a stream, flushing at the end.
    fn collect_segments<D: Decoder>(stream: &mut DecodeSegmentStream<D>, tokens: &[String]) -> Vec<String> {
        let mut segments = Vec::new();
        for token in tokens {
            if let Some(seg) = stream.step(token.clone()).unwrap() {
                segments.push(seg);
            }
        }
        if let Some(seg) = stream.flush().unwrap() {
            segments.push(seg);
        }
        segments
    }

    /// The concatenation of all segments equals the batch decode result.
    fn assert_segments_match_batch(decoder: &(impl Decoder + Clone), tokens: &[String]) {
        let batch = decoder.decode(tokens.to_vec()).unwrap();
        let mut stream = DecodeSegmentStream::new(decoder.clone());
        let segments = collect_segments(&mut stream, tokens);
        let streamed: String = segments.join("");
        assert_eq!(streamed, batch, "streamed segments must equal batch decode");
    }

    #[test]
    fn ascii_tokens_form_complete_segments() {
        let decoder = ByteLevelDecoder::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        assert_eq!(stream.step("Hello".to_string()).unwrap(), Some("Hello".to_string()));
        assert_eq!(stream.step("Ġ".to_string()).unwrap(), Some(" ".to_string()));
        assert_eq!(stream.step("world".to_string()).unwrap(), Some("world".to_string()));
        assert!(!stream.has_pending());
        assert_eq!(stream.flush().unwrap(), None);
    }

    #[test]
    fn multi_byte_character_split_across_tokens_buffers_without_replacement() {
        let decoder = ByteLevelDecoder::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        // "世" = U+4E16 = E4 B8 96 in UTF-8. Split across three byte-level tokens.
        let token_e4 = bytes_to_token(&[0xE4]);
        let token_b8 = bytes_to_token(&[0xB8]);
        let token_96 = bytes_to_token(&[0x96]);

        // First byte: incomplete, no output.
        assert_eq!(stream.step(token_e4).unwrap(), None);
        assert!(stream.has_pending());

        // Second byte: still incomplete, no output.
        assert_eq!(stream.step(token_b8).unwrap(), None);
        assert!(stream.has_pending());

        // Third byte: complete character, output.
        assert_eq!(stream.step(token_96).unwrap(), Some("世".to_string()));
        assert!(!stream.has_pending());
    }

    #[test]
    fn multi_byte_character_followed_by_ascii() {
        let decoder = ByteLevelDecoder::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        // "世界" = E4 B8 96 E7 95 8C in UTF-8, split into individual bytes.
        let bytes: Vec<u8> = vec![0xE4, 0xB8, 0x96, 0xE7, 0x95, 0x8C];
        let mut segments = Vec::new();
        for &b in &bytes {
            if let Some(seg) = stream.step(bytes_to_token(&[b])).unwrap() {
                segments.push(seg);
            }
        }
        assert_eq!(segments, vec!["世".to_string(), "界".to_string()]);
        assert!(!stream.has_pending());
    }

    #[test]
    fn whitespace_tokens_form_segments() {
        let decoder = ByteLevelDecoder::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        // "Ġ" is the byte-level encoding of a space.
        assert_eq!(stream.step("Ġ".to_string()).unwrap(), Some(" ".to_string()));
        assert_eq!(stream.step("ĠĠ".to_string()).unwrap(), Some("  ".to_string()));
        assert!(!stream.has_pending());
    }

    #[test]
    fn special_tokens_pass_through_unchanged() {
        let decoder = ByteLevelDecoder::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        // Special tokens like "[PAD]" are not in the byte-level alphabet, so they
        // pass through as their own bytes (valid UTF-8), forming complete segments.
        assert_eq!(stream.step("[PAD]".to_string()).unwrap(), Some("[PAD]".to_string()));
        assert_eq!(stream.step("".to_string()).unwrap(), Some("".to_string()));
        assert!(!stream.has_pending());
    }

    #[test]
    fn flush_outputs_remaining_incomplete_bytes_as_replacement() {
        let decoder = ByteLevelDecoder::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        // Feed only the first byte of "世" (E4). It never completes.
        assert_eq!(stream.step(bytes_to_token(&[0xE4])).unwrap(), None);
        assert!(stream.has_pending());

        // Flush outputs the incomplete byte as a replacement character.
        assert_eq!(stream.flush().unwrap(), Some("\u{fffd}".to_string()));
        assert!(!stream.has_pending());
    }

    #[test]
    fn streamed_segments_equal_batch_decode_for_mixed_input() {
        let decoder = ByteLevelDecoder::new();

        // Mix of ASCII, whitespace, multi-byte chars, and special tokens.
        let tokens: Vec<String> = vec![
            "Hello".to_string(),
            "Ġ".to_string(),
            bytes_to_token(&[0xE4]),
            bytes_to_token(&[0xB8, 0x96]),
            bytes_to_token(&[0xE7, 0x95, 0x8C]),
            "Ġ".to_string(),
            "[PAD]".to_string(),
        ];

        assert_segments_match_batch(&decoder, &tokens);
    }

    #[test]
    fn streamed_segments_equal_batch_decode_for_individual_bytes() {
        let decoder = ByteLevelDecoder::new();

        // Every byte of "你好世界" as a separate token.
        let text = "你好世界";
        let tokens: Vec<String> = text
            .as_bytes()
            .iter()
            .map(|&b| bytes_to_token(&[b]))
            .collect();

        assert_segments_match_batch(&decoder, &tokens);
    }

    #[test]
    fn streamed_segments_equal_batch_decode_for_byte_fallback() {
        let decoder = ByteFallback::new();

        // ByteFallback decodes "<0x61>" style tokens to bytes.
        // "叫" = E5 8F AB, split across three byte tokens.
        let tokens: Vec<String> = vec![
            "<0xE5>".to_string(),
            "<0x8F>".to_string(),
            "<0xAB>".to_string(),
            "a".to_string(),
        ];

        assert_segments_match_batch(&decoder, &tokens);
    }

    #[test]
    fn byte_fallback_incomplete_bytes_buffer_without_replacement() {
        let decoder = ByteFallback::new();
        let mut stream = DecodeSegmentStream::new(decoder);

        // First two bytes of "叫" (E5 8F) are incomplete.
        assert_eq!(stream.step("<0xE5>".to_string()).unwrap(), None);
        assert!(stream.has_pending());
        assert_eq!(stream.step("<0x8F>".to_string()).unwrap(), None);
        assert!(stream.has_pending());

        // Third byte completes the character.
        assert_eq!(stream.step("<0xAB>".to_string()).unwrap(), Some("叫".to_string()));
        assert!(!stream.has_pending());
    }

    #[test]
    fn decode_token_bytes_bypasses_lossy_conversion() {
        let decoder = ByteLevelDecoder::new();

        // decode_token_bytes returns raw bytes, not a lossy string.
        // "世" = E4 B8 96. A single byte token returns one byte.
        assert_eq!(decoder.decode_token_bytes(&bytes_to_token(&[0xE4])).unwrap(), vec![0xE4]);
        assert_eq!(
            decoder.decode_token_bytes(&bytes_to_token(&[0xE4, 0xB8, 0x96])).unwrap(),
            vec![0xE4, 0xB8, 0x96]
        );

        // A non-byte-level token returns its own bytes.
        assert_eq!(decoder.decode_token_bytes("Hello").unwrap(), b"Hello".to_vec());
    }
}
