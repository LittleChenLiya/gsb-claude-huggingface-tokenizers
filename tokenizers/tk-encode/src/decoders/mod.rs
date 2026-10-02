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
}

/// What [`FragmentDecoder::push`] reports for the token it just consumed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragmentBoundary {
    /// The token completed a text fragment; the payload is the newly
    /// completed text. It never ends with a half UTF-8 character: a
    /// trailing `U+FFFD` (a byte token whose character has not fully
    /// arrived yet) is held back instead of being emitted.
    Complete(String),
    /// The token is still mid-fragment — e.g. one byte of a multi-byte
    /// UTF-8 character arrived but the character is not whole yet.
    /// Nothing was emitted for this token.
    Pending,
}

/// Streaming decode with fragment boundaries, over any [`Decoder`].
///
/// Tokens are pushed one at a time; each [`push`](Self::push) tells the
/// caller whether that token completed a text fragment
/// ([`FragmentBoundary::Complete`]) or is still forming one
/// ([`FragmentBoundary::Pending`]). The guarantee on the emitted pieces:
///
/// * no half characters — a trailing `U+FFFD` replacement char is never
///   emitted, because it may still resolve into a real character once the
///   remaining byte tokens arrive;
/// * lossless — the concatenation of every `Complete` payload plus the
///   final [`flush`](Self::flush) equals the one-shot
///   [`Decoder::decode`] of the same token sequence. `flush` is what
///   emits a genuinely invalid trailing byte (as `U+FFFD`), so invalid
///   input round-trips exactly like the batch decode.
///
/// Tokens not yet emitted are kept as decode context, so cross-token
/// decoders (byte fallback, WordPiece `##`, …) see the same input a
/// one-shot `decode` would.
#[derive(Clone, Debug)]
pub struct FragmentDecoder<D> {
    decoder: D,
    /// Tokens whose decoded text is not fully emitted yet.
    pending: Vec<String>,
    /// Text already handed to the caller; a prefix of every future
    /// decode of `pending`.
    emitted: String,
}

impl<D: Decoder> FragmentDecoder<D> {
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            pending: Vec::new(),
            emitted: String::new(),
        }
    }

    /// Push one token and report whether it completed a fragment.
    pub fn push(&mut self, token: impl Into<String>) -> Result<FragmentBoundary> {
        self.pending.push(token.into());
        let decoded = self.decoder.decode(self.pending.clone())?;
        let Some(delta) = decoded.strip_prefix(self.emitted.as_str()) else {
            return Err(format!(
                "fragment decoder: {decoded:?} does not extend emitted prefix {:?}",
                self.emitted
            )
            .into());
        };
        // Hold back a trailing U+FFFD run: it may be a half UTF-8
        // character whose remaining byte tokens have not arrived yet.
        let complete = delta.trim_end_matches('\u{fffd}');
        if complete.is_empty() {
            Ok(FragmentBoundary::Pending)
        } else {
            let fragment = complete.to_string();
            self.emitted.push_str(&fragment);
            Ok(FragmentBoundary::Complete(fragment))
        }
    }

    /// End of stream: emit whatever is still held back, including any
    /// trailing `U+FFFD` that turned out to be a genuinely invalid byte
    /// rather than a half character. After this the decoder is reusable.
    pub fn flush(&mut self) -> Result<String> {
        let decoded = self.decoder.decode(std::mem::take(&mut self.pending))?;
        let Some(rest) = decoded.strip_prefix(self.emitted.as_str()) else {
            return Err(format!(
                "fragment decoder: {decoded:?} does not extend emitted prefix {:?}",
                self.emitted
            )
            .into());
        };
        let rest = rest.to_string();
        self.emitted.clear();
        Ok(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::byte_fallback::ByteFallback;

    /// Feed `tokens` one at a time, returning every boundary and the
    /// concatenation of all fragments (flush included).
    fn run_stream<D: Decoder>(
        decoder: D,
        tokens: &[&str],
    ) -> (Vec<FragmentBoundary>, String) {
        let mut stream = FragmentDecoder::new(decoder);
        let mut boundaries = Vec::new();
        let mut glued = String::new();
        for token in tokens {
            let boundary = stream.push(*token).unwrap();
            if let FragmentBoundary::Complete(fragment) = &boundary {
                glued.push_str(fragment);
            }
            boundaries.push(boundary);
        }
        glued.push_str(&stream.flush().unwrap());
        (boundaries, glued)
    }

    #[test]
    fn fragment_boundaries_multibyte_whitespace_special() {
        // Multi-byte character 叫 = E5 8F AB arriving one byte token at a
        // time, whitespace tokens, and a special token, all through the
        // byte-fallback decoder.
        let tokens = [
            "<s>", "Hey", " ", "<0xE5>", "<0x8f>", "<0xab>", " ", "world", "!",
        ];
        let (boundaries, glued) = run_stream(ByteFallback::new(), &tokens);
        assert_eq!(
            boundaries,
            vec![
                FragmentBoundary::Complete("<s>".into()),
                FragmentBoundary::Complete("Hey".into()),
                FragmentBoundary::Complete(" ".into()),
                // Half of 叫: no replacement char may leak out.
                FragmentBoundary::Pending,
                FragmentBoundary::Pending,
                FragmentBoundary::Complete("叫".into()),
                FragmentBoundary::Complete(" ".into()),
                FragmentBoundary::Complete("world".into()),
                FragmentBoundary::Complete("!".into()),
            ]
        );
        // The glued fragments equal the one-shot batch decode.
        let batch = ByteFallback::new()
            .decode(tokens.iter().map(|t| t.to_string()).collect())
            .unwrap();
        assert_eq!(glued, "<s>Hey 叫 world!");
        assert_eq!(glued, batch);
    }

    #[test]
    fn fragment_boundaries_invalid_trailing_byte_flushes_replacement() {
        // A genuinely invalid trailing byte stays Pending while streamed
        // and surfaces as U+FFFD only at flush, matching the batch decode.
        let tokens = ["a", "<0xE5>"];
        let (boundaries, glued) = run_stream(ByteFallback::new(), &tokens);
        assert_eq!(
            boundaries,
            vec![
                FragmentBoundary::Complete("a".into()),
                FragmentBoundary::Pending,
            ]
        );
        let batch = ByteFallback::new()
            .decode(tokens.iter().map(|t| t.to_string()).collect())
            .unwrap();
        assert_eq!(glued, "a\u{fffd}");
        assert_eq!(glued, batch);
    }

    #[test]
    fn fragment_boundaries_invalid_byte_mid_stream() {
        // An invalid byte followed by a normal token: the replacement
        // char is emitted once it is known not to be a half character,
        // exactly where the batch decode puts it.
        let tokens = ["<0xE5>", "b"];
        let (boundaries, glued) = run_stream(ByteFallback::new(), &tokens);
        assert_eq!(
            boundaries,
            vec![
                FragmentBoundary::Pending,
                FragmentBoundary::Complete("\u{fffd}b".into()),
            ]
        );
        let batch = ByteFallback::new()
            .decode(tokens.iter().map(|t| t.to_string()).collect())
            .unwrap();
        assert_eq!(glued, batch);
    }
}
