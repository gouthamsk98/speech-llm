use candle_core::{ Device, IndexOp, Tensor };
use candle_nn::{ ops::softmax, VarBuilder };
use candle_transformers::models::whisper::{ self as m, audio, Config };
use anyhow::{ Error as E, Result };
use tokenizers::Tokenizer;
use rand::{ distributions::Distribution, SeedableRng };
use crate::token_id;
use clap::{ Parser, ValueEnum };
use candle_examples::device;
pub enum WhisperModel {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}

// Maybe we should use some traits rather than doing the dispatch for all these.
impl WhisperModel {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DecodingResult {
    tokens: Vec<u32>,
    pub text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Segment {
    start: f64,
    duration: f64,
    pub dr: DecodingResult,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Task {
    Transcribe,
    Translate,
}
pub struct Decoder {
    model: WhisperModel,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}
impl Decoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: WhisperModel,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if
                    model.config().suppress_tokens.contains(&i) ||
                    (timestamps && i == no_timestamps_token)
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    pub fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model
                    .decoder_final_linear(&ys.i(..1)?)?
                    .i(0)?
                    .i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / (tokens.len() as f64);

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback =
                        dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD ||
                        dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}");
                }
            }
        }
        unreachable!()
    }

    pub fn run(&mut self, mel: &Tensor, times: Option<(f64, f64)>) -> Result<Vec<Segment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        while seek < content_frames {
            let start = std::time::Instant::now();
            let time_offset = ((seek * m::HOP_LENGTH) as f64) / (m::SAMPLE_RATE as f64);
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration =
                ((segment_size * m::HOP_LENGTH) as f64) / (m::SAMPLE_RATE as f64);
            let dr = self.decode_with_fallback(&mel_segment)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!("{:.1}s -- {:.1}s", segment.start, segment.start + segment.duration);
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = ((token - self.no_timestamps_token + 1) as f32) / 50.0;
                        if !tokens_to_decode.is_empty() {
                            let text = self.tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear();
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token);
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self.tokenizer.decode(&tokens_to_decode, true).map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear();
                }
            } else {
                match times {
                    Some((start, end)) => {
                        println!("{:.1}s -- {:.1}s: {}", start, end, segment.dr.text);
                    }
                    None => {
                        println!(
                            "{:.1}s -- {:.1}s: {}",
                            segment.start,
                            segment.start + segment.duration,
                            segment.dr.text
                        );
                    }
                }
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment);
        }
        Ok(segments)
    }

    pub fn set_language_token(&mut self, language_token: Option<u32>) {
        self.language_token = language_token;
    }

    #[allow(dead_code)]
    pub fn reset_kv_cache(&mut self) {
        match &mut self.model {
            WhisperModel::Normal(m) => m.reset_kv_cache(),
            WhisperModel::Quantized(m) => m.reset_kv_cache(),
        }
    }

    fn model(&mut self) -> &mut WhisperModel {
        &mut self.model
    }
}

pub fn load_model(device: Device) -> Decoder {
    use hf_hub::{ api::sync::Api, Repo, RepoType };
    println!("loading whisper model...");
    let start = std::time::Instant::now();
    let seed = 299792458;
    let api = Api::new().unwrap();
    let whisper_repo = api.repo(
        Repo::with_revision("openai/whisper-tiny".to_string(), RepoType::Model, "main".to_string())
    );
    let (config_filename, tokenizer_filename, weights_filename) = {
        let config = whisper_repo.get("config.json").unwrap();
        let tokenizer = whisper_repo.get("tokenizer.json").unwrap();
        let model = whisper_repo.get("model.safetensors").unwrap();
        (config, tokenizer, model)
    };
    let whisper_config: Config = serde_json
        ::from_str(&std::fs::read_to_string(config_filename).unwrap())
        .unwrap();
    let whisper_tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg).unwrap();
    let whisper_model = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device).unwrap()
        };
        WhisperModel::Normal(m::model::Whisper::load(&vb, whisper_config.clone()).unwrap())
    };
    let language_token = token_id(&whisper_tokenizer, &format!("<|en|>")).unwrap();
    let mut decoder = Decoder::new(
        whisper_model,
        whisper_tokenizer,
        seed,
        &device,
        None,
        Some(Task::Transcribe),
        false,
        false
    ).unwrap();
    decoder.set_language_token(Some(language_token));
    println!("loaded whisper model in {:?}", start.elapsed());
    decoder
}
pub fn get_config() -> Config {
    use hf_hub::{ api::sync::Api, Repo, RepoType };
    let api = Api::new().unwrap();
    let whisper_repo = api.repo(
        Repo::with_revision("openai/whisper-tiny".to_string(), RepoType::Model, "main".to_string())
    );
    let config_filename = whisper_repo.get("config.json").unwrap();
    let whisper_config: Config = serde_json
        ::from_str(&std::fs::read_to_string(config_filename).unwrap())
        .unwrap();
    whisper_config
}
