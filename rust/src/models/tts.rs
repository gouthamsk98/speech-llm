use anyhow::Ok;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::encodec;
use candle_transformers::models::flux::autoencoder::DiagonalGaussian;
use candle_transformers::models::metavoice::{ adapters, gpt, tokenizers, transformer };
use candle_transformers::models::quantized_metavoice::transformer as qtransformer;
use candle_core::{ DType, Device, Tensor };
use anyhow::{ Error as E, Result };
use rand::{ distributions::Distribution, SeedableRng };

pub const ENCODEC_NTOKENS: u32 = 1024;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ArgDType {
    F32,
    F16,
    Bf16,
}
pub enum TTSModel {
    Normal(transformer::Model),
    Quantized(qtransformer::Model),
}
pub struct TTS {
    first_stage_model: TTSModel,
    second_stage_model: gpt::Model,
    encodec_model: encodec::Model,
    spk_emb: Tensor,
    device: Device,
    encodec_device: Device,
    seed: u64,
    tokenizer: tokenizers::BPE,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    guidance_scale: f64,
}
impl TTS {
    pub fn new(
        first_stage_model: TTSModel,
        second_stage_model: gpt::Model,
        encodec_model: encodec::Model,
        tokenizer: tokenizers::BPE,
        seed: u64,
        spk_emb: Tensor,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        guidance_scale: f64,
        device: &Device,
        encodec_device: &Device
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        Self {
            first_stage_model,
            second_stage_model,
            encodec_model,
            seed,
            spk_emb,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            guidance_scale,
            device: device.clone(),
            encodec_device: encodec_device.clone(),
        }
    }
    pub fn run(&mut self, text: &str, max_token: u64) -> Result<()> {
        let prompt_tokens = self.tokenizer.encode(text)?;
        let mut tokens = prompt_tokens.clone();
        use std::io::Write;

        // First stage generation.
        for index in 0..max_token {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?;
            let input = Tensor::stack(&[&input, &input], 0)?;
            let logits = match &mut self.first_stage_model {
                TTSModel::Normal(m) =>
                    m.forward(&input, &self.spk_emb, tokens.len() - context_size)?,
                TTSModel::Quantized(m) => {
                    m.forward(&input, &self.spk_emb, tokens.len() - context_size)?
                }
            };
            let logits0 = logits.get(0)?.get(0)?; // Access the first tensor and the first element
            let logits1 = logits.get(1)?.get(0)?;
            let logits = ((logits0 * self.guidance_scale)? +
                logits1 * (1.0 - self.guidance_scale))?;
            let logits = logits.to_dtype(DType::F32)?;
            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            print!(".");
            std::io::stdout().flush()?;
            if next_token == 2048 {
                break;
            }
        }
        let fie2c = adapters::FlattenedInterleavedEncodec2Codebook::new(ENCODEC_NTOKENS);
        let (text_ids, ids1, ids2) = fie2c.decode(&tokens);
        println!("text ids len: {}", text_ids.len());
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed + 1337);
        // TODO: Use the config rather than hardcoding the offset here.
        let encoded_text: Vec<_> = prompt_tokens
            .iter()
            .map(|v| v - 1024)
            .collect();
        let mut hierarchies_in1 = [
            encoded_text.as_slice(),
            ids1.as_slice(),
            &[ENCODEC_NTOKENS],
        ].concat();
        let mut hierarchies_in2 = [
            vec![ENCODEC_NTOKENS; encoded_text.len()].as_slice(),
            ids2.as_slice(),
            &[ENCODEC_NTOKENS],
        ].concat();
        let second_stage_config = gpt::Config::cfg1b_v0_1();
        hierarchies_in1.resize(second_stage_config.block_size, ENCODEC_NTOKENS);
        hierarchies_in2.resize(second_stage_config.block_size, ENCODEC_NTOKENS);
        let in_x1 = Tensor::new(hierarchies_in1, &self.device)?;
        let in_x2 = Tensor::new(hierarchies_in2, &self.device)?;
        let in_x = Tensor::stack(&[in_x1, in_x2], 0)?.unsqueeze(0)?;
        let logits = self.second_stage_model.forward(&in_x)?;
        println!("sampling from logits...");
        let mut codes = vec![];
        for logits in logits.iter() {
            let logits = logits.squeeze(0)?;
            let (seq_len, _) = logits.dims2()?;
            let mut codes_ = Vec::with_capacity(seq_len);
            for step in 0..seq_len {
                let logits = logits.get(step)?.to_dtype(DType::F32)?;
                let logits = &(&logits / 1.0)?;
                let prs = candle_nn::ops::softmax_last_dim(logits)?.to_vec1::<f32>()?;
                let distr = rand::distributions::WeightedIndex::new(prs.as_slice())?;
                let sample = distr.sample(&mut rng) as u32;
                codes_.push(sample);
            }
            codes.push(codes_);
        }
        let codes = Tensor::new(codes, &self.device)?.unsqueeze(0)?;
        let codes = Tensor::cat(&[in_x, codes], 1)?;
        println!("codes: {codes}");
        let tilted_encodec = adapters::TiltedEncodec::new(ENCODEC_NTOKENS);
        let codes = codes.get(0)?.to_vec2::<u32>()?;
        let (text_ids, audio_ids) = tilted_encodec.decode(&codes);
        println!("text_ids len: {:?}", text_ids.len());
        let audio_ids = Tensor::new(audio_ids, &self.encodec_device)?.unsqueeze(0)?;
        println!("audio_ids shape: {:?}", audio_ids.shape());
        let pcm = self.encodec_model.decode(&audio_ids)?;
        println!("output pcm shape: {:?}", pcm.shape());
        let pcm = pcm.get(0)?.get(0)?.to_dtype(DType::F32)?;
        let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
        let pcm = pcm.to_vec1::<f32>()?;
        let mut output = std::fs::File::create("./output.wav")?;
        candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;
        Ok(())
    }
}
