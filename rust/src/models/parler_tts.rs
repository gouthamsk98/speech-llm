use candle_transformers::models::parler_tts::{ Config, Model };
use candle_examples::token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;
use anyhow::{ Error as E, Result };
use clap::Parser;

use candle_core::{ DType, IndexOp, Tensor, Device };
use candle_nn::VarBuilder;
pub struct ParlerTTS {
    model: Model,
    tokenizer: Tokenizer,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    device: Device,
    logits_processor: LogitsProcessor,
    temperature: Option<f64>,
    top_p: Option<f64>,
}
impl ParlerTTS {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
        temperature: Option<f64>
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            logits_processor,
            tokenizer,
            seed,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
            temperature,
            top_p,
        }
    }
    pub fn run(&mut self, prompt: &str, description: &str, max_step: usize) -> Result<()> {
        let description_tokens = self.tokenizer
            .encode(description, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let description_tokens = Tensor::new(description_tokens, &self.device)?.unsqueeze(0)?;
        let prompt_tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?.get_ids().to_vec();
        let prompt_tokens = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;
        let lp = candle_transformers::generation::LogitsProcessor::new(
            self.seed,
            self.temperature,
            self.top_p
        );
        println!("starting generation...");
        let codes = self.model.generate(&prompt_tokens, &description_tokens, lp, max_step)?;
        println!("generated codes\n{codes}");
        let codes = codes.to_dtype(DType::I64)?;
        codes.save_safetensors("codes", "out.safetensors")?;
        let codes = codes.unsqueeze(0)?;
        let pcm = self.model.audio_encoder.decode_codes(&codes.to_device(&self.device)?)?;
        let pcm = pcm.i((0, 0))?;
        let pcm = candle_examples::audio::normalize_loudness(&pcm, 24_000, true)?;
        let pcm = pcm.to_vec1::<f32>()?;
        let mut output = std::fs::File::create("./output.wav")?;
        candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 44100)?;
        Ok(())
    }
}
