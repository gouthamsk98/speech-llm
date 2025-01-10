use candle_transformers::models::{ flux::model, quantized_stable_lm::Model as QStableLM };
use candle_transformers::models::stable_lm::{ Model as StableLM, Config };
use candle_core::{ DType, Device, Tensor };
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use anyhow::{ Error as E, Result };
use clap::{ Parser, ValueEnum };
use tokenizers::Tokenizer;
pub enum Model {
    StableLM(StableLM),
    Quantized(QStableLM),
}
pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}
fn sample_top_k(logits: &Tensor, k: usize) -> Result<usize> {
    let logits = logits.to_dtype(DType::F32)?;
    use rand::prelude::*;
    let logits_vec = logits.to_vec1::<f32>()?;
    let mut indices: Vec<usize> = (0..logits_vec.len()).collect();

    // Sort indices by logits in descending order
    indices.sort_by(|&a, &b| logits_vec[b].partial_cmp(&logits_vec[a]).unwrap());

    // Keep only the top-k tokens
    let top_k = &indices[..k.min(indices.len())];

    // Extract top-k logits and normalize them to probabilities
    let top_k_logits: Vec<f32> = top_k
        .iter()
        .map(|&i| logits_vec[i])
        .collect();
    let max_logit = top_k_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let top_k_probs: Vec<f32> = top_k_logits
        .iter()
        .map(|&logit| (logit - max_logit).exp())
        .collect();
    let sum_probs: f32 = top_k_probs.iter().sum();
    let normalized_probs: Vec<f32> = top_k_probs
        .iter()
        .map(|&prob| prob / sum_probs)
        .collect();

    // Sample a token from the top-k set based on probabilities
    let dist = rand::distributions::WeightedIndex
        ::new(&normalized_probs)
        .map_err(|_| anyhow::anyhow!("Failed to create WeightedIndex"))?;
    let mut rng = rand::thread_rng();
    let sampled_index = dist.sample(&mut rng);

    // Return the token ID
    Ok(top_k[sampled_index])
}
impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        self.tokenizer.clear();
        match &mut self.model {
            Model::StableLM(m) => m.reset(),
            Model::Quantized(m) => {}
        }
        let mut tokens = self.tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut result = String::new();

        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                result.push_str(&t);
            }
        }
        let prompt_len = result.len();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::StableLM(m) => m.forward(&input, start_pos)?,
                Model::Quantized(m) => m.forward(&input, start_pos)?,
            };
            let mut logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..]
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;

            // Apply top-k sampling
            // let next_token = sample_top_k(&logits, 2)? as u32;

            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                result.push_str(&t);
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            result.push_str(&rest);
        }

        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            (generated_tokens as f64) / dt.as_secs_f64()
        );
        //remove intial pushed token
        let generated_result = &result[prompt_len..];
        Ok(generated_result.to_string())
    }
}
pub fn load_model(device: Device) -> TextGeneration {
    use candle_transformers::models::stable_lm::{ self };
    use hf_hub::{ api::sync::Api, Repo, RepoType };
    println!("loading textllm model...");
    let start = std::time::Instant::now();
    let api = Api::new().unwrap();
    let llm_repo = api.repo(
        Repo::with_revision(
            "stabilityai/stablelm-2-zephyr-1_6b".to_string(),
            RepoType::Model,
            "main".to_string()
        )
    );
    //non quantized model
    let (llm_config_filename, llm_tokenizer_filename, llm_weights_filenames) = {
        let config = llm_repo.get("config.json").unwrap();
        let tokenizer = llm_repo.get("tokenizer.json").unwrap();
        let model = vec![llm_repo.get("model.safetensors").unwrap()];
        (config, tokenizer, model)
    };
    //quantized model
    // let (llm_config_filename, llm_tokenizer_filename, llm_weights_filenames) = {
    //     let config = llm_repo.get("config.json").unwrap();
    //     let tokenizer = llm_repo.get("tokenizer.json").unwrap();
    //     let model = vec![llm_repo.get("stablelm-2-1_6b-q4k.gguf").unwrap()];
    //     (config, tokenizer, model)
    // };

    // let llm_config: Config = serde_json
    //     ::from_str(&std::fs::read_to_string(llm_config_filename).unwrap())
    //     .unwrap();
    let llm_config = {
        let config = std::fs::read_to_string(llm_config_filename).unwrap();
        let mut config: Config = serde_json::from_str(&config).unwrap();
        config.set_use_flash_attn(device.is_cuda());
        config
    };
    let llm_tokenizer = Tokenizer::from_file(llm_tokenizer_filename).map_err(E::msg).unwrap();
    let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };
    let llm_model = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&llm_weights_filenames, dtype, &device).unwrap()
        };
        let model = StableLM::new(&llm_config, vb).unwrap();
        Model::StableLM(model)
    };
    let seed = 299792458;
    let temperature: Option<f64> = Some(0.9);
    let top_p: Option<f64> = None;
    let repeat_penalty = 1.1;
    let repeat_last_n: usize = 64;
    let pipeline = TextGeneration::new(
        llm_model,
        llm_tokenizer,
        seed,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &device
    );
    println!("loaded textllm model in {:.2?}", start.elapsed());
    pipeline
}
