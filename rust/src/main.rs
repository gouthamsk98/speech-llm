use anyhow::{ Error as E, Result };

use hf_hub::{ api::sync::Api, Repo, RepoType };
use models::tts::TTS;
use symphonia::core::{ meta, sample };
use tokenizers::Tokenizer;
use candle_transformers::models::whisper::{ self as m, audio, Config };
use candle_transformers::models::stable_lm::{ Model as StableLM, self };
use candle_core::{ Tensor, DType };
use models::whisper::WhisperModel;
use models::textllm::{ Model as TextLLM, TextGeneration };
use candle_nn::{ ops::softmax, VarBuilder };
use candle_examples::device;
use crate::models::whisper::{ Decoder, Task };
use candle_transformers::models::whisper;
use rubato::{ Resampler, FastFixedIn, PolynomialDegree };
use std::error::Error;
use candle_transformers::models::metavoice::{
    adapters,
    gpt,
    tokenizers as meta_tokenizers,
    transformer,
};
use candle_transformers::models::encodec;
use crate::models::tts::TTSModel;
use candle_transformers::models::quantized_metavoice::transformer as qtransformer;

mod pcm_decode;
mod models;
mod multilingual;

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

fn resample_audio(input: &[f32], sample_rate: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    // Check if resampling is needed
    if sample_rate != (whisper::SAMPLE_RATE as u32) {
        // Compute the resampling ratio
        let resample_ratio = (whisper::SAMPLE_RATE as f64) / (sample_rate as f64);

        // Create the resampler
        let mut resampler = FastFixedIn::<f32>::new(
            resample_ratio,
            10.0, // Number of chunks to process
            PolynomialDegree::Septic,
            1024, // Chunk size
            1 // Number of channels
        )?;

        // Prepare input as a vector of slices (one per channel)
        let input_vec: Vec<&[f32]> = vec![input];

        // Resample the audio
        let output = resampler.process(&input_vec, None)?;

        // The output is a vector of slices, combine it into a single Vec<f32>
        let resampled_audio: Vec<f32> = output[0].to_vec();

        Ok(resampled_audio)
    } else {
        // If no resampling is needed, return the input
        Ok(input.to_vec())
    }
}
#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ArgDType {
    F32,
    F16,
    Bf16,
}
fn main() -> Result<()> {
    let whisper_model_id = "openai/whisper-tiny".to_string();
    // let whisper_model_id = "openai/whisper-base.en".to_string();
    let whisper_revision = "main".to_string();
    let llm_model_id = "stabilityai/stablelm-2-zephyr-1_6b".to_string();
    let llm_revision = "main".to_string();
    let device = device(false)?;

    let seed = 299792458;
    let sample_len = 64;
    let temperature: Option<f64> = None;
    let top_p: Option<f64> = None;
    let repeat_penalty = 1.1;
    let repeat_last_n: usize = 64;
    let guidance_scale: f64 = 3.0;
    let max_token: u64 = 256;

    let input = std::path::PathBuf::from("input_3.wav");

    let api = Api::new()?;
    //load whisper model
    let whisper_repo = api.repo(
        Repo::with_revision(whisper_model_id, RepoType::Model, whisper_revision)
    );
    let (config_filename, tokenizer_filename, weights_filename) = {
        let config = whisper_repo.get("config.json")?;
        let tokenizer = whisper_repo.get("tokenizer.json")?;
        let model = whisper_repo.get("model.safetensors")?;
        (config, tokenizer, model)
    };

    let whisper_config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let whisper_tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let whisper_model = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)?
        };
        WhisperModel::Normal(m::model::Whisper::load(&vb, whisper_config.clone())?)
    };

    //load textllm model
    let llm_repo = api.repo(Repo::with_revision(llm_model_id, RepoType::Model, llm_revision));
    let (llm_config_filename, llm_tokenizer_filename, llm_weights_filenames) = {
        let config = llm_repo.get("config.json")?;
        let tokenizer = llm_repo.get("tokenizer.json")?;
        let model = vec![llm_repo.get("model.safetensors")?];
        (config, tokenizer, model)
    };
    let llm_config: stable_lm::Config = serde_json::from_str(
        &std::fs::read_to_string(llm_config_filename)?
    )?;
    let llm_tokenizer = Tokenizer::from_file(llm_tokenizer_filename).map_err(E::msg)?;
    let dtype = if device.is_metal() { DType::BF16 } else { DType::F32 };
    let llm_model = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&llm_weights_filenames, dtype, &device)?
        };
        let model = StableLM::new(&llm_config, vb)?;
        TextLLM::StableLM(model)
    };

    //load metavoice model
    let meta_repo = api.model("lmz/candle-metavoice".to_string());
    let first_stage_meta = meta_repo.get("first_stage.meta.json")?;
    let first_stage_meta: serde_json::Value = serde_json::from_reader(
        &std::fs::File::open(first_stage_meta)?
    )?;
    let first_stage_tokenizer = match first_stage_meta.as_object() {
        None => anyhow::bail!("not a json object"),
        Some(j) =>
            match j.get("tokenizer") {
                None => anyhow::bail!("no tokenizer key"),
                Some(j) => j,
            }
    };
    let fs_tokenizer = meta_tokenizers::BPE::from_json(first_stage_tokenizer, 512)?;
    let second_stage_weights = meta_repo.get("second_stage.safetensors")?;
    let encodec_weights = api.model("facebook/encodec_24khz".to_string()).get("model.safetensors")?;
    let first_stage_config = transformer::Config::cfg1b_v0_1();
    // let mut first_stage_model = {
    //     let first_stage_weights = meta_repo.get("first_stage.safetensors")?;
    //     let first_stage_vb = unsafe {
    //         VarBuilder::from_mmaped_safetensors(&[first_stage_weights], dtype, &device)?
    //     };
    //     let first_stage_model = transformer::Model::new(&first_stage_config, first_stage_vb)?;
    //     TTSModel::Normal(first_stage_model)
    // };
    let first_stage_model = {
        let first_stage_weights = meta_repo.get("first_stage_q4k.gguf")?;
        let first_stage_vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            first_stage_weights,
            &device
        )?;
        let first_stage_model = qtransformer::Model::new(&first_stage_config, first_stage_vb)?;
        TTSModel::Quantized(first_stage_model)
    };
    let second_stage_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[second_stage_weights], dtype, &device)?
    };
    let second_stage_config = gpt::Config::cfg1b_v0_1();
    let second_stage_model = gpt::Model::new(second_stage_config.clone(), second_stage_vb)?;
    let encodec_device = if device.is_metal() { &candle_core::Device::Cpu } else { &device };
    let encodec_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[encodec_weights], DType::F32, encodec_device)?
    };
    let encodec_config = encodec::Config::default();
    let encodec_model = encodec::Model::new(&encodec_config, encodec_vb)?;

    let spk_emb_file = meta_repo.get("spk_emb.safetensors")?;
    let spk_emb = candle_core::safetensors::load(&spk_emb_file, &device)?;
    let spk_emb = match spk_emb.get("spk_emb") {
        None => anyhow::bail!("missing spk_emb tensor in {spk_emb_file:?}"),
        Some(spk_emb) => spk_emb.to_dtype(dtype)?,
    };
    let spk_emb = spk_emb.to_device(&device)?;

    //print model dimention
    let model_dim = whisper_model.config().encoder_layers;
    println!("model_dim: {:?}", model_dim);

    let mel_bytes = match whisper_config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    let (pcm_data, sample_rate) = pcm_decode::pcm_decode(input)?;
    if sample_rate != (whisper::SAMPLE_RATE as u32) {
        anyhow::bail!("unexpected sample rate {sample_rate}");
    }
    let mel = audio::pcm_to_mel(&whisper_config, &pcm_data, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, whisper_config.num_mel_bins, mel_len / whisper_config.num_mel_bins),
        &device
    )?;
    println!("loaded mel: {:?}", mel.dims());

    // detect language
    // let language_token = multilingual::detect_language(&mut model, &tokenizer, &mel)?;
    let language_token = token_id(&whisper_tokenizer, &format!("<|en|>"))?;
    let mut decoder = Decoder::new(
        whisper_model,
        whisper_tokenizer,
        seed,
        &device,
        None,
        Some(Task::Transcribe),
        false,
        false
    )?;
    decoder.set_language_token(Some(language_token));
    let decode = decoder.run(&mel, None)?;
    let prompt = if let Some(first_segment) = decode.get(0) {
        &first_segment.dr.text
    } else {
        "sorry I didn't get that"
    };

    let mut pipeline = TextGeneration::new(
        llm_model,
        llm_tokenizer,
        seed,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
        &device
    );
    let reply = pipeline.run(&prompt, sample_len)?;
    println!("reply: {reply}");
    let mut meta_pipeline = TTS::new(
        first_stage_model,
        second_stage_model,
        encodec_model,
        fs_tokenizer,
        seed,
        spk_emb,
        Some(1.0),
        top_p,
        repeat_penalty,
        repeat_last_n,
        guidance_scale,
        &device,
        &encodec_device
    );
    meta_pipeline.run(&reply, max_token)?;
    decoder.reset_kv_cache();
    Ok(())
}
