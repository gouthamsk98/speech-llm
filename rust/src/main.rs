use anyhow::{ Error as E, Result };

use hf_hub::{ api::sync::Api, Repo, RepoType };
use tokenizers::Tokenizer;
use candle_transformers::models::whisper::{ self as m, audio, Config };
use candle_core::Tensor;
use models::layers::Model;
use candle_nn::{ ops::softmax, VarBuilder };
use candle_examples::device;

mod pcm_decode;
mod models;
mod multilingual;
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}
fn main() -> Result<()> {
    let model_id = "openai/whisper-tiny".to_string();
    // let model_id = "lmz/candle-whisper".to_string();
    let revision = "main".to_string();
    let device = device(true)?;

    let input = std::path::PathBuf::from("input.wav");

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let (config_filename, tokenizer_filename, weights_filename) = {
        let config = repo.get("config.json")?;
        let tokenizer = repo.get("tokenizer.json")?;
        let model = repo.get("model.safetensors")?;
        (config, tokenizer, model)
    };

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let mut model = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)?
        };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };

    //print model dimention
    let model_dim = model.config().encoder_layers;
    println!("model_dim: {:?}", model_dim);

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
    let (pcm_data, sample_rate) = pcm_decode::pcm_decode(input)?;
    println!("pcm data loaded {}", pcm_data.len());
    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &device
    )?;
    println!("loaded mel: {:?}", mel.dims());

    // detect language
    let language_token = multilingual::detect_language(&mut model, &tokenizer, &mel)?;
    println!("language_token: {:?}", language_token);
    Ok(())
}
