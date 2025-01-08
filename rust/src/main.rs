#[macro_use]
extern crate rocket;
mod pcm_decode;
mod models;
mod utils;
mod cli;
use std::env;
use candle_transformers::pipelines;

use models::parler_tts::ParlerTTS;
use models::tts::TTS;
use models::{ whisper, textllm };
use candle_examples::device;
use rocket::{ Data, State, tokio::io::AsyncReadExt };
use std::sync::Arc;
use rocket::data::ToByteUnit;
use anyhow::{ Error as E, Result, self };
use hound::{ WavSpec, WavWriter };
use candle_transformers::models::whisper::{ self as m, audio };
pub enum TTSType {
    ParlerTTS,
    MetaVoice,
    GTTS,
}

pub enum TTSPipeline {
    ParlerTTS(ParlerTTS),
    MetaVoice(TTS),
    GTTS,
}
use candle_transformers::models::whisper::Config;
struct Pipelines {
    whisper_pipeline: whisper::Decoder,
    llm_pipeline: textllm::TextGeneration,
    whisper_config: Config,
    mel_filters: Vec<f32>,
    device: candle_core::Device,
}
use tokenizers::Tokenizer;
pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}
fn parse_args(args: Vec<String>) -> (String, u16, bool, TTSType) {
    let mut port = 8000; // default port
    let mut host = "127.0.0.1".to_string(); // default host
    let mut serve = false;
    let mut tts_type = TTSType::GTTS; // default TTS type

    if args.len() > 1 {
        if args[1] == "--serve" {
            serve = true;
            for i in 2..args.len() {
                match args[i].as_str() {
                    "--port" if i + 1 < args.len() => {
                        port = args[i + 1].parse().unwrap_or(8000);
                    }
                    "--host" if i + 1 < args.len() => {
                        host = args[i + 1].clone();
                    }
                    "--tts" if i + 1 < args.len() => {
                        tts_type = match args[i + 1].as_str() {
                            "parler" => TTSType::ParlerTTS,
                            "meta" => TTSType::MetaVoice,
                            "gtts" => TTSType::GTTS,
                            _ => TTSType::GTTS,
                        };
                    }
                    _ => {}
                }
            }
        }
    }

    (host, port, serve, tts_type)
}
use tokio::sync::RwLock;
use std::io;
use std::fs::File as STD_FILE;
#[post("/<service_id>", data = "<data>")]
async fn index(
    data: Data<'_>,
    service_id: i32,
    pipelines: &State<Arc<RwLock<Pipelines>>>
) -> String {
    let mut buffer = Vec::new();
    let mut pipelines = pipelines.inner().write().await;
    match data.open((512).kibibytes()).read_to_end(&mut buffer).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to read data: {}", e);
        }
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let file_name = format!("static/service_{}_output.wav", service_id);
    let file = STD_FILE::create(&file_name).unwrap();
    //Write the data to the file
    let mut writer = WavWriter::new(file, spec)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        .unwrap();
    for sample in buffer.chunks_exact(2) {
        // Convert bytes to i16 samples (assuming incoming data is in PCM format)
        let sample = i16::from_le_bytes([sample[0], sample[1]]);
        writer
            .write_sample(sample)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
            .unwrap();
    }
    writer
        .finalize()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        .unwrap();
    let start = std::time::Instant::now();
    let (pcm_data, sample_rate) = crate::pcm_decode::pcm_decode(file_name).unwrap();
    let mel = audio::pcm_to_mel(&pipelines.whisper_config, &pcm_data, &pipelines.mel_filters);
    let mel_len = mel.len();
    let mel = candle_core::Tensor
        ::from_vec(
            mel,
            (
                1,
                pipelines.whisper_config.num_mel_bins,
                mel_len / pipelines.whisper_config.num_mel_bins,
            ),
            &pipelines.device
        )
        .unwrap();
    if sample_rate != (16000 as u32) {
        "unexpected sample rate {sample_rate}".to_string();
    }
    let decode = pipelines.whisper_pipeline.run(&mel, None).unwrap();
    let prompt = if let Some(first_segment) = decode.get(0) {
        &first_segment.dr.text
    } else {
        "sorry I didn't get that"
    };

    match pipelines.llm_pipeline.run(prompt, 100) {
        Ok(result) => {
            println!("{}s", start.elapsed().as_secs());
            result
        }
        Err(e) => { e.to_string() }
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let (host, port, serve, tts_type) = parse_args(args);
    if serve {
        let device = device(false).unwrap();
        let whisper_pipeline = whisper::load_model(device.clone());
        let whisper_config = whisper::get_config();
        let llm_pipeline = textllm::load_model(device.clone());
        let mel_bytes = match whisper_config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => { panic!("Unsupported mel filter size") }
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters
        );
        let pipelines = Arc::new(
            RwLock::new(Pipelines {
                whisper_pipeline,
                llm_pipeline,
                whisper_config,
                mel_filters,
                device,
            })
        );
        rocket
            ::build()
            .manage(pipelines)
            .mount("/", routes![index])
            .configure(rocket::Config {
                address: host.parse().unwrap(),
                port,
                ..rocket::Config::default()
            })
            .launch().await
            .unwrap();
    } else {
        cli::cli(tts_type).await.unwrap();
    }
}
