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
use rocket::data::{ ToByteUnit, N };
use anyhow::{ Error as E, Result, self };
use hound::{ WavSpec, WavWriter };
use candle_transformers::models::whisper::{ self as m, audio };
use tokio::sync::RwLock;
use std::io;
use std::fs::File as STD_FILE;
use std::collections::HashMap;
// Shared chat history structure
type ChatHistory = Arc<RwLock<HashMap<i32, Vec<String>>>>;
type OpenAIChatHistory = Arc<RwLock<HashMap<i32, Vec<OpenAIChatMessage>>>>;

const MAX_HISTORY_LENGTH: usize = 10;

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
    llm_pipeline: Option<textllm::TextGeneration>,
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
use serde::{ Deserialize, Serialize };
use serde_json::Value;
#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: String,
}
#[derive(Serialize, Deserialize, Clone)]
struct OpenAIChatMessage {
    role: String, // "user" or "assistant"
    content: String, // The message content
}

use std::error::Error;
async fn get_openai_response(
    messages: &[OpenAIChatMessage],
    api_key: &str,
    max_token: &usize
) -> Result<String, Box<dyn Error>> {
    let client = reqwest::Client::new();
    let url = "https://api.openai.com/v1/chat/completions";
    let request_body =
        serde_json::json!({
        "model": "gpt-4o-mini",
        "messages":messages ,
        "max_tokens": max_token
    });

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send().await?;

    // Check for non-200 HTTP status
    if !response.status().is_success() {
        return Err(format!("Failed to get response: {}", response.status()).into());
    }

    // Parse the JSON response
    let response_body: OpenAIResponse = response.json().await?;

    // Handle the case where `choices` is empty
    if let Some(choice) = response_body.choices.get(0) {
        Ok(choice.message.content.clone())
    } else {
        Err("No choices found in the response".into())
    }
}

fn parse_args(args: Vec<String>) -> (String, u16, bool, TTSType, String, usize, bool) {
    let mut port = 8000; // default port
    let mut host = "127.0.0.1".to_string(); // default host
    let mut serve = false;
    let mut tts_type = TTSType::GTTS; // default TTS type
    let mut api_key = "".to_string();
    let mut max_token = 64;
    let mut history = true; // default to keeping history

    // Display help message if --help is provided
    if args.contains(&"--help".to_string()) {
        println!(
            "Usage: [OPTIONS]

Options:
  --serve                   Start the server.
  --host <HOST>             Set the server host (default: 127.0.0.1).
  --port <PORT>             Set the server port (default: 8000).
  --tts <TYPE>              Set the TTS type. Available types: gtts (default), parler, meta.
  --api-key <KEY>           Set the API key for external services.
  --max-token <NUMBER>      Set the maximum token limit for API responses (default: 64).
  --no-history              Disable saving chat history.
  --help                    Show this help message and exit.
"
        );
        std::process::exit(0); // Exit the program after showing help
    }

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
                    "--api-key" if i + 1 < args.len() => {
                        api_key = args[i + 1].to_string();
                    }
                    "--max-token" if i + 1 < args.len() => {
                        max_token = args[i + 1].parse().unwrap_or(64);
                    }
                    "--no-history" => {
                        history = false;
                    }
                    _ => {}
                }
            }
        }
    }

    (host, port, serve, tts_type, api_key, max_token, history)
}

#[post("/<service_id>", data = "<data>")]
async fn index(
    data: Data<'_>,
    service_id: i32,
    pipelines: &State<Arc<RwLock<Pipelines>>>,
    chat_history: &State<ChatHistory>,
    open_ai_chat_history: &State<OpenAIChatHistory>,
    history: &State<bool>,
    api_key: &State<String>,
    max_token: &State<usize>
) -> String {
    let mut buffer = Vec::new();
    let mut pipelines = pipelines.inner().write().await;

    // Read data
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

    // Write the data to the file
    let mut writer = WavWriter::new(file, spec)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        .unwrap();

    for sample in buffer.chunks_exact(2) {
        let sample = i16::from_le_bytes([sample[0], sample[1]]);
        writer.write_sample(sample).unwrap();
    }

    writer.finalize().unwrap();

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

    if sample_rate != 16000 {
        return format!("unexpected sample rate {sample_rate}");
    }

    let decode = pipelines.whisper_pipeline.run(&mel, None).unwrap();
    let user_prompt = if let Some(first_segment) = decode.get(0) {
        &first_segment.dr.text
    } else {
        "sorry I didn't get that"
    };

    if pipelines.llm_pipeline.is_none() {
        if !**history {
            let chat: OpenAIChatMessage = OpenAIChatMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            };
            match get_openai_response(&[chat], api_key, max_token).await {
                Ok(response) => {
                    println!("Inference time: {:?} millis", start.elapsed().as_millis());
                    println!("Response: {}", response);
                    return response;
                }
                Err(err) => {
                    println!("Error: {}", err);
                    return err.to_string();
                }
            }
        }
        // Update chat history
        let mut history = open_ai_chat_history.inner().write().await;
        let chat = history.entry(service_id).or_insert_with(Vec::new);

        // Add the user's message to history
        chat.push(OpenAIChatMessage {
            role: "user".to_string(),
            content: user_prompt.to_string(),
        });
        // Enforce history length limit
        if chat.len() > MAX_HISTORY_LENGTH {
            chat.drain(0..chat.len() - MAX_HISTORY_LENGTH);
        }

        // Enforce history length limit
        if chat.len() > MAX_HISTORY_LENGTH {
            chat.drain(0..chat.len() - MAX_HISTORY_LENGTH);
        }
        match get_openai_response(&chat, api_key, max_token).await {
            Ok(response) => {
                // Add the assistant's response to history
                chat.push(OpenAIChatMessage {
                    role: "assistant".to_string(),
                    content: response.clone(),
                });
                println!("Inference time: {:?} millis", start.elapsed().as_millis());
                println!("Response: {}", response);
                return response;
            }
            Err(err) => {
                println!("Error: {}", err);
                return err.to_string();
            }
        }
    } else {
        if !**history {
            let prompt =
                format!("<|im_start|>user\n{}\n<|im_end|><|im_start|>assistant\n", user_prompt);
            match
                pipelines.llm_pipeline
                    .as_mut()
                    .expect("LLM pipeline is not initialized")
                    .run(&prompt, **max_token)
            {
                Ok(result) => {
                    println!("Inference time: {:?} millis", start.elapsed().as_millis());
                    println!("Response: {}", result);
                    return result.replace("\n", " ");
                }
                Err(e) => {
                    return e.to_string();
                }
            }
        }
        // Retrieve and update chat history for this service_id
        let mut history = chat_history.inner().write().await;
        let chat = history.entry(service_id).or_insert_with(Vec::new);

        // Append the user prompt
        chat.push(format!("<|im_start|>user\n{}\n<|im_end|>", user_prompt));

        // Enforce history length limit
        if chat.len() > MAX_HISTORY_LENGTH {
            chat.drain(0..chat.len() - MAX_HISTORY_LENGTH); // Remove oldest entries
        }
        let prompt = format!("{}<|im_start|>assistant\n", chat.join(""));
        match
            pipelines.llm_pipeline
                .as_mut()
                .expect("LLM pipeline is not initialized")
                .run(&prompt, **max_token)
        {
            Ok(result) => {
                println!("Inference time: {:?} millis", start.elapsed().as_millis());
                println!("Response: {}", result);

                // Append the assistant response
                chat.push(format!("{}\n<|im_end|>", result));

                // Enforce history length limit again after appending
                if chat.len() > MAX_HISTORY_LENGTH {
                    chat.drain(0..chat.len() - MAX_HISTORY_LENGTH);
                }

                result.replace("\n", " ")
            }
            Err(e) => e.to_string(),
        }
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let (host, port, serve, tts_type, api_key, max_token, history) = parse_args(args);
    if serve {
        let device = device(false).unwrap();
        let whisper_pipeline = whisper::load_model(device.clone());
        let whisper_config = whisper::get_config();
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
                llm_pipeline: if api_key.is_empty() {
                    Some(textllm::load_model(device.clone(), true))
                } else {
                    None
                },
                // llm_pipeline,
                whisper_config,
                mel_filters,
                device,
            })
        );
        // Create shared chat history
        let chat_history: ChatHistory = Arc::new(RwLock::new(HashMap::new()));
        let open_ai_chat_history: OpenAIChatHistory = Arc::new(RwLock::new(HashMap::new()));
        rocket
            ::build()
            .manage(pipelines)
            .manage(api_key)
            .manage(max_token)
            .manage(chat_history)
            .manage(open_ai_chat_history)
            .manage(history)
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
