use std::fs::File;
use std::io::{ Write, Error as IoError };
use reqwest::{ Client, Error as ReqwestError };
fn amplify_mp3(
    input_path: &str,
    output_path: &str,
    amplification: f32
) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    use std::path::Path;
    // Ensure the input file exists
    if !Path::new(input_path).exists() {
        return Err(format!("Input file '{}' does not exist.", input_path).into());
    }

    // Construct the ffmpeg command to amplify the audio
    let status = Command::new("ffmpeg")
        .arg("-y") // Add this argument to force overwrite
        .arg("-i")
        .arg(input_path)
        .arg("-filter:a")
        .arg(&format!("volume={}dB", amplification))
        .arg(output_path)
        .status()
        .map_err(|e| format!("Failed to execute ffmpeg command: {}", e))?;

    // Check if the command executed successfully
    if !status.success() {
        return Err("Failed to amplify the MP3 file.".into());
    }

    Ok(())
}
pub async fn tts_to_file(
    text: String,
    filename: &str,
    language: &str,
    tld: &str
) -> Result<(), String> {
    const GOOGLE_TTS_MAX_CHARS: usize = 100;
    let len = text.len();
    let text = if len > GOOGLE_TTS_MAX_CHARS { &text[..GOOGLE_TTS_MAX_CHARS] } else { &text };
    let encoded_text = urlencoding::encode(text);
    let url = format!(
        "https://translate.google.{}/translate_tts?ie=UTF-8&q={}&tl={}&client=tw-ob",
        tld,
        encoded_text,
        language
    );
    let client = Client::new();
    let res = client
        .get(&url)
        .send().await
        .map_err(|e| format!("HTTP request error: {}", e))?
        .bytes().await
        .map_err(|e| format!("Failed to read response bytes: {}", e));
    let response = match res {
        Ok(r) => r,
        Err(e) => {
            return Err(e);
        }
    };
    //write to file
    let mut file = File::create(filename).map_err(|e| format!("File creation error: {}", e))?;
    file.write_all(&response).map_err(|e| format!("File writing error: {}", e))?;
    let output_path = format!("amplified_{}", filename);
    let amplification = 10.0; // Amplify by 10 dB

    match amplify_mp3(filename, &output_path, amplification) {
        Ok(_) => println!("Successfully amplified the MP3 file."),
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}
