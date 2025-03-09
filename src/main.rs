use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use gpt_sovits_rs::{GPTSovits, GPTSovitsConfig};
use serde::Deserialize;
use simple_logger::SimpleLogger;
use std::{error::Error, fs, io::Cursor, sync::mpsc, thread};
use tokio::{net::TcpListener, sync::oneshot};

struct MonoWav {
    samples: Vec<f32>,
    fs: u32,
}

impl MonoWav {
    fn to_bytes(&self) -> Result<Vec<u8>, hound::Error> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.fs,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut buffer = Cursor::new(Vec::with_capacity(
            size_of::<f32>() * self.samples.len() + 68,
            // 68 comes from hound::write::write_headers, seems to be the upperbound size of header
        ));
        {
            let mut writer = hound::WavWriter::new(&mut buffer, spec)?;
            for sample in self.samples.iter().copied() {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        Ok(buffer.into_inner())
    }
}

fn tts_infer(
    spk_name: String,
    text: String,
    gpt_sovits: &GPTSovits,
) -> Result<MonoWav, Box<dyn Error>> {
    let audio = gpt_sovits.segment_infer(spk_name.as_str(), &text, 50)?;
    let audio_size = audio.size1().unwrap() as usize;
    let mut samples = vec![0f32; audio_size];
    audio.f_copy_data(&mut samples, audio_size).unwrap();
    Ok(MonoWav { samples, fs: 32000 })
}

type RequestSender = mpsc::SyncSender<(String, oneshot::Sender<MonoWav>)>;

#[derive(Deserialize)]
struct TTSRequest {
    text: String,
}

async fn tts_handler(
    State(sender): State<RequestSender>,
    Query(TTSRequest { text }): Query<TTSRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    log::info!("inferring request: {}", text);
    let (resp_send, resp_recv) = oneshot::channel();
    let _ = sender.send((text, resp_send));
    let wav = resp_recv.await.map_err(|_| StatusCode::REQUEST_TIMEOUT)?;
    let bytes = wav
        .to_bytes()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "audio/wav".parse().unwrap());
    headers.insert("Content-Length", bytes.len().into());

    let body = Body::from(bytes);
    log::info!("request done");
    Ok((headers, body).into_response())
}

trait SampleConvert<T> {
    fn convert(self) -> T;
}

impl SampleConvert<f32> for i8 {
    fn convert(self) -> f32 {
        self as f32 / 256.0
    }
}

impl SampleConvert<f32> for i16 {
    fn convert(self) -> f32 {
        self as f32 / 32768.0
    }
}

impl SampleConvert<f32> for f32 {
    fn convert(self) -> f32 {
        self
    }
}

fn collect_f32_samples<E, T: SampleConvert<f32>, I: Iterator<Item = Result<T, E>>>(
    i: I,
) -> Result<Vec<f32>, E> {
    i.map(|v| v.map(|v| v.convert())).collect()
}

fn read_wav(path: &str) -> Result<(usize, Vec<f32>), hound::Error> {
    let reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => collect_f32_samples(reader.into_samples::<f32>()),
        (hound::SampleFormat::Int, 16) => collect_f32_samples(reader.into_samples::<i16>()),
        (hound::SampleFormat::Int, 8) => collect_f32_samples(reader.into_samples::<i8>()),
        _ => Err(hound::Error::InvalidSampleFormat),
    }
    .map(|v| (spec.sample_rate as usize, v))
}

fn init_gpt_sovits(
    spk_name: &str,
    ref_path: &str,
    ref_text: &str,
) -> Result<GPTSovits, hound::Error> {
    let gpt_config = GPTSovitsConfig::new(
        "./models/ssl_model.pt".to_string(),
        "./resource/mini-bart-g2p.pt".to_string(),
    )
    .with_chinese(
        "./resource/g2pw.pt".to_string(),
        "./models/bert_model.pt".to_string(),
    );

    let device = gpt_sovits_rs::Device::cuda_if_available();
    log::info!("device: {:?}", device);

    let mut gpt_sovits = gpt_config.build(device).unwrap();

    let (fs, ref_audio_samples) = read_wav(ref_path)?;

    log::info!("load ref {:?} done", ref_path);

    gpt_sovits
        .create_speaker(
            spk_name,
            "./models/gpt_sovits_model.pt",
            &ref_audio_samples,
            fs,
            ref_text,
        )
        .unwrap();

    log::info!("gpt init done");

    Ok(gpt_sovits)
}

#[derive(Debug, Deserialize)]
struct ServerConfig<'s> {
    level_filter: &'s str,
    ref_path: &'s str,
    ref_text: &'s str,
    spk_name: &'s str,
    host: &'s str,
    port: usize,
}

#[tokio::main]
async fn main() {
    let config_str = fs::read_to_string("config.json").unwrap();
    let ServerConfig {
        level_filter,
        ref_path,
        spk_name,
        ref_text,
        host,
        port,
    } = serde_json::from_str(&config_str).unwrap();
    let level = match level_filter.to_lowercase().as_str() {
        "error" => log::LevelFilter::Error,
        "warn" => log::LevelFilter::Warn,
        "info" => log::LevelFilter::Info,
        "debug" => log::LevelFilter::Debug,
        "trace" => log::LevelFilter::Trace,
        _ => log::LevelFilter::Off,
    };

    SimpleLogger::new().with_level(level).init().unwrap();

    log::info!("server starting...");

    let gpt_sovits = init_gpt_sovits(&spk_name, &ref_path, &ref_text).unwrap();
    let (send, recv) = mpsc::sync_channel::<(String, oneshot::Sender<MonoWav>)>(0);
    let spk_name = spk_name.to_owned(); // tts thread might outlive main thread, we can't borrow

    // start tts thread
    thread::spawn(move || {
        for (text, respond_to) in recv {
            let result = tts_infer(spk_name.clone(), text.clone(), &gpt_sovits).and_then(|wav| {
                respond_to
                    .send(wav)
                    .map_err(|_| format!("Failed to send infer result of {}", text).into())
            });
            if let Err(err) = result {
                log::error!("{}", err.to_string());
            }
        }
    });

    // start axum server
    let app = Router::new()
        .route("/", get(|| async { "Server is on" }))
        .route("/tts", get(tts_handler))
        .with_state(send);

    let addr = format!("{}:{}", host, port);
    log::info!("serving at {:?}", addr);
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
