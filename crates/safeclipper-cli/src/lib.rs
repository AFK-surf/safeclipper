use std::{
    collections::HashMap,
    ffi::{c_char, CStr, CString},
    fs,
    ops::Range,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use image::Rgba;
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use tokenizers::{Encoding, Tokenizer};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Provider::Coreml)]
    provider: Provider,
    #[arg(long)]
    text: Option<String>,
    #[arg(long)]
    file: Option<PathBuf>,
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long)]
    output_image: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = OcrBackend::Auto)]
    ocr_backend: OcrBackend,
    #[arg(long, default_value = "tesseract")]
    tesseract_bin: PathBuf,
    #[arg(long, default_value = "eng")]
    tesseract_lang: String,
    #[arg(long, default_value_t = 6)]
    tesseract_psm: u8,
    #[arg(long, default_value_t = 2)]
    mask_padding: u32,
    #[arg(long, default_value_t = 1)]
    intra_threads: usize,
    #[arg(long)]
    sequence_length: Option<usize>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    output: OutputFormat,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Provider {
    Cpu,
    Coreml,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Json,
    Text,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OcrBackend {
    Auto,
    Vision,
    Tesseract,
}

impl OcrBackend {
    fn resolved(self) -> Result<ResolvedOcrBackend> {
        match self {
            OcrBackend::Auto => Ok(default_ocr_backend()),
            OcrBackend::Vision => {
                if cfg!(target_os = "macos") {
                    Ok(ResolvedOcrBackend::Vision)
                } else {
                    bail!("--ocr-backend vision is only available on macOS")
                }
            }
            OcrBackend::Tesseract => Ok(ResolvedOcrBackend::Tesseract),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ResolvedOcrBackend {
    Vision,
    Tesseract,
}

impl ResolvedOcrBackend {
    fn as_str(self) -> &'static str {
        match self {
            ResolvedOcrBackend::Vision => "vision",
            ResolvedOcrBackend::Tesseract => "tesseract",
        }
    }
}

fn default_ocr_backend() -> ResolvedOcrBackend {
    if cfg!(target_os = "macos") {
        ResolvedOcrBackend::Vision
    } else {
        ResolvedOcrBackend::Tesseract
    }
}

#[derive(Debug, Deserialize)]
struct RedactImageRequest {
    model: PathBuf,
    tokenizer: PathBuf,
    config: Option<PathBuf>,
    image: PathBuf,
    output_image: PathBuf,
    provider: Option<String>,
    ocr_backend: Option<String>,
    tesseract_bin: Option<PathBuf>,
    tesseract_lang: Option<String>,
    tesseract_psm: Option<u8>,
    mask_padding: Option<u32>,
    intra_threads: Option<usize>,
    sequence_length: Option<usize>,
}

#[derive(Debug, Serialize)]
struct Response {
    schema_version: u8,
    summary: Summary,
    text: String,
    detected_spans: Vec<SensitiveSpan>,
    redacted_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_redaction: Option<ImageRedaction>,
}

#[derive(Debug, Serialize)]
struct Summary {
    output_mode: &'static str,
    input_mode: &'static str,
    span_count: usize,
    by_label: HashMap<String, usize>,
    latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct SensitiveSpan {
    label: String,
    start: usize,
    end: usize,
    text: String,
    placeholder: String,
}

#[derive(Debug, Serialize)]
struct ImageRedaction {
    input_path: String,
    output_path: Option<String>,
    ocr_backend: &'static str,
    ocr_token_count: usize,
    mask_count: usize,
}

#[derive(Debug)]
struct InputDocument {
    mode: InputMode,
    text: String,
    image: Option<ImageInput>,
}

#[derive(Debug)]
struct ImageInput {
    path: PathBuf,
    ocr: OcrDocument,
    ocr_backend: ResolvedOcrBackend,
}

#[derive(Debug)]
struct OcrDocument {
    tokens: Vec<OcrToken>,
}

#[derive(Debug)]
struct OcrToken {
    range: Range<usize>,
    bounding_box: NormalizedRect,
}

#[derive(Clone, Copy, Debug)]
struct NormalizedRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

#[derive(Clone, Copy, Debug)]
enum InputMode {
    Text,
    File,
    Image,
}

impl InputMode {
    fn as_str(self) -> &'static str {
        match self {
            InputMode::Text => "text",
            InputMode::File => "file",
            InputMode::Image => "image",
        }
    }
}

pub fn cli_main() -> Result<()> {
    let args = Args::parse();
    let response = run_redaction(&args)?;

    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&response)?),
        OutputFormat::Text => println!("{}", response.redacted_text),
    }
    Ok(())
}

fn run_redaction(args: &Args) -> Result<Response> {
    let input = load_input_document(&args)?;

    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|err| anyhow::anyhow!("failed to load tokenizer: {err}"))?;
    let encoding = tokenizer
        .encode(input.text.as_str(), true)
        .map_err(|err| anyhow::anyhow!("failed to tokenize input: {err}"))?;

    let id2label = load_id2label(args.config.as_ref())?;
    let pad_token_id = load_pad_token_id(args.config.as_ref())?.unwrap_or(199999);
    let mut session = build_session(args)?;

    let started = Instant::now();
    let logits = run_model(&mut session, &encoding, args.sequence_length, pad_token_id)?;
    let spans = decode_argmax_spans(&input.text, &encoding, &logits, &id2label);
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut by_label = HashMap::new();
    for span in &spans {
        *by_label.entry(span.label.clone()).or_insert(0) += 1;
    }
    let redacted_text = redact_text(&input.text, &spans);
    let image_redaction = match input.image.as_ref() {
        Some(image) => Some(redact_image_if_requested(
            image,
            &spans,
            args.output_image.as_deref(),
            args.mask_padding,
        )?),
        None => None,
    };

    Ok(Response {
        schema_version: 1,
        summary: Summary {
            output_mode: match args.output {
                OutputFormat::Json => "json",
                OutputFormat::Text => "text",
            },
            input_mode: input.mode.as_str(),
            span_count: spans.len(),
            by_label,
            latency_ms,
        },
        text: input.text,
        detected_spans: spans,
        redacted_text,
        image_redaction,
    })
}

#[no_mangle]
pub extern "C" fn safeclipper_redact_image_json(
    request_json: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut c_char {
    ffi_boundary(error_out, || {
        if request_json.is_null() {
            bail!("request_json is null");
        }
        let request_json = unsafe { CStr::from_ptr(request_json) }
            .to_str()
            .context("request_json is not valid UTF-8")?;
        let request: RedactImageRequest = serde_json::from_str(request_json)
            .with_context(|| format!("failed to decode redaction request: {request_json}"))?;
        let args = args_from_redact_image_request(request)?;
        let response = run_redaction(&args)?;
        serde_json::to_string(&response).context("failed to encode redaction response")
    })
}

#[no_mangle]
pub extern "C" fn safeclipper_free_string(value: *mut c_char) {
    if !value.is_null() {
        unsafe {
            let _ = CString::from_raw(value);
        }
    }
}

fn ffi_boundary<F>(error_out: *mut *mut c_char, operation: F) -> *mut c_char
where
    F: FnOnce() -> Result<String>,
{
    match operation() {
        Ok(value) => string_to_c(value),
        Err(error) => {
            if !error_out.is_null() {
                unsafe {
                    *error_out = string_to_c(error.to_string());
                }
            }
            std::ptr::null_mut()
        }
    }
}

fn string_to_c(value: String) -> *mut c_char {
    let sanitized = value.replace('\0', " ");
    CString::new(sanitized)
        .expect("nul bytes were removed")
        .into_raw()
}

fn args_from_redact_image_request(request: RedactImageRequest) -> Result<Args> {
    Ok(Args {
        model: request.model,
        tokenizer: request.tokenizer,
        config: request.config,
        provider: parse_provider(request.provider.as_deref().unwrap_or("coreml"))?,
        text: None,
        file: None,
        image: Some(request.image),
        output_image: Some(request.output_image),
        ocr_backend: parse_ocr_backend(request.ocr_backend.as_deref().unwrap_or("auto"))?,
        tesseract_bin: request
            .tesseract_bin
            .unwrap_or_else(|| PathBuf::from("tesseract")),
        tesseract_lang: request.tesseract_lang.unwrap_or_else(|| "eng".to_string()),
        tesseract_psm: request.tesseract_psm.unwrap_or(6),
        mask_padding: request.mask_padding.unwrap_or(2),
        intra_threads: request.intra_threads.unwrap_or(1),
        sequence_length: request.sequence_length,
        output: OutputFormat::Json,
    })
}

fn parse_provider(value: &str) -> Result<Provider> {
    Provider::from_str(value, true)
        .map_err(|message| anyhow::anyhow!("invalid provider {value}: {message}"))
}

fn parse_ocr_backend(value: &str) -> Result<OcrBackend> {
    OcrBackend::from_str(value, true)
        .map_err(|message| anyhow::anyhow!("invalid OCR backend {value}: {message}"))
}

fn load_input_document(args: &Args) -> Result<InputDocument> {
    let input_count = usize::from(args.text.is_some())
        + usize::from(args.file.is_some())
        + usize::from(args.image.is_some());
    if input_count != 1 {
        bail!("use exactly one input: --text, --file, or --image");
    }
    if args.output_image.is_some() && args.image.is_none() {
        bail!("--output-image requires --image");
    }

    if let Some(text) = &args.text {
        return Ok(InputDocument {
            mode: InputMode::Text,
            text: text.clone(),
            image: None,
        });
    }

    if let Some(path) = &args.file {
        return Ok(InputDocument {
            mode: InputMode::File,
            text: fs::read_to_string(path)
                .with_context(|| format!("failed to read {}", path.display()))?,
            image: None,
        });
    }

    let image_path = args.image.as_ref().expect("image checked above");
    let ocr_backend = args.ocr_backend.resolved()?;
    let (text, ocr) = recognize_image_text(image_path, ocr_backend, args)
        .with_context(|| format!("failed to OCR image {}", image_path.display()))?;
    Ok(InputDocument {
        mode: InputMode::Image,
        text,
        image: Some(ImageInput {
            path: image_path.clone(),
            ocr,
            ocr_backend,
        }),
    })
}

#[derive(Debug, Deserialize)]
struct RawOcrDocument {
    lines: Vec<RawOcrLine>,
}

#[derive(Debug, Deserialize)]
struct RawOcrLine {
    text: String,
    tokens: Vec<RawOcrToken>,
}

#[derive(Debug, Deserialize)]
struct RawOcrToken {
    text: String,
    bounding_box: [f32; 4],
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn safeclipper_vision_ocr(
        image_path: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
    ) -> *mut std::os::raw::c_char;
    fn safeclipper_free_c_string(value: *mut std::os::raw::c_char);
}

fn recognize_image_text(
    image_path: &Path,
    backend: ResolvedOcrBackend,
    args: &Args,
) -> Result<(String, OcrDocument)> {
    match backend {
        ResolvedOcrBackend::Vision => recognize_image_text_with_vision(image_path),
        ResolvedOcrBackend::Tesseract => recognize_image_text_with_tesseract(image_path, args),
    }
}

#[cfg(target_os = "macos")]
fn recognize_image_text_with_vision(image_path: &Path) -> Result<(String, OcrDocument)> {
    use std::ffi::{CStr, CString};

    let path = CString::new(image_path.to_string_lossy().as_bytes())
        .context("image path contains an interior NUL byte")?;
    let mut error_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
    let result_ptr = unsafe { safeclipper_vision_ocr(path.as_ptr(), &mut error_ptr) };

    if result_ptr.is_null() {
        let message = if error_ptr.is_null() {
            "Vision OCR failed".to_string()
        } else {
            let error = unsafe { CStr::from_ptr(error_ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { safeclipper_free_c_string(error_ptr) };
            error
        };
        bail!("{message}");
    }

    let json = unsafe { CStr::from_ptr(result_ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { safeclipper_free_c_string(result_ptr) };

    let raw: RawOcrDocument = serde_json::from_str(&json)
        .with_context(|| format!("failed to decode Vision OCR response: {json}"))?;
    Ok(build_ocr_document(raw))
}

#[cfg(not(target_os = "macos"))]
fn recognize_image_text_with_vision(_image_path: &Path) -> Result<(String, OcrDocument)> {
    bail!("Apple Vision OCR is only available on macOS")
}

fn recognize_image_text_with_tesseract(image_path: &Path, args: &Args) -> Result<(String, OcrDocument)> {
    let output = Command::new(&args.tesseract_bin)
        .arg(image_path)
        .arg("stdout")
        .arg("-l")
        .arg(&args.tesseract_lang)
        .arg("--psm")
        .arg(args.tesseract_psm.to_string())
        .arg("tsv")
        .output()
        .with_context(|| {
            format!(
                "failed to run Tesseract binary {}",
                args.tesseract_bin.display()
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "Tesseract exited with status {}. {}",
            output.status,
            stderr.trim()
        );
    }

    let tsv = String::from_utf8(output.stdout).context("Tesseract TSV output was not UTF-8")?;
    let (width, height) = image::image_dimensions(image_path)
        .with_context(|| format!("failed to read image dimensions {}", image_path.display()))?;
    build_ocr_document_from_tesseract_tsv(&tsv, width, height)
}

fn build_ocr_document(raw: RawOcrDocument) -> (String, OcrDocument) {
    let mut text = String::new();
    let mut tokens = Vec::new();

    for line in raw.lines {
        if !text.is_empty() {
            text.push('\n');
        }

        let line_start = text.len();
        text.push_str(&line.text);
        let mut search_start = 0;

        for token in line.tokens {
            let Some(relative_start) = line.text[search_start..].find(&token.text) else {
                continue;
            };
            let token_start_in_line = search_start + relative_start;
            let token_end_in_line = token_start_in_line + token.text.len();
            search_start = token_end_in_line;

            tokens.push(OcrToken {
                range: (line_start + token_start_in_line)..(line_start + token_end_in_line),
                bounding_box: NormalizedRect {
                    x: token.bounding_box[0],
                    y: token.bounding_box[1],
                    width: token.bounding_box[2],
                    height: token.bounding_box[3],
                },
            });
        }
    }

    (text, OcrDocument { tokens })
}

#[derive(Debug)]
struct TesseractLine {
    key: (String, String, String),
    words: Vec<TesseractWord>,
}

#[derive(Debug)]
struct TesseractWord {
    text: String,
    bounding_box: NormalizedRect,
}

fn build_ocr_document_from_tesseract_tsv(
    tsv: &str,
    image_width: u32,
    image_height: u32,
) -> Result<(String, OcrDocument)> {
    let mut lines: Vec<TesseractLine> = Vec::new();

    for row in tsv.lines().skip(1) {
        let columns: Vec<&str> = row.split('\t').collect();
        if columns.len() < 12 || columns[0] != "5" {
            continue;
        }

        let text = columns[11..].join("\t");
        let text = text.trim().to_string();
        if text.is_empty() {
            continue;
        }

        let left = parse_tesseract_u32(columns[6], "left")?;
        let top = parse_tesseract_u32(columns[7], "top")?;
        let width = parse_tesseract_u32(columns[8], "width")?;
        let height = parse_tesseract_u32(columns[9], "height")?;
        if width == 0 || height == 0 || image_width == 0 || image_height == 0 {
            continue;
        }

        let key = (
            columns[2].to_string(),
            columns[3].to_string(),
            columns[4].to_string(),
        );
        let word = TesseractWord {
            text,
            bounding_box: NormalizedRect {
                x: left as f32 / image_width as f32,
                y: 1.0 - ((top + height) as f32 / image_height as f32),
                width: width as f32 / image_width as f32,
                height: height as f32 / image_height as f32,
            },
        };

        match lines.last_mut() {
            Some(line) if line.key == key => line.words.push(word),
            _ => lines.push(TesseractLine {
                key,
                words: vec![word],
            }),
        }
    }

    let mut text = String::new();
    let mut tokens = Vec::new();

    for line in lines {
        if !text.is_empty() {
            text.push('\n');
        }

        for (index, word) in line.words.into_iter().enumerate() {
            if index > 0 {
                text.push(' ');
            }
            let start = text.len();
            text.push_str(&word.text);
            let end = text.len();
            tokens.push(OcrToken {
                range: start..end,
                bounding_box: word.bounding_box,
            });
        }
    }

    Ok((text, OcrDocument { tokens }))
}

fn parse_tesseract_u32(value: &str, field: &str) -> Result<u32> {
    value
        .parse()
        .with_context(|| format!("invalid Tesseract TSV {field}: {value}"))
}

fn redact_image_if_requested(
    image: &ImageInput,
    spans: &[SensitiveSpan],
    output_path: Option<&Path>,
    padding: u32,
) -> Result<ImageRedaction> {
    let masks = masks_for_spans(&image.ocr.tokens, spans);
    if let Some(output_path) = output_path {
        write_redacted_image(&image.path, output_path, &masks, padding)?;
    }

    Ok(ImageRedaction {
        input_path: image.path.display().to_string(),
        output_path: output_path.map(|path| path.display().to_string()),
        ocr_backend: image.ocr_backend.as_str(),
        ocr_token_count: image.ocr.tokens.len(),
        mask_count: masks.len(),
    })
}

fn masks_for_spans(tokens: &[OcrToken], spans: &[SensitiveSpan]) -> Vec<NormalizedRect> {
    let mut masks = Vec::new();
    for span in spans {
        for token in tokens {
            if token.range.start < span.end && token.range.end > span.start {
                masks.push(token.bounding_box);
            }
        }
    }
    masks
}

fn write_redacted_image(
    input_path: &Path,
    output_path: &Path,
    masks: &[NormalizedRect],
    padding: u32,
) -> Result<()> {
    let mut image = image::open(input_path)
        .with_context(|| format!("failed to open image {}", input_path.display()))?
        .to_rgba8();
    let (width, height) = image.dimensions();

    for mask in masks {
        let (x0, y0, x1, y1) = normalized_rect_to_pixels(*mask, width, height, padding);
        for y in y0..y1 {
            for x in x0..x1 {
                image.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
    }

    image
        .save(output_path)
        .with_context(|| format!("failed to save redacted image {}", output_path.display()))
}

fn normalized_rect_to_pixels(
    rect: NormalizedRect,
    image_width: u32,
    image_height: u32,
    padding: u32,
) -> (u32, u32, u32, u32) {
    let x0 = (rect.x * image_width as f32).floor().max(0.0) as u32;
    let y0 = ((1.0 - rect.y - rect.height) * image_height as f32).floor().max(0.0) as u32;
    let x1 = ((rect.x + rect.width) * image_width as f32)
        .ceil()
        .clamp(0.0, image_width as f32) as u32;
    let y1 = ((1.0 - rect.y) * image_height as f32)
        .ceil()
        .clamp(0.0, image_height as f32) as u32;

    (
        x0.saturating_sub(padding),
        y0.saturating_sub(padding),
        x1.saturating_add(padding).min(image_width),
        y1.saturating_add(padding).min(image_height),
    )
}

fn build_session(args: &Args) -> Result<Session> {
    let mut builder = Session::builder().map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3).map_err(ort_error)?
        .with_intra_threads(args.intra_threads).map_err(ort_error)?
        .with_memory_pattern(false).map_err(ort_error)?;
    if let Some(sequence_length) = args.sequence_length {
        builder = builder
            .with_dimension_override("batch_size", 1).map_err(ort_error)?
            .with_dimension_override("sequence_length", sequence_length.try_into()?).map_err(ort_error)?
            .with_dimension_override("total_sequence_length", sequence_length.try_into()?).map_err(ort_error)?;
    }

    builder = match args.provider {
        Provider::Cpu => builder.with_execution_providers([
            ep::CPU::default().build(),
        ]).map_err(ort_error)?,
        Provider::Coreml => builder.with_execution_providers([
            ep::CoreML::default()
                .with_model_format(ep::coreml::ModelFormat::MLProgram)
                .with_compute_units(ep::coreml::ComputeUnits::All)
                .with_subgraphs(true)
                .with_low_precision_accumulation_on_gpu(true)
                .build(),
            ep::CPU::default().build(),
        ]).map_err(ort_error)?,
    };

    builder
        .commit_from_file(&args.model)
        .map_err(ort_error)
        .with_context(|| format!("failed to load ONNX model {}", args.model.display()))
}

fn run_model(
    session: &mut Session,
    encoding: &Encoding,
    sequence_length: Option<usize>,
    pad_token_id: i64,
) -> Result<Vec<Vec<f32>>> {
    let original_len = encoding.get_ids().len();
    let target_len = sequence_length.unwrap_or(original_len);
    if original_len > target_len {
        bail!("input has {original_len} tokens, exceeding fixed sequence length {target_len}");
    }

    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|id| i64::from(*id)).collect();
    let mut mask: Vec<i64> = vec![1; ids.len()];
    ids.resize(target_len, pad_token_id);
    mask.resize(target_len, 0);

    let outputs = session.run(ort::inputs![
        "input_ids" => TensorRef::from_array_view(([1usize, target_len], ids.as_slice())).map_err(ort_error)?,
        "attention_mask" => TensorRef::from_array_view(([1usize, target_len], mask.as_slice())).map_err(ort_error)?,
    ]).map_err(ort_error)?;

    let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
    if shape.len() != 3 || shape[0] != 1 {
        bail!("unexpected logits shape: {shape:?}");
    }

    let tokens = usize::try_from(shape[1]).context("invalid token dimension")?;
    let labels = usize::try_from(shape[2]).context("invalid label dimension")?;
    let flat = logits;
    let mut rows = Vec::with_capacity(tokens);
    for token_index in 0..tokens {
        let start = token_index * labels;
        rows.push(flat[start..start + labels].to_vec());
    }
    Ok(rows)
}

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}

fn decode_argmax_spans(
    text: &str,
    encoding: &Encoding,
    logits: &[Vec<f32>],
    id2label: &HashMap<usize, String>,
) -> Vec<SensitiveSpan> {
    let offsets = encoding.get_offsets();
    let mut spans = Vec::new();
    let mut active: Option<(String, usize, usize)> = None;

    for (index, row) in logits.iter().enumerate() {
        let Some(&(start, end)) = offsets.get(index) else {
            continue;
        };
        if start == end {
            continue;
        }

        let label_id = argmax(row);
        let raw_label = id2label
            .get(&label_id)
            .map(String::as_str)
            .unwrap_or("O");
        let Some((prefix, label)) = parse_bioes_label(raw_label) else {
            close_active(text, &mut spans, &mut active);
            continue;
        };

        match prefix {
            "S" => {
                close_active(text, &mut spans, &mut active);
                push_span(text, &mut spans, label, start, end);
            }
            "B" => {
                close_active(text, &mut spans, &mut active);
                active = Some((label.to_string(), start, end));
            }
            "I" | "E" => {
                if let Some((active_label, _, active_end)) = active.as_mut() {
                    if active_label == label {
                        *active_end = end;
                    } else {
                        close_active(text, &mut spans, &mut active);
                        active = Some((label.to_string(), start, end));
                    }
                } else {
                    active = Some((label.to_string(), start, end));
                }
                if prefix == "E" {
                    close_active(text, &mut spans, &mut active);
                }
            }
            _ => close_active(text, &mut spans, &mut active),
        }
    }

    close_active(text, &mut spans, &mut active);
    spans
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn parse_bioes_label(raw: &str) -> Option<(&str, &str)> {
    if raw == "O" || raw == "o" {
        return None;
    }
    raw.split_once('-')
        .or_else(|| raw.split_once('_'))
        .and_then(|(prefix, label)| {
            if prefix.eq_ignore_ascii_case("B") {
                Some(("B", label))
            } else if prefix.eq_ignore_ascii_case("I") {
                Some(("I", label))
            } else if prefix.eq_ignore_ascii_case("E") {
                Some(("E", label))
            } else if prefix.eq_ignore_ascii_case("S") {
                Some(("S", label))
            } else {
                None
            }
        })
}

fn close_active(text: &str, spans: &mut Vec<SensitiveSpan>, active: &mut Option<(String, usize, usize)>) {
    if let Some((label, start, end)) = active.take() {
        push_span(text, spans, &label, start, end);
    }
}

fn push_span(text: &str, spans: &mut Vec<SensitiveSpan>, label: &str, start: usize, end: usize) {
    if start >= end || end > text.len() {
        return;
    }
    let value = text[start..end].to_string();
    spans.push(SensitiveSpan {
        label: label.to_string(),
        start,
        end,
        text: value,
        placeholder: format!("<{}>", label.to_ascii_uppercase()),
    });
}

fn redact_text(text: &str, spans: &[SensitiveSpan]) -> String {
    let mut result = String::new();
    let mut cursor = 0;
    for span in spans {
        if span.start < cursor || span.end > text.len() {
            continue;
        }
        result.push_str(&text[cursor..span.start]);
        result.push_str(&span.placeholder);
        cursor = span.end;
    }
    result.push_str(&text[cursor..]);
    result
}

fn load_id2label(config: Option<&PathBuf>) -> Result<HashMap<usize, String>> {
    let Some(config) = config else {
        return Ok(HashMap::new());
    };
    let value: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(config).with_context(|| format!("failed to read {}", config.display()))?,
    )?;
    let mut map = HashMap::new();
    if let Some(labels) = value.get("id2label").and_then(|v| v.as_object()) {
        for (key, value) in labels {
            if let (Ok(index), Some(label)) = (key.parse::<usize>(), value.as_str()) {
                map.insert(index, label.to_string());
            }
        }
    }
    Ok(map)
}

fn load_pad_token_id(config: Option<&PathBuf>) -> Result<Option<i64>> {
    let Some(config) = config else {
        return Ok(None);
    };
    let value: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(config).with_context(|| format!("failed to read {}", config.display()))?,
    )?;
    Ok(value.get("pad_token_id").and_then(|v| v.as_i64()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_text_replaces_spans_with_placeholders() {
        let text = "Alice email alice@example.com";
        let spans = vec![
            SensitiveSpan {
                label: "private_person".to_string(),
                start: 0,
                end: 5,
                text: "Alice".to_string(),
                placeholder: "<PRIVATE_PERSON>".to_string(),
            },
            SensitiveSpan {
                label: "private_email".to_string(),
                start: 12,
                end: 29,
                text: "alice@example.com".to_string(),
                placeholder: "<PRIVATE_EMAIL>".to_string(),
            },
        ];

        assert_eq!(
            redact_text(text, &spans),
            "<PRIVATE_PERSON> email <PRIVATE_EMAIL>"
        );
    }

    #[test]
    fn redact_text_skips_overlapping_spans() {
        let text = "token abc123";
        let spans = vec![
            SensitiveSpan {
                label: "secret".to_string(),
                start: 6,
                end: 12,
                text: "abc123".to_string(),
                placeholder: "<SECRET>".to_string(),
            },
            SensitiveSpan {
                label: "secret".to_string(),
                start: 8,
                end: 12,
                text: "c123".to_string(),
                placeholder: "<SECRET>".to_string(),
            },
        ];

        assert_eq!(redact_text(text, &spans), "token <SECRET>");
    }

    #[test]
    fn masks_for_spans_selects_overlapping_ocr_tokens() {
        let tokens = vec![
            OcrToken {
                range: 0..5,
                bounding_box: NormalizedRect {
                    x: 0.1,
                    y: 0.2,
                    width: 0.1,
                    height: 0.05,
                },
            },
            OcrToken {
                range: 6..23,
                bounding_box: NormalizedRect {
                    x: 0.3,
                    y: 0.2,
                    width: 0.3,
                    height: 0.05,
                },
            },
        ];
        let spans = vec![SensitiveSpan {
            label: "private_email".to_string(),
            start: 10,
            end: 23,
            text: "example.com".to_string(),
            placeholder: "<PRIVATE_EMAIL>".to_string(),
        }];

        let masks = masks_for_spans(&tokens, &spans);

        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0].x, 0.3);
    }

    #[test]
    fn normalized_rect_to_pixels_flips_vision_y_axis() {
        let rect = NormalizedRect {
            x: 0.25,
            y: 0.25,
            width: 0.5,
            height: 0.25,
        };

        assert_eq!(normalized_rect_to_pixels(rect, 200, 100, 0), (50, 50, 150, 75));
        assert_eq!(normalized_rect_to_pixels(rect, 200, 100, 4), (46, 46, 154, 79));
    }

    #[test]
    fn build_ocr_document_from_tesseract_tsv_groups_words_by_line() {
        let tsv = "\
level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n\
5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t95\tAlice\n\
5\t1\t1\t1\t1\t2\t50\t20\t40\t10\t94\tZhang\n\
5\t1\t1\t1\t2\t1\t10\t50\t60\t10\t90\talice@example.com\n";

        let (text, document) = build_ocr_document_from_tesseract_tsv(tsv, 200, 100).unwrap();

        assert_eq!(text, "Alice Zhang\nalice@example.com");
        assert_eq!(document.tokens.len(), 3);
        assert_eq!(document.tokens[0].range, 0..5);
        assert_eq!(document.tokens[1].range, 6..11);
        assert_eq!(document.tokens[2].range, 12..29);
        assert_eq!(normalized_rect_to_pixels(document.tokens[0].bounding_box, 200, 100, 0), (10, 20, 40, 31));
    }
}
