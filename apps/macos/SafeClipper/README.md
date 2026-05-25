# safeclipper macOS

safeclipper for macOS is a local screenshot redaction app. The SwiftUI app links the bundled Rust dylib for OCR, sensitive span detection, and image redaction.

The product is designed for proactive-agent workflows where screenshots and app context may be collected frequently. Run the screenshot through safeclipper first, then pass the redacted image to the agent or cloud service.

The Rust CLI can also run the same image chain directly with `--image` and `--output-image`. On macOS it supports both `--ocr-backend vision` and `--ocr-backend tesseract`; on non-macOS platforms it supports `--ocr-backend tesseract`.

The app does not spawn the CLI for redaction. `build-app.sh` links `safeclipper` against `libsafeclipper_cli.dylib` and copies that dylib into `Contents/Frameworks`.

## Run

From the repository root:

```bash
./models/download-openai-privacy-filter-q4.sh
cd apps/macos/SafeClipper
./build-app.sh debug
open build/safeclipper.app
```

To preload the fixture image:

```bash
open build/safeclipper.app --args "$OLDPWD/fixtures/privacy-screenshot.png"
```
