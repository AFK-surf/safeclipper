#!/usr/bin/env bash
set -euo pipefail

CONFIGURATION="${1:-debug}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
APP_DIR="$SCRIPT_DIR/build/safeclipper.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
RUST_CLI_DIR="$REPO_ROOT/crates/safeclipper-cli"
RUST_LIB_PATH="$REPO_ROOT/target/release/libsafeclipper_cli.dylib"

(cd "$RUST_CLI_DIR" && cargo +1.88.0 build --release)
install_name_tool -id @rpath/libsafeclipper_cli.dylib "$RUST_LIB_PATH"

cd "$SCRIPT_DIR"
swift build -c "$CONFIGURATION" --product safeclipper

EXECUTABLE_PATH="$(swift build -c "$CONFIGURATION" --product safeclipper --show-bin-path)/safeclipper"

rm -rf "$APP_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$FRAMEWORKS_DIR"
cp "$EXECUTABLE_PATH" "$MACOS_DIR/safeclipper"
if [[ -f "$RUST_LIB_PATH" ]]; then
  cp "$RUST_LIB_PATH" "$FRAMEWORKS_DIR/libsafeclipper_cli.dylib"
fi

cat > "$CONTENTS_DIR/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>safeclipper</string>
  <key>CFBundleIdentifier</key>
  <string>local.safeclipper.app</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>safeclipper</string>
  <key>CFBundleDisplayName</key>
  <string>safeclipper</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSSupportsAutomaticGraphicsSwitching</key>
  <true/>
</dict>
</plist>
PLIST

echo "$APP_DIR"
