import AppKit
import PrivacyFilterCore
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = AppViewModel()
    @State private var isDropTargeted = false

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(width: 300)
                .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            imageWorkspace
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(nsColor: .windowBackgroundColor))
        }
        .alert(
            "Redaction failed",
            isPresented: Binding(
                get: { viewModel.errorMessage != nil },
                set: { if !$0 { viewModel.errorMessage = nil } }
            )
        ) {
            Button("OK") {
                viewModel.errorMessage = nil
            }
        } message: {
            Text(viewModel.errorMessage ?? "")
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("safeclipper")
                .font(.title2.weight(.semibold))

            VStack(alignment: .leading, spacing: 8) {
                Text("Image")
                    .font(.headline)
                TextField("Image path", text: $viewModel.imagePath)
                    .textFieldStyle(.roundedBorder)
                Button {
                    viewModel.loadImageFromPath()
                } label: {
                    Label("Load Path", systemImage: "folder")
                }
            }

            HStack(spacing: 8) {
                Button {
                    viewModel.selectImage()
                } label: {
                    Label("Open", systemImage: "photo")
                }

                Button {
                    viewModel.redactImage()
                } label: {
                    Label("Redact", systemImage: "eye.slash")
                }
                .disabled(!viewModel.canRedact)

                Button {
                    viewModel.saveRedactedImage()
                } label: {
                    Label("Save", systemImage: "square.and.arrow.down")
                }
                .disabled(!viewModel.canSave)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Model")
                    .font(.headline)
                TextField("Rust library", text: $viewModel.nativeLibraryPath)
                    .textFieldStyle(.roundedBorder)
                TextField("ONNX model", text: $viewModel.nativeModelPath)
                    .textFieldStyle(.roundedBorder)
                TextField("Tokenizer", text: $viewModel.nativeTokenizerPath)
                    .textFieldStyle(.roundedBorder)
                TextField("Config", text: $viewModel.nativeConfigPath)
                    .textFieldStyle(.roundedBorder)
                Picker("Provider", selection: $viewModel.nativeProvider) {
                    Text("CoreML").tag("coreml")
                    Text("CPU").tag("cpu")
                }
                .pickerStyle(.segmented)
                Picker("OCR", selection: $viewModel.nativeOCRBackend) {
                    Text("Auto").tag("auto")
                    Text("Vision").tag("vision")
                    Text("Tesseract").tag("tesseract")
                }
                .pickerStyle(.segmented)
                TextField("Sequence length", text: $viewModel.nativeSequenceLength)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Status")
                    .font(.headline)
                if viewModel.isProcessing {
                    ProgressView()
                        .controlSize(.small)
                }
                Text(viewModel.status)
                    .foregroundStyle(.secondary)
                if let sourceURL = viewModel.sourceURL {
                    Text(sourceURL.lastPathComponent)
                        .lineLimit(2)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Detected")
                    .font(.headline)
                HStack {
                    MetricView(title: "OCR tokens", value: "\(viewModel.ocrTokenCount)")
                    MetricView(title: "Masks", value: "\(viewModel.maskCount)")
                }
            }

            SpanList(spans: viewModel.spans)

            Spacer(minLength: 0)
        }
        .padding(20)
    }

    private var imageWorkspace: some View {
        ZStack {
            if let redactedImage = viewModel.redactedImage {
                ImagePreview(image: redactedImage)
            } else if let sourceImage = viewModel.sourceImage {
                ImagePreview(image: sourceImage)
            } else {
                emptyState
            }

            if viewModel.isProcessing {
                Rectangle()
                    .fill(.black.opacity(0.12))
                ProgressView(viewModel.status)
                    .padding(18)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .overlay {
            RoundedRectangle(cornerRadius: 0)
                .stroke(isDropTargeted ? Color.accentColor : .clear, lineWidth: 4)
        }
        .onDrop(of: [.fileURL, .image], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 14) {
            Image(systemName: "photo.badge.plus")
                .font(.system(size: 46))
                .foregroundStyle(.secondary)
            Text("Drop a screenshot or open an image")
                .font(.title3.weight(.medium))
            Button {
                viewModel.selectImage()
            } label: {
                Label("Open Image", systemImage: "photo")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first(where: { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }) else {
            return false
        }

        provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, _ in
            guard
                let data = item as? Data,
                let url = URL(dataRepresentation: data, relativeTo: nil)
            else {
                return
            }
            Task { @MainActor in
                viewModel.loadImage(url: url)
            }
        }

        return true
    }
}

private struct ImagePreview: View {
    let image: NSImage

    var body: some View {
        Image(nsImage: image)
            .resizable()
            .scaledToFit()
            .padding(24)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct MetricView: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.title3.weight(.semibold))
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct SpanList: View {
    let spans: [SensitiveSpan]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sensitive spans")
                .font(.headline)

            if spans.isEmpty {
                Text("No spans detected yet.")
                    .foregroundStyle(.secondary)
            } else {
                List(spans) { span in
                    VStack(alignment: .leading, spacing: 3) {
                        Text(span.label)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(span.text)
                            .lineLimit(2)
                    }
                    .padding(.vertical, 3)
                }
                .listStyle(.inset)
                .frame(minHeight: 160)
            }
        }
    }
}
