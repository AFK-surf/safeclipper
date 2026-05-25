import AppKit
import Foundation
import PrivacyFilterCore
import SwiftUI
import UniformTypeIdentifiers

enum NativeRuntimeLocator {
    static func defaultLibraryPath() -> String {
        if let explicit = ProcessInfo.processInfo.environment["SAFECLIPER_RUST_LIBRARY"], !explicit.isEmpty {
            return explicit
        }
        if let bundled = Bundle.main.privateFrameworksURL?.appendingPathComponent("libsafeclipper_cli.dylib"),
           FileManager.default.fileExists(atPath: bundled.path) {
            return bundled.path
        }
        if let repositoryRoot = defaultRepositoryRoot() {
            let library = repositoryRoot.appendingPathComponent("target/release/libsafeclipper_cli.dylib")
            if FileManager.default.fileExists(atPath: library.path) {
                return library.path
            }
        }
        return ""
    }

    static func defaultModelPath() -> String {
        if let explicit = ProcessInfo.processInfo.environment["OPF_NATIVE_MODEL"], !explicit.isEmpty {
            return explicit
        }
        return defaultModelFile(named: "model_q4_embedded.onnx")
    }

    static func defaultTokenizerPath() -> String {
        if let explicit = ProcessInfo.processInfo.environment["OPF_NATIVE_TOKENIZER"], !explicit.isEmpty {
            return explicit
        }
        return defaultModelFile(named: "tokenizer.json")
    }

    static func defaultConfigPath() -> String {
        if let explicit = ProcessInfo.processInfo.environment["OPF_NATIVE_CONFIG"], !explicit.isEmpty {
            return explicit
        }
        return defaultModelFile(named: "config.json")
    }

    private static func defaultModelFile(named name: String) -> String {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent(name),
           FileManager.default.fileExists(atPath: bundled.path) {
            return bundled.path
        }
        guard let repositoryRoot = defaultRepositoryRoot() else {
            return ""
        }
        let base = repositoryRoot.appendingPathComponent("models/openai-privacy-filter")
        let path = name.hasSuffix(".onnx")
            ? base.appendingPathComponent("onnx/\(name)")
            : base.appendingPathComponent(name)
        return path.path
    }

    private static func defaultRepositoryRoot() -> URL? {
        let startURLs = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
            Bundle.main.bundleURL
        ]

        for startURL in startURLs {
            if let root = searchParents(from: startURL) {
                return root
            }
        }

        return nil
    }

    private static func searchParents(from startURL: URL) -> URL? {
        var url = startURL
        if !url.hasDirectoryPath {
            url.deleteLastPathComponent()
        }

        for _ in 0..<12 {
            let cargoManifest = url.appendingPathComponent("crates/safeclipper-cli/Cargo.toml")
            if FileManager.default.fileExists(atPath: cargoManifest.path) {
                return url
            }

            let previous = url
            url.deleteLastPathComponent()
            if previous == url {
                break
            }
        }

        return nil
    }
}

@MainActor
final class AppViewModel: ObservableObject {
    @Published var sourceURL: URL?
    @Published var sourceImage: NSImage?
    @Published var redactedImage: NSImage?
    @Published var redactedImageURL: URL?
    @Published var imagePath = ""
    @Published var spans: [SensitiveSpan] = []
    @Published var ocrTokenCount = 0
    @Published var maskCount = 0
    @Published var status = "Select an image"
    @Published var errorMessage: String?
    @Published var isProcessing = false
    @Published var nativeLibraryPath = NativeRuntimeLocator.defaultLibraryPath()
    @Published var nativeModelPath = NativeRuntimeLocator.defaultModelPath()
    @Published var nativeTokenizerPath = NativeRuntimeLocator.defaultTokenizerPath()
    @Published var nativeConfigPath = NativeRuntimeLocator.defaultConfigPath()
    @Published var nativeProvider = ProcessInfo.processInfo.environment["OPF_NATIVE_PROVIDER"] ?? "coreml"
    @Published var nativeSequenceLength = ProcessInfo.processInfo.environment["OPF_NATIVE_SEQUENCE_LENGTH"] ?? "512"
    @Published var nativeOCRBackend = ProcessInfo.processInfo.environment["SAFECLIPER_OCR_BACKEND"] ?? "auto"

    init() {
        guard let initialPath = CommandLine.arguments.dropFirst().first, !initialPath.isEmpty else {
            return
        }
        loadImage(url: URL(fileURLWithPath: NSString(string: initialPath).expandingTildeInPath))
    }

    var canRedact: Bool {
        sourceImage != nil && sourceURL != nil && !isProcessing
    }

    var canSave: Bool {
        redactedImage != nil && !isProcessing
    }

    func selectImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image]

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        loadImage(url: url)
    }

    func loadImage(url: URL) {
        guard let image = NSImage(contentsOf: url) else {
            errorMessage = "Could not open \(url.lastPathComponent)."
            return
        }

        sourceURL = url
        imagePath = url.path
        sourceImage = image
        redactedImage = nil
        redactedImageURL = nil
        spans = []
        ocrTokenCount = 0
        maskCount = 0
        errorMessage = nil
        status = "Ready"
    }

    func loadImageFromPath() {
        let expanded = NSString(string: imagePath).expandingTildeInPath
        loadImage(url: URL(fileURLWithPath: expanded))
    }

    func redactImage() {
        guard let sourceURL else {
            return
        }

        isProcessing = true
        errorMessage = nil
        status = "Running Rust library"

        Task {
            do {
                let client = OPFNativeRuntimeClient(
                    modelPath: nativeModelPath,
                    tokenizerPath: nativeTokenizerPath,
                    configPath: nativeConfigPath,
                    provider: nativeProvider,
                    sequenceLength: Int(nativeSequenceLength.trimmingCharacters(in: .whitespacesAndNewlines)),
                    ocrBackend: nativeOCRBackend
                )
                let outputURL = temporaryRedactedImageURL(for: sourceURL)
                let response = try await client.redactImage(
                    inputPath: sourceURL.path,
                    outputPath: outputURL.path
                )
                guard let rendered = NSImage(contentsOf: outputURL) else {
                    throw AppViewModelError.couldNotOpenRedactedImage(outputURL.path)
                }

                redactedImage = rendered
                redactedImageURL = outputURL
                spans = response.detectedSpans
                ocrTokenCount = response.imageRedaction?.ocrTokenCount ?? 0
                maskCount = response.imageRedaction?.maskCount ?? 0
                let backend = response.imageRedaction?.ocrBackend ?? nativeOCRBackend
                status = "\(response.detectedSpans.count) sensitive spans, \(maskCount) image masks (\(backend))"
            } catch {
                errorMessage = error.localizedDescription
                status = "Failed"
            }
            isProcessing = false
        }
    }

    func saveRedactedImage() {
        guard let redactedImageURL else {
            return
        }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = suggestedOutputName()

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            try FileManager.default.copyItem(at: redactedImageURL, to: url)
            status = "Saved \(url.lastPathComponent)"
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func suggestedOutputName() -> String {
        guard let sourceURL else {
            return "redacted.png"
        }
        return "\(sourceURL.deletingPathExtension().lastPathComponent)-redacted.png"
    }

    private func temporaryRedactedImageURL(for sourceURL: URL) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(sourceURL.deletingPathExtension().lastPathComponent)-safeclipper-\(UUID().uuidString)")
            .appendingPathExtension("png")
    }
}

enum AppViewModelError: LocalizedError {
    case couldNotOpenRedactedImage(String)

    var errorDescription: String? {
        switch self {
        case .couldNotOpenRedactedImage(let path):
            return "Could not open redacted image at \(path)."
        }
    }
}
