import Foundation

public struct SensitiveSpan: Decodable, Hashable, Identifiable {
    public let label: String
    public let start: Int
    public let end: Int
    public let text: String
    public let placeholder: String

    public var id: String {
        "\(start)-\(end)-\(label)-\(text)"
    }

    public init(
        label: String,
        start: Int,
        end: Int,
        text: String,
        placeholder: String
    ) {
        self.label = label
        self.start = start
        self.end = end
        self.text = text
        self.placeholder = placeholder
    }
}

public struct OPFRedactionResponse: Decodable {
    public let schemaVersion: Int
    public let detectedSpans: [SensitiveSpan]
    public let redactedText: String
    public let imageRedaction: OPFImageRedaction?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case detectedSpans = "detected_spans"
        case redactedText = "redacted_text"
        case imageRedaction = "image_redaction"
    }
}

public struct OPFImageRedaction: Decodable {
    public let inputPath: String
    public let outputPath: String?
    public let ocrBackend: String
    public let ocrTokenCount: Int
    public let maskCount: Int

    enum CodingKeys: String, CodingKey {
        case inputPath = "input_path"
        case outputPath = "output_path"
        case ocrBackend = "ocr_backend"
        case ocrTokenCount = "ocr_token_count"
        case maskCount = "mask_count"
    }
}
