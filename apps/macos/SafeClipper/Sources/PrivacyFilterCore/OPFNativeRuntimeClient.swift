import Foundation

@_silgen_name("safeclipper_redact_image_json")
private func safeclipperRedactImageJSON(
    _ requestJSON: UnsafePointer<CChar>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutablePointer<CChar>?

@_silgen_name("safeclipper_free_string")
private func safeclipperFreeString(_ value: UnsafeMutablePointer<CChar>?)

public enum OPFNativeRuntimeError: LocalizedError {
    case nativeError(String)
    case missingJSONOutput
    case invalidJSON(String)

    public var errorDescription: String? {
        switch self {
        case .nativeError(let message):
            return "Native privacy filter failed. \(message)"
        case .missingJSONOutput:
            return "Native privacy filter did not return JSON output."
        case .invalidJSON(let message):
            return "Could not parse native privacy filter JSON output. \(message)"
        }
    }
}

public final class OPFNativeRuntimeClient: @unchecked Sendable {
    public let modelPath: String
    public let tokenizerPath: String
    public let configPath: String
    public let provider: String
    public let sequenceLength: Int?
    public let ocrBackend: String

    public init(
        modelPath: String,
        tokenizerPath: String,
        configPath: String,
        provider: String = "coreml",
        sequenceLength: Int? = 512,
        ocrBackend: String = "auto"
    ) {
        self.modelPath = modelPath
        self.tokenizerPath = tokenizerPath
        self.configPath = configPath
        self.provider = provider
        self.sequenceLength = sequenceLength
        self.ocrBackend = ocrBackend
    }

    public func redactImage(inputPath: String, outputPath: String) async throws -> OPFRedactionResponse {
        try await Task.detached {
            try self.redactImageSync(inputPath: inputPath, outputPath: outputPath)
        }.value
    }

    private func redactImageSync(inputPath: String, outputPath: String) throws -> OPFRedactionResponse {
        let request = NativeRedactImageRequest(
            model: modelPath,
            tokenizer: tokenizerPath,
            config: configPath.isEmpty ? nil : configPath,
            image: inputPath,
            outputImage: outputPath,
            provider: provider,
            ocrBackend: ocrBackend,
            tesseractBin: nil,
            tesseractLang: nil,
            tesseractPsm: nil,
            maskPadding: nil,
            intraThreads: nil,
            sequenceLength: sequenceLength
        )

        let requestData = try JSONEncoder().encode(request)
        guard let requestJSON = String(data: requestData, encoding: .utf8) else {
            throw OPFNativeRuntimeError.invalidJSON("Could not encode request as UTF-8.")
        }

        var errorPointer: UnsafeMutablePointer<CChar>?
        let responsePointer = requestJSON.withCString { pointer in
            safeclipperRedactImageJSON(pointer, &errorPointer)
        }

        defer {
            safeclipperFreeString(responsePointer)
            safeclipperFreeString(errorPointer)
        }

        guard let responsePointer else {
            let message = errorPointer.map { String(cString: $0) } ?? "unknown native error"
            throw OPFNativeRuntimeError.nativeError(message)
        }

        let responseJSON = String(cString: responsePointer)
        guard let responseData = responseJSON.data(using: .utf8) else {
            throw OPFNativeRuntimeError.missingJSONOutput
        }

        do {
            return try JSONDecoder().decode(OPFRedactionResponse.self, from: responseData)
        } catch {
            throw OPFNativeRuntimeError.invalidJSON("\(error)\n\(responseJSON)")
        }
    }
}

private struct NativeRedactImageRequest: Encodable {
    let model: String
    let tokenizer: String
    let config: String?
    let image: String
    let outputImage: String
    let provider: String
    let ocrBackend: String
    let tesseractBin: String?
    let tesseractLang: String?
    let tesseractPsm: Int?
    let maskPadding: Int?
    let intraThreads: Int?
    let sequenceLength: Int?

    enum CodingKeys: String, CodingKey {
        case model
        case tokenizer
        case config
        case image
        case outputImage = "output_image"
        case provider
        case ocrBackend = "ocr_backend"
        case tesseractBin = "tesseract_bin"
        case tesseractLang = "tesseract_lang"
        case tesseractPsm = "tesseract_psm"
        case maskPadding = "mask_padding"
        case intraThreads = "intra_threads"
        case sequenceLength = "sequence_length"
    }
}
