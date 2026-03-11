// ChatViewModel.swift
// Observable ViewModel that drives ChatView.
// Calls InferenceRunner on a background thread; publishes state changes on main.

import Combine
import Foundation
import UIKit

// ---------------------------------------------------------------------------
// Domain model
// ---------------------------------------------------------------------------

enum MessageRole: Equatable {
    case user
    case assistant
}

struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    var role: MessageRole
    var text: String
    var image: UIImage?
    var isStreaming: Bool

    init(role: MessageRole, text: String, image: UIImage? = nil,
         isStreaming: Bool = false) {
        self.id = UUID()
        self.role = role
        self.text = text
        self.image = image
        self.isStreaming = isStreaming
    }

    static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        lhs.id == rhs.id && lhs.role == rhs.role &&
        lhs.text == rhs.text && lhs.isStreaming == rhs.isStreaming
    }
}

// ---------------------------------------------------------------------------
// ViewModel
// ---------------------------------------------------------------------------

@MainActor
final class ChatViewModel: ObservableObject {

    // ---- Published state --------------------------------------------------

    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var modelState: ModelState = .idle
    @Published private(set) var errorMessage: String?

    enum ModelState: Equatable {
        case idle
        case loading
        case ready(info: String)
        case generating
        case failed(reason: String)
    }

    // ---- Configuration ----------------------------------------------------

    /// Adjust generation parameters from outside (e.g. a settings sheet).
    var generationConfig: GemmaGenerationConfig = .defaultConfig()

    // ---- Private ----------------------------------------------------------

    private let runner = InferenceRunner()

    /// Index into `messages` of the in-progress assistant bubble, if any.
    private var streamingIndex: Int?

    // ---- Initialiser ------------------------------------------------------

    init() {
        // Load the model as soon as the ViewModel is alive so it is ready
        // before the user types their first message.
        Task { await loadModel() }
    }

    // ---- Model loading ----------------------------------------------------

    private func loadModel() async {
        guard case .idle = modelState else { return }
        modelState = .loading

        // Resolve bundle paths on the calling (main) thread before hopping off
        guard
            let modelURL = Bundle.main.url(
                forResource: "gemma3_4b_int4_coreml", withExtension: "pte"),
            let tokenizerURL = Bundle.main.url(
                forResource: "tokenizer", withExtension: "model")
        else {
            modelState = .failed(reason:
                "Model artefacts not found in app bundle.\n" +
                "Add gemma3_4b_int4_coreml.pte and tokenizer.model " +
                "to the Xcode project under 'Copy Bundle Resources'.")
            return
        }

        // Vision encoder is optional — present only for multimodal builds
        let visionURL = Bundle.main.url(
            forResource: "gemma3_vision_encoder", withExtension: "pte")

        // Heavy I/O + memory-map on a background thread
        let modelPath     = modelURL.path
        let tokenizerPath = tokenizerURL.path
        let visionPath    = visionURL?.path
        let runner        = self.runner

        let result = await Task.detached(priority: .userInitiated) {
            () -> Result<String, Error> in
            do {
                try runner.loadModel(
                    atPath: modelPath,
                    tokenizerPath: tokenizerPath,
                    visionEncoderPath: visionPath)
                return .success(runner.modelInfo ?? "Gemma-3 ready")
            } catch {
                return .failure(error)
            }
        }.value

        switch result {
        case .success(let info):
            modelState = .ready(info: info)
        case .failure(let err):
            modelState = .failed(reason: err.localizedDescription)
        }
    }

    // ---- Chat send --------------------------------------------------------

    func sendMessage(_ text: String, image: UIImage? = nil) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard case .ready = modelState else { return }

        // Append user bubble (with optional image)
        messages.append(ChatMessage(role: .user, text: trimmed, image: image))

        // Append empty assistant bubble (streamed into)
        let assistantMsg = ChatMessage(role: .assistant, text: "", isStreaming: true)
        messages.append(assistantMsg)
        streamingIndex = messages.count - 1

        modelState = .generating
        errorMessage = nil

        // Preprocess image on main thread (fast enough for a single 896² resize)
        let pixelData: NSData? = image.flatMap { ImagePreprocessor.preprocess($0) }

        if let pixelData, runner.isMultimodal {
            runner.generateFromPrompt(
                trimmed,
                pixelData: pixelData,
                config: generationConfig,
                onToken: { [weak self] token, isDone in
                    guard let self, let idx = self.streamingIndex else { return }
                    self.messages[idx].text += token
                    if isDone {
                        self.messages[idx].isStreaming = false
                        self.streamingIndex = nil
                    }
                },
                completion: { [weak self] _, error in
                    guard let self else { return }
                    self.modelState = .ready(info: self.runner.modelInfo ?? "")
                    if let error {
                        self.errorMessage = error.localizedDescription
                        if let idx = self.streamingIndex {
                            self.messages[idx].isStreaming = false
                            self.streamingIndex = nil
                        }
                    }
                })
        } else {
            runner.generateFromPrompt(
                trimmed,
                config: generationConfig,
                onToken: { [weak self] token, isDone in
                    guard let self, let idx = self.streamingIndex else { return }
                    self.messages[idx].text += token
                    if isDone {
                        self.messages[idx].isStreaming = false
                        self.streamingIndex = nil
                    }
                },
                completion: { [weak self] _, error in
                    guard let self else { return }
                    self.modelState = .ready(info: self.runner.modelInfo ?? "")
                    if let error {
                        self.errorMessage = error.localizedDescription
                        if let idx = self.streamingIndex {
                            self.messages[idx].isStreaming = false
                            self.streamingIndex = nil
                        }
                    }
                })
        }
    }

    // ---- Cancel -----------------------------------------------------------

    func cancelGeneration() {
        runner.cancelGeneration()
        if let idx = streamingIndex {
            messages[idx].isStreaming = false
            streamingIndex = nil
        }
        if case .generating = modelState {
            modelState = .ready(info: runner.modelInfo ?? "")
        }
    }

    // ---- Clear history ----------------------------------------------------

    func clearHistory() {
        guard case .generating = modelState else {
            messages.removeAll()
            return
        }
        cancelGeneration()
        messages.removeAll()
    }
}
