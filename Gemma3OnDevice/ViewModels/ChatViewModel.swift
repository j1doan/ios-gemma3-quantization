// ChatViewModel.swift
// Observable ViewModel that drives ChatView.
// Calls InferenceRunner on a background thread; publishes state changes on main.

import Combine
import Foundation

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
    var isStreaming: Bool

    init(role: MessageRole, text: String, isStreaming: Bool = false) {
        self.id = UUID()
        self.role = role
        self.text = text
        self.isStreaming = isStreaming
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
                forResource: "gemma3_1b_int8_coreml", withExtension: "pte"),
            let tokenizerURL = Bundle.main.url(
                forResource: "tokenizer", withExtension: "model")
        else {
            modelState = .failed(reason:
                "Model artefacts not found in app bundle.\n" +
                "Add gemma3_1b_int8_coreml.pte and tokenizer.model " +
                "to the Xcode project under 'Copy Bundle Resources'.")
            return
        }

        // Heavy I/O + memory-map on a background thread
        let modelPath     = modelURL.path
        let tokenizerPath = tokenizerURL.path
        let runner        = self.runner

        let result = await Task.detached(priority: .userInitiated) {
            () -> Result<String, Error> in
            do {
                // Swift bridges ObjC `(BOOL)method:(NSError **)error` as a
                // throwing function — the `error:` label is dropped.
                try runner.loadModel(atPath: modelPath, tokenizerPath: tokenizerPath)
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

    func sendMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard case .ready = modelState else { return }

        // Append user bubble
        messages.append(ChatMessage(role: .user, text: trimmed))

        // Append empty assistant bubble (streamed into)
        let assistantMsg = ChatMessage(role: .assistant, text: "", isStreaming: true)
        messages.append(assistantMsg)
        streamingIndex = messages.count - 1

        modelState = .generating
        errorMessage = nil

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
                    // Mark the streaming bubble as done even on error
                    if let idx = self.streamingIndex {
                        self.messages[idx].isStreaming = false
                        self.streamingIndex = nil
                    }
                }
            })
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
