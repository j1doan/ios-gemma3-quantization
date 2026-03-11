// ChatView.swift
// SwiftUI chat interface for on-device Gemma-3 inference.

import PhotosUI
import SwiftUI

// ---------------------------------------------------------------------------
// Top-level ChatView
// ---------------------------------------------------------------------------

struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    @State private var inputText: String = ""
    @State private var selectedImage: UIImage?
    @State private var photoPickerItem: PhotosPickerItem?
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // ---- Status banner ----------------------------------------
                ModelStatusBanner(state: viewModel.modelState)

                // ---- Message list -----------------------------------------
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }

                            // Error inline message
                            if let err = viewModel.errorMessage {
                                ErrorBubble(text: err)
                                    .id("error")
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                    }
                    .onChange(of: viewModel.messages.count) {
                        if let last = viewModel.messages.last {
                            withAnimation(.easeOut(duration: 0.2)) {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                    // Scroll to bottom when the last token streams in
                    .onChange(of: viewModel.messages.last?.text) {
                        if let last = viewModel.messages.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }

                Divider()

                // ---- Input bar --------------------------------------------
                InputBar(
                    text: $inputText,
                    isFocused: $inputFocused,
                    selectedImage: $selectedImage,
                    photoPickerItem: $photoPickerItem,
                    isGenerating: viewModel.modelState == .generating,
                    isModelReady: {
                        if case .ready = viewModel.modelState { return true }
                        return false
                    }(),
                    onSend: {
                        let text = inputText
                        let image = selectedImage
                        inputText = ""
                        selectedImage = nil
                        viewModel.sendMessage(text, image: image)
                    },
                    onCancel: {
                        viewModel.cancelGeneration()
                    }
                )
            }
            .navigationTitle("Gemma-3 On-Device")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        viewModel.clearHistory()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .disabled(viewModel.messages.isEmpty)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model status banner
// ---------------------------------------------------------------------------

private struct ModelStatusBanner: View {
    let state: ChatViewModel.ModelState

    var body: some View {
        Group {
            switch state {
            case .idle:
                EmptyView()

            case .loading:
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.8)
                    Text("Loading model…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
                .background(Color(.systemGroupedBackground))

            case .ready(let info):
                Text(info)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                    .background(Color(.systemGroupedBackground))

            case .generating:
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.8)
                    Text("Generating…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 6)
                .background(Color(.systemGroupedBackground))

            case .failed(let reason):
                Text("⚠️ \(reason)")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemGroupedBackground))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

private struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack(alignment: .bottom, spacing: 6) {
            if message.role == .assistant {
                // Model avatar
                Image(systemName: "cpu")
                    .font(.caption)
                    .padding(6)
                    .background(Color.blue.opacity(0.15))
                    .clipShape(Circle())
            }

            VStack(alignment: message.role == .user ? .trailing : .leading,
                   spacing: 2) {
                // Image attachment (user messages only)
                if let img = message.image {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: 200, maxHeight: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                Text(message.text.isEmpty && message.isStreaming ? "▋" : message.text)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(bubbleColor)
                    .foregroundStyle(foregroundColor)
                    .clipShape(BubbleShape(role: message.role))
                    .textSelection(.enabled)

                if message.isStreaming {
                    TypingIndicator()
                }
            }

            if message.role == .user {
                Image(systemName: "person.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.blue)
            }
        }
        .frame(maxWidth: .infinity,
               alignment: message.role == .user ? .trailing : .leading)
    }

    private var bubbleColor: Color {
        message.role == .user ? Color.blue : Color(.secondarySystemBackground)
    }

    private var foregroundColor: Color {
        message.role == .user ? .white : .primary
    }
}

// ---------------------------------------------------------------------------
// Error bubble
// ---------------------------------------------------------------------------

private struct ErrorBubble: View {
    let text: String
    var body: some View {
        Text("Error: \(text)")
            .font(.caption)
            .foregroundStyle(.red)
            .padding(10)
            .background(Color.red.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .frame(maxWidth: .infinity, alignment: .center)
    }
}

// ---------------------------------------------------------------------------
// Typing indicator (three bouncing dots)
// ---------------------------------------------------------------------------

private struct TypingIndicator: View {
    @State private var animate = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .frame(width: 6, height: 6)
                    .foregroundStyle(.secondary)
                    .scaleEffect(animate ? 1.3 : 0.85)
                    .animation(
                        .easeInOut(duration: 0.4)
                            .repeatForever(autoreverses: true)
                            .delay(Double(i) * 0.15),
                        value: animate)
            }
        }
        .padding(.leading, 14)
        .onAppear { animate = true }
    }
}

// ---------------------------------------------------------------------------
// Chat bubble shape — pointed corner on the appropriate side
// ---------------------------------------------------------------------------

private struct BubbleShape: Shape {
    let role: MessageRole
    private let radius: CGFloat = 16
    private let tailSize: CGFloat = 6

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let isUser = role == .user
        let tailX: CGFloat = isUser ? rect.maxX : rect.minX

        // Simple rounded rect; a pointed tail is an aesthetic enhancement
        path.addRoundedRect(in: rect,
                            cornerSize: CGSize(width: radius, height: radius))
        return path
    }
}

// ---------------------------------------------------------------------------
// Input bar
// ---------------------------------------------------------------------------

private struct InputBar: View {
    @Binding var text: String
    var isFocused: FocusState<Bool>.Binding
    @Binding var selectedImage: UIImage?
    @Binding var photoPickerItem: PhotosPickerItem?
    let isGenerating: Bool
    let isModelReady: Bool
    let onSend: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            // Image preview strip
            if let img = selectedImage {
                HStack {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 60)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    Button {
                        selectedImage = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .padding(.horizontal, 12)
            }

            HStack(alignment: .bottom, spacing: 10) {
                // Photo picker
                PhotosPicker(selection: $photoPickerItem,
                             matching: .images) {
                    Image(systemName: "photo")
                        .font(.system(size: 22))
                        .foregroundStyle(.blue)
                }
                .disabled(isGenerating)
                .onChange(of: photoPickerItem) {
                    Task {
                        if let data = try? await photoPickerItem?.loadTransferable(
                            type: Data.self),
                           let uiImage = UIImage(data: data) {
                            selectedImage = uiImage
                        }
                        photoPickerItem = nil
                    }
                }

                // Text field
                TextField("Message", text: $text, axis: .vertical)
                    .lineLimit(1...6)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.secondarySystemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .focused(isFocused)
                    .disabled(isGenerating)
                    .onSubmit {
                        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            onSend()
                        }
                    }

                // Send / Cancel button
                if isGenerating {
                    Button(action: onCancel) {
                        Image(systemName: "stop.circle.fill")
                            .font(.system(size: 30))
                            .foregroundStyle(.red)
                    }
                } else {
                    Button(action: onSend) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 30))
                            .foregroundStyle(canSend ? Color.blue : Color.gray.opacity(0.4))
                    }
                    .disabled(!canSend)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .background(Color(.systemBackground))
    }

    private var canSend: Bool {
        isModelReady && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

// ---------------------------------------------------------------------------
// Preview
// ---------------------------------------------------------------------------

#Preview {
    ChatView()
}
