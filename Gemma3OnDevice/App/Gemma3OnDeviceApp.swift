// Gemma3OnDeviceApp.swift
// @main entry point — loads InferenceRunner eagerly on launch.

import SwiftUI

@main
struct Gemma3OnDeviceApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView()
        }
    }
}
