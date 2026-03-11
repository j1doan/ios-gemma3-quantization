// ImagePreprocessor.swift
// Prepares UIImage data for the Gemma-3 SigLIP vision encoder.
//
// Pipeline (Section 10 of edge spec):
//   input_image → resize 896×896 → normalize → Float32 CHW buffer
//
// All preprocessing runs on CPU outside the ExecuTorch graph.
// The output NSData can be passed directly to InferenceRunner's
// generateFromPrompt:pixelData:config:onToken:completion: method.

import CoreGraphics
import Foundation
import UIKit

enum ImagePreprocessor {

    // SigLIP normalisation constants (ImageNet-derived)
    private static let mean: [Float] = [0.5, 0.5, 0.5]
    private static let std:  [Float] = [0.5, 0.5, 0.5]

    static let inputSize: Int = 896

    /// Preprocess a UIImage into a Float32 pixel buffer suitable for the
    /// vision encoder .pte.
    ///
    /// - Returns: NSData containing `3 * 896 * 896` Float32 values in
    ///   CHW layout (channel-first, row-major), or nil on failure.
    static func preprocess(_ image: UIImage) -> NSData? {
        let size = CGSize(width: inputSize, height: inputSize)

        // 1. Resize to 896×896 using CoreGraphics (bilinear interpolation)
        guard let cgImage = image.cgImage else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * inputSize
        var rawPixels = [UInt8](repeating: 0,
                                count: inputSize * inputSize * bytesPerPixel)

        guard let context = CGContext(
            data: &rawPixels,
            width: inputSize,
            height: inputSize,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(origin: .zero, size: size))

        // 2. Convert to Float32 CHW layout with normalisation
        //    pixel = (pixel / 255.0 - mean) / std
        let npixels = inputSize * inputSize
        var chw = [Float](repeating: 0, count: 3 * npixels)

        for y in 0..<inputSize {
            for x in 0..<inputSize {
                let srcIdx = (y * inputSize + x) * bytesPerPixel
                let dstIdx = y * inputSize + x

                let r = Float(rawPixels[srcIdx])     / 255.0
                let g = Float(rawPixels[srcIdx + 1]) / 255.0
                let b = Float(rawPixels[srcIdx + 2]) / 255.0

                chw[0 * npixels + dstIdx] = (r - mean[0]) / std[0]
                chw[1 * npixels + dstIdx] = (g - mean[1]) / std[1]
                chw[2 * npixels + dstIdx] = (b - mean[2]) / std[2]
            }
        }

        return NSData(bytes: &chw, length: chw.count * MemoryLayout<Float>.size)
    }
}
