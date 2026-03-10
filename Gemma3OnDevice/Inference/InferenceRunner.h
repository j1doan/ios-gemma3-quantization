// InferenceRunner.h
// Objective-C++ bridge — exposes GemmaInference to Swift.
//
// IMPORTANT: Do NOT import this header from pure Swift files directly.
// Instead, import it from the Bridging Header:
//   Gemma3OnDevice-Bridging-Header.h
//
// Threading model:
//   - loadModel(atPath:tokenizerPath:) is synchronous and should be called
//     from a background thread (it may take ~2–10 seconds on first load).
//   - generate(prompt:config:onToken:completion:) dispatches inference on an
//     internal serial queue and streams tokens via 'onToken' callbacks.
//     Both 'onToken' and 'completion' are called on the MAIN thread.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// ---------------------------------------------------------------------------
// ObjC generation-config mirror (matches gemma::GenerationConfig)
// ---------------------------------------------------------------------------

/// Configuration passed to -generateFromPrompt:config:onToken:completion:.
@interface GemmaGenerationConfig : NSObject

/// Hard cap on generated tokens (default 512).
@property (nonatomic) NSInteger maxNewTokens;

/// Sampling temperature; 0 → greedy argmax (default 0.7).
@property (nonatomic) float temperature;

/// Nucleus sampling threshold [0,1]; 1 disables it (default 0.9).
@property (nonatomic) float topP;

/// Top-k candidate pool; 0 disables it (default 40).
@property (nonatomic) NSInteger topK;

/// Maximum total context length in tokens (default 2048).
@property (nonatomic) NSInteger maxSequenceLength;

/// Returns a config populated with the default values above.
+ (instancetype)defaultConfig;

@end

// ---------------------------------------------------------------------------
// InferenceRunner
// ---------------------------------------------------------------------------

/// Loads a Gemma-3 .pte model via FileDataLoader and runs autoregressive
/// text generation using the ExecuTorch runtime with CoreML backend.
@interface InferenceRunner : NSObject

/// Designated initialiser.
- (instancetype)init NS_DESIGNATED_INITIALIZER;

/// Load the model artefacts synchronously.
///
/// @param modelPath      Absolute path to gemma3_4b_int4_coreml.pte
/// @param tokenizerPath  Absolute path to tokenizer.model
/// @param error          Populated on failure with a descriptive string
/// @return YES if the model loaded successfully.
- (BOOL)loadModelAtPath:(NSString *)modelPath
          tokenizerPath:(NSString *)tokenizerPath
                  error:(NSError *_Nullable *_Nullable)error;

/// Generate text for a user prompt, streaming each token fragment as it
/// is decoded.
///
/// @param prompt      Raw user text (chat template is applied internally).
/// @param config      Sampling configuration.
/// @param onToken     Called on the MAIN thread for each decoded fragment.
///                    'isDone' is YES on the final call.
/// @param completion  Called on the MAIN thread once generation finishes.
///                    'fullText' is the complete generated response.
///                    'error' is non-nil if inference failed mid-way.
- (void)generateFromPrompt:(NSString *)prompt
                    config:(GemmaGenerationConfig *)config
                   onToken:(void(^)(NSString *token, BOOL isDone))onToken
                completion:(void(^)(NSString *fullText,
                                    NSError *_Nullable error))completion;

/// Cancels an in-progress generation.  The completion block will still fire
/// with whatever text was generated up to this point.
- (void)cancelGeneration;

/// YES once -loadModel… has returned YES successfully.
@property (nonatomic, readonly) BOOL isModelLoaded;

/// Human-readable description of the loaded model (vocab size, backend, …).
@property (nonatomic, readonly, nullable) NSString *modelInfo;

@end

NS_ASSUME_NONNULL_END
