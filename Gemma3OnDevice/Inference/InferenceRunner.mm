// InferenceRunner.mm
// Objective-C++ implementation — bridges GemmaInference (C++) → Swift

#import "InferenceRunner.h"
#include "GemmaInference.hpp"

#include <atomic>
#include <string>

// ---------------------------------------------------------------------------
// GemmaGenerationConfig
// ---------------------------------------------------------------------------

@implementation GemmaGenerationConfig

+ (instancetype)defaultConfig {
    GemmaGenerationConfig *cfg = [[GemmaGenerationConfig alloc] init];
    cfg.maxNewTokens     = 512;
    cfg.temperature      = 0.7f;
    cfg.topP             = 0.9f;
    cfg.topK             = 40;
    cfg.maxSequenceLength = 2048;
    return cfg;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        // Mirror the C++ defaults
        _maxNewTokens      = 512;
        _temperature       = 0.7f;
        _topP              = 0.9f;
        _topK              = 40;
        _maxSequenceLength = 2048;
    }
    return self;
}

@end

// ---------------------------------------------------------------------------
// InferenceRunner private extension
// ---------------------------------------------------------------------------

@interface InferenceRunner () {
    std::unique_ptr<gemma::GemmaInference> _engine;
    // std::atomic<bool> must be declared as a plain ivar, NOT an ObjC property.
    // ObjC property synthesis generates a getter/setter that attempts to
    // copy-construct std::atomic<bool>, which is deleted — causing a compile error.
    std::atomic<bool> _cancelRequested;
}

@property (nonatomic, strong) dispatch_queue_t inferenceQueue;

@end

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

@implementation InferenceRunner

- (instancetype)init {
    self = [super init];
    if (self) {
        _engine = std::make_unique<gemma::GemmaInference>();
        // Serial queue so that only one generate() runs at a time
        _inferenceQueue = dispatch_queue_create(
            "com.gemma3ondevice.inference",
            dispatch_queue_attr_make_with_qos_class(
                DISPATCH_QUEUE_SERIAL,
                QOS_CLASS_USER_INITIATED, 0));
        _cancelRequested.store(false);
    }
    return self;
}

// ---------------------------------------------------------------------------
- (BOOL)isModelLoaded {
    return _engine && _engine->isLoaded();
}

// ---------------------------------------------------------------------------
- (BOOL)isMultimodal {
    return _engine && _engine->isMultimodal();
}

// ---------------------------------------------------------------------------
- (nullable NSString *)modelInfo {
    if (!_engine || !_engine->isLoaded()) return nil;
    const std::string info = _engine->modelInfo();
    return [NSString stringWithUTF8String:info.c_str()];
}

// ---------------------------------------------------------------------------
- (BOOL)loadModelAtPath:(NSString *)modelPath
          tokenizerPath:(NSString *)tokenizerPath
      visionEncoderPath:(nullable NSString *)visionEncoderPath
                  error:(NSError **)error {
    NSParameterAssert(modelPath);
    NSParameterAssert(tokenizerPath);

    const std::string cppModelPath      = modelPath.UTF8String;
    const std::string cppTokenizerPath  = tokenizerPath.UTF8String;
    const std::string cppVisionPath     = visionEncoderPath
        ? std::string(visionEncoderPath.UTF8String) : std::string();

    const bool ok = _engine->load(cppModelPath, cppTokenizerPath, cppVisionPath);
    if (!ok && error) {
        *error = [NSError
            errorWithDomain:@"GemmaInferenceErrorDomain"
                       code:100
                   userInfo:@{
            NSLocalizedDescriptionKey:
                [NSString stringWithFormat:
                 @"Failed to load model at: %@", modelPath]
        }];
    }
    return ok ? YES : NO;
}

// ---------------------------------------------------------------------------
- (void)generateFromPrompt:(NSString *)prompt
                    config:(GemmaGenerationConfig *)config
                   onToken:(void(^)(NSString *, BOOL))onToken
                completion:(void(^)(NSString *, NSError *))completion {
    NSParameterAssert(prompt);
    NSParameterAssert(config);

    if (!self.isModelLoaded) {
        NSError *err = [NSError
            errorWithDomain:@"GemmaInferenceErrorDomain"
                       code:101
                   userInfo:@{NSLocalizedDescriptionKey: @"Model not loaded"}];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) completion(@"", err);
        });
        return;
    }

    _cancelRequested.store(false);

    // Copy config values to C++ struct before crossing into background thread
    gemma::GenerationConfig cppConfig;
    cppConfig.max_new_tokens      = (int)config.maxNewTokens;
    cppConfig.temperature         = config.temperature;
    cppConfig.top_p               = config.topP;
    cppConfig.top_k               = (int)config.topK;
    cppConfig.max_sequence_length = (int)config.maxSequenceLength;

    const std::string cppPrompt = prompt.UTF8String;

    // Capture a __block accumulator for the full response string
    __block NSMutableString *fullText = [NSMutableString string];

    // Keep a weak reference to self to avoid retain cycles in the block
    __weak typeof(self) weakSelf = self;

    dispatch_async(self.inferenceQueue, ^{
        __strong typeof(weakSelf) strongSelf = weakSelf;
        if (!strongSelf) return;

        __block NSError *inferenceError = nil;

        @try {
            strongSelf->_engine->generate(
                cppPrompt,
                cppConfig,
                [&](const std::string& token, bool isDone) {
                    // Check for cancellation before processing each token
                    if (strongSelf->_cancelRequested.load()) {
                        return;  // The loop in GemmaInference will not be
                                 // interrupted from here; isDone handles exit.
                    }

                    NSString *nsToken = token.empty()
                        ? @""
                        : [NSString stringWithUTF8String:token.c_str()];

                    dispatch_async(dispatch_get_main_queue(), ^{
                        __strong typeof(weakSelf) ss = weakSelf;
                        if (!ss) return;
                        if (nsToken.length > 0) [fullText appendString:nsToken];
                        if (onToken) onToken(nsToken, isDone ? YES : NO);
                    });
                });
        } @catch (NSException *ex) {
            inferenceError = [NSError
                errorWithDomain:@"GemmaInferenceErrorDomain"
                           code:102
                       userInfo:@{NSLocalizedDescriptionKey: ex.reason ?: @"Unknown error"}];
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) completion([fullText copy], inferenceError);
        });
    });
}

// ---------------------------------------------------------------------------
- (void)generateFromPrompt:(NSString *)prompt
                 pixelData:(NSData *)pixelData
                    config:(GemmaGenerationConfig *)config
                   onToken:(void(^)(NSString *, BOOL))onToken
                completion:(void(^)(NSString *, NSError *))completion {
    NSParameterAssert(prompt);
    NSParameterAssert(pixelData);
    NSParameterAssert(config);

    if (!self.isModelLoaded || !self.isMultimodal) {
        NSError *err = [NSError
            errorWithDomain:@"GemmaInferenceErrorDomain"
                       code:103
                   userInfo:@{NSLocalizedDescriptionKey:
                       @"Model not loaded or vision encoder not available"}];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) completion(@"", err);
        });
        return;
    }

    _cancelRequested.store(false);

    gemma::GenerationConfig cppConfig;
    cppConfig.max_new_tokens      = (int)config.maxNewTokens;
    cppConfig.temperature         = config.temperature;
    cppConfig.top_p               = config.topP;
    cppConfig.top_k               = (int)config.topK;
    cppConfig.max_sequence_length = (int)config.maxSequenceLength;

    const std::string cppPrompt = prompt.UTF8String;
    // Copy pixel data pointer for use on the inference queue.
    // pixelData is retained by the block.
    const float* pixelPtr = (const float*)pixelData.bytes;

    __block NSMutableString *fullText = [NSMutableString string];
    __weak typeof(self) weakSelf = self;

    dispatch_async(self.inferenceQueue, ^{
        __strong typeof(weakSelf) strongSelf = weakSelf;
        if (!strongSelf) return;

        __block NSError *inferenceError = nil;

        @try {
            strongSelf->_engine->generateWithImage(
                cppPrompt,
                pixelPtr,
                cppConfig,
                [&](const std::string& token, bool isDone) {
                    if (strongSelf->_cancelRequested.load()) return;

                    NSString *nsToken = token.empty()
                        ? @""
                        : [NSString stringWithUTF8String:token.c_str()];

                    dispatch_async(dispatch_get_main_queue(), ^{
                        __strong typeof(weakSelf) ss = weakSelf;
                        if (!ss) return;
                        if (nsToken.length > 0) [fullText appendString:nsToken];
                        if (onToken) onToken(nsToken, isDone ? YES : NO);
                    });
                });
        } @catch (NSException *ex) {
            inferenceError = [NSError
                errorWithDomain:@"GemmaInferenceErrorDomain"
                           code:102
                       userInfo:@{NSLocalizedDescriptionKey:
                           ex.reason ?: @"Unknown error"}];
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            if (completion) completion([fullText copy], inferenceError);
        });
    });
}

// ---------------------------------------------------------------------------
- (void)cancelGeneration {
    // Signal the C++ generate() loop to stop at the next token boundary.
    if (_engine) _engine->requestCancel();
    // Also prevent any already-queued callbacks from reaching the UI.
    _cancelRequested.store(true);
}

@end
