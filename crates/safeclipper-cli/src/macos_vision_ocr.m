#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#include <string.h>

static char *safeclipper_copy_string(NSString *value) {
    const char *utf8 = [value UTF8String];
    if (utf8 == NULL) {
        utf8 = "";
    }
    return strdup(utf8);
}

static void safeclipper_set_error(char **error_out, NSString *message) {
    if (error_out != NULL) {
        *error_out = safeclipper_copy_string(message);
    }
}

static NSArray<NSDictionary *> *safeclipper_tokens_for_candidate(VNRecognizedText *candidate) {
    NSMutableArray<NSDictionary *> *tokens = [NSMutableArray array];
    NSString *line = candidate.string ?: @"";
    NSCharacterSet *whitespace = [NSCharacterSet whitespaceAndNewlineCharacterSet];
    NSUInteger index = 0;

    while (index < line.length) {
        while (index < line.length && [whitespace characterIsMember:[line characterAtIndex:index]]) {
            index += 1;
        }
        if (index >= line.length) {
            break;
        }

        NSUInteger start = index;
        while (index < line.length && ![whitespace characterIsMember:[line characterAtIndex:index]]) {
            index += 1;
        }

        NSRange range = NSMakeRange(start, index - start);
        NSError *boxError = nil;
        VNRectangleObservation *box = [candidate boundingBoxForRange:range error:&boxError];
        if (box == nil || boxError != nil) {
            continue;
        }

        CGRect rect = box.boundingBox;
        NSString *tokenText = [line substringWithRange:range];
        [tokens addObject:@{
            @"text": tokenText,
            @"bounding_box": @[
                @(rect.origin.x),
                @(rect.origin.y),
                @(rect.size.width),
                @(rect.size.height)
            ]
        }];
    }

    return tokens;
}

char *safeclipper_vision_ocr(const char *image_path, char **error_out) {
    @autoreleasepool {
        if (image_path == NULL || strlen(image_path) == 0) {
            safeclipper_set_error(error_out, @"missing image path");
            return NULL;
        }

        NSString *path = [NSString stringWithUTF8String:image_path];
        if (path == nil) {
            safeclipper_set_error(error_out, @"image path is not valid UTF-8");
            return NULL;
        }

        NSURL *url = [NSURL fileURLWithPath:path];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = YES;
        request.recognitionLanguages = @[ @"en-US", @"zh-Hans", @"zh-Hant" ];

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:url options:@{}];
        NSError *performError = nil;
        BOOL ok = [handler performRequests:@[ request ] error:&performError];
        if (!ok || performError != nil) {
            safeclipper_set_error(error_out, performError.localizedDescription ?: @"Vision OCR failed");
            return NULL;
        }

        NSArray<VNRecognizedTextObservation *> *results = request.results ?: @[];
        NSArray<VNRecognizedTextObservation *> *sorted = [results sortedArrayUsingComparator:^NSComparisonResult(
            VNRecognizedTextObservation *left,
            VNRecognizedTextObservation *right
        ) {
            CGRect leftBox = left.boundingBox;
            CGRect rightBox = right.boundingBox;
            CGFloat rowTolerance = 0.02;
            if (fabs(CGRectGetMidY(leftBox) - CGRectGetMidY(rightBox)) > rowTolerance) {
                return CGRectGetMidY(leftBox) > CGRectGetMidY(rightBox) ? NSOrderedAscending : NSOrderedDescending;
            }
            if (CGRectGetMinX(leftBox) < CGRectGetMinX(rightBox)) {
                return NSOrderedAscending;
            }
            if (CGRectGetMinX(leftBox) > CGRectGetMinX(rightBox)) {
                return NSOrderedDescending;
            }
            return NSOrderedSame;
        }];

        NSMutableArray<NSDictionary *> *lines = [NSMutableArray array];
        for (VNRecognizedTextObservation *observation in sorted) {
            VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
            if (candidate == nil || candidate.string.length == 0) {
                continue;
            }

            [lines addObject:@{
                @"text": candidate.string,
                @"bounding_box": @[
                    @(observation.boundingBox.origin.x),
                    @(observation.boundingBox.origin.y),
                    @(observation.boundingBox.size.width),
                    @(observation.boundingBox.size.height)
                ],
                @"tokens": safeclipper_tokens_for_candidate(candidate)
            }];
        }

        NSDictionary *payload = @{ @"lines": lines };
        NSError *jsonError = nil;
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&jsonError];
        if (jsonData == nil || jsonError != nil) {
            safeclipper_set_error(error_out, jsonError.localizedDescription ?: @"failed to encode OCR JSON");
            return NULL;
        }

        NSString *json = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];
        return safeclipper_copy_string(json ?: @"{}");
    }
}

void safeclipper_free_c_string(char *value) {
    if (value != NULL) {
        free(value);
    }
}
