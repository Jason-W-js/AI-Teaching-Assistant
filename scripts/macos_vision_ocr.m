#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

static NSDictionary *RecognizePage(NSString *path, NSUInteger index) {
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];
    if (image == nil) {
        return @{ @"index": @(index), @"path": path, @"text": @"", @"error": @"无法读取页面图像" };
    }
    NSRect rect = NSMakeRect(0, 0, image.size.width, image.size.height);
    CGImageRef cgImage = [image CGImageForProposedRect:&rect context:nil hints:nil];
    if (cgImage == nil) {
        return @{ @"index": @(index), @"path": path, @"text": @"", @"error": @"无法转换页面图像" };
    }

    VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
    request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
    request.usesLanguageCorrection = YES;
    request.recognitionLanguages = @[ @"zh-Hans", @"en-US" ];
    VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
    NSError *error = nil;
    if (![handler performRequests:@[ request ] error:&error]) {
        return @{
            @"index": @(index), @"path": path, @"text": @"",
            @"error": error.localizedDescription ?: @"OCR 执行失败"
        };
    }

    NSArray<VNRecognizedTextObservation *> *observations = [request.results sortedArrayUsingComparator:
        ^NSComparisonResult(VNRecognizedTextObservation *left, VNRecognizedTextObservation *right) {
            CGFloat verticalGap = fabs(CGRectGetMidY(left.boundingBox) - CGRectGetMidY(right.boundingBox));
            if (verticalGap > 0.012) {
                return CGRectGetMidY(left.boundingBox) > CGRectGetMidY(right.boundingBox)
                    ? NSOrderedAscending : NSOrderedDescending;
            }
            return CGRectGetMinX(left.boundingBox) < CGRectGetMinX(right.boundingBox)
                ? NSOrderedAscending : NSOrderedDescending;
        }
    ];
    NSMutableArray<NSString *> *lines = [NSMutableArray array];
    for (VNRecognizedTextObservation *observation in observations) {
        VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
        if (candidate.string.length > 0) {
            [lines addObject:candidate.string];
        }
    }
    return @{
        @"index": @(index), @"path": path,
        @"text": [lines componentsJoinedByString:@"\n"], @"error": [NSNull null]
    };
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSMutableArray<NSDictionary *> *results = [NSMutableArray array];
        for (int index = 1; index < argc; index++) {
            NSString *path = [NSString stringWithUTF8String:argv[index]];
            [results addObject:RecognizePage(path, (NSUInteger)(index - 1))];
        }
        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:results options:0 error:&error];
        if (data == nil) {
            fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
            return 2;
        }
        [[NSFileHandle fileHandleWithStandardOutput] writeData:data];
    }
    return 0;
}
