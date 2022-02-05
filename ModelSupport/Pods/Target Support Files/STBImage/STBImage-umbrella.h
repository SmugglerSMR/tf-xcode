#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#else
#ifndef FOUNDATION_EXPORT
#if defined(__cplusplus)
#define FOUNDATION_EXPORT extern "C"
#else
#define FOUNDATION_EXPORT extern
#endif
#endif
#endif

#import "STBImage.h"
#import "stb_image.h"
#import "stb_image_write.h"

FOUNDATION_EXPORT double STBImageVersionNumber;
FOUNDATION_EXPORT const unsigned char STBImageVersionString[];

