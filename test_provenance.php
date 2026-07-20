<?php
/**
 * test_provenance.php — checks default-mode provenance-XMP carriage.
 *
 * In default (stripping) mode Photon_OpenCV retains only a small allowlist of
 * provenance XMP keys from the source — Iptc4xmpExt:DigitalSourceType and
 * DigitalSourceFileType — so a machine-readable provenance marker survives
 * transforms, while EXIF, GPS, contact info and all other XMP stay stripped.
 * strip=none (keep_original_metadata = true) keeps everything.
 *
 * This repo has no standing PHPUnit harness; this mirrors the historical
 * root-level test_*.php scripts. Run on any build where the extension is loaded:
 *
 *     php test_provenance.php        # exit 0 = all pass, 1 = a check failed
 *
 * FIXTURE: a 64x48 gradient PNG carrying, at rest —
 *   XMP  : Iptc4xmpExt:DigitalSourceType / DigitalSourceFileType = trainedAlgorithmicMedia  (the allowlist)
 *          photoshop:Credit                                       (control: dropped by default)
 *          dc:description                                         (control: dropped by default)
 *   EXIF : GPS lat/long, Image.Make/Model                        (control: dropped by default)
 * Regeneration commands are documented at the bottom of this file.
 */

if (!extension_loaded('photon-opencv')) {
    fwrite(STDERR, "Error: photon-opencv extension not loaded\n");
    exit(2);
}

// Markers we grep for in the encoded output bytes (all appear verbatim as ASCII
// inside the XMP packet / EXIF TIFF, for both PNG and WebP containers).
const DST_KEY           = 'DigitalSourceType';         // Iptc4xmpExt:DigitalSourceType key
const DSFT_KEY          = 'DigitalSourceFileType';     // Iptc4xmpExt:DigitalSourceFileType key
const CREDIT_MARKER     = 'Made with Google AI';       // photoshop:Credit (not in allowlist)
const PRIVATE_XMP       = 'a private caption';          // dc:description   -> must drop
const PRIVATE_EXIF      = 'TestCameraMake';             // Exif.Image.Make  -> must drop

$fixture_b64 = <<<'B64'
iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAAxGVYSWZJSSoACAAAAAMADwECAA8A
AAAyAAAAEAECABAAAABCAAAAJYgEAAEAAABSAAAAAAAAAFRlc3RDYW1lcmFNYWtlAABUZXN0Q2Ft
ZXJhTW9kZWwABQAAAAEABAAAAAIDAAABAAIAAgAAAE4AAAACAAUAAwAAAJQAAAADAAIAAgAAAFcA
AAAEAAUAAwAAAKwAAAAAAAAAMwAAAAEAAAAeAAAAAQAAAAAAAAABAAAAAAAAAAEAAAAHAAAAAQAA
AAAAAAABAAAAk3wjvgAAC35pVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdp
bj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+Cjx4OnhtcG1ldGEgeG1sbnM6
eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDQuNC4wLUV4aXYyIj4KIDxyZGY6
UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5z
IyI+CiAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgIHhtbG5zOmlwdGNFeHQ9Imh0
dHA6Ly9pcHRjLm9yZy9zdGQvSXB0YzR4bXBFeHQvMjAwOC0wMi0yOS8iCiAgICB4bWxuczpwaG90
b3Nob3A9Imh0dHA6Ly9ucy5hZG9iZS5jb20vcGhvdG9zaG9wLzEuMC8iCiAgICB4bWxuczpkYz0i
aHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iCiAgIGlwdGNFeHQ6RGlnaXRhbFNvdXJj
ZVR5cGU9Imh0dHA6Ly9jdi5pcHRjLm9yZy9uZXdzY29kZXMvZGlnaXRhbHNvdXJjZXR5cGUvdHJh
aW5lZEFsZ29yaXRobWljTWVkaWEiCiAgIGlwdGNFeHQ6RGlnaXRhbFNvdXJjZUZpbGVUeXBlPSJo
dHRwOi8vY3YuaXB0Yy5vcmcvbmV3c2NvZGVzL2RpZ2l0YWxzb3VyY2V0eXBlL3RyYWluZWRBbGdv
cml0aG1pY01lZGlhIgogICBwaG90b3Nob3A6Q3JlZGl0PSJNYWRlIHdpdGggR29vZ2xlIEFJIj4K
ICAgPGRjOmRlc2NyaXB0aW9uPgogICAgPHJkZjpBbHQ+CiAgICAgPHJkZjpsaSB4bWw6bGFuZz0i
eC1kZWZhdWx0Ij5hIHByaXZhdGUgY2FwdGlvbiB0aGF0IG11c3Qgbm90IHN1cnZpdmUgZGVmYXVs
dCBtb2RlPC9yZGY6bGk+CiAgICA8L3JkZjpBbHQ+CiAgIDwvZGM6ZGVzY3JpcHRpb24+CiAgPC9y
ZGY6RGVzY3JpcHRpb24+CiA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgogICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
IAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAog
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgCjw/eHBhY2tldCBlbmQ9InciPz4J
QcM9AAABKUlEQVR4nNXZ6w6DIAyG4XeA3v8lL7KRuIgDlUO/xB/EqHmIpaHlBXhc+F4+ZMbZm8UH
hr21OPhcPg324+zN4gPj3gppApJ60gRU9WxRtArrOVkDMnpya0BJTwwhYT3xDwjrKaVR63q2CazC
erY1sAjrubuVsKInhZCqnhhCwnp+06ienl0ISeqp3koY1XNYA2J6KrYSpvXcSqOG9JipyNztz15K
o+b0XEmjFvXMrsjc88/WpFG7emIICeuZVJG1/Gw4T6MCes4bWxp6xlZkVpq73o6eQ2NLTM+Qiqyj
nivNXYt6Oldk3fXUNXft6ulWkQ3S0+eMbJyev81dAT2tK7LRepqekU3Qk+tKKOlpVJFN09PijGym
nlJjy7qeZxXZfD0PzshM6MG9ARmXHgHRMIqLAAAAAElFTkSuQmCC
B64;
$fixture = base64_decode($fixture_b64);

$failures = 0;
function check($label, $cond) {
    global $failures;
    printf("  [%s] %s\n", $cond ? 'PASS' : 'FAIL', $label);
    if (!$cond) { $failures++; }
}

/**
 * Push the fixture through a real transform and return the encoded bytes.
 * scaleimage() adds an operation, so the extension re-encodes and the metadata
 * write-back path runs (no-op requests take the passthrough path, tested
 * separately below).
 */
function transform($data, $keep_metadata, $format) {
    $img = new Photon_OpenCV();
    $img->readimageblob($data, $keep_metadata);
    $img->scaleimage(32, 24);       // downscale => genuine re-encode
    $img->setimageformat($format);  // 'PNG' | 'WEBP'
    return $img->getimageblob();
}

foreach (array('PNG', 'WEBP') as $format) {
    echo "== default mode ($format): provenance kept, private data stripped ==\n";
    $out = transform($fixture, false, $format);
    check("$format default keeps DigitalSourceType key",     strpos($out, DST_KEY) !== false);
    check("$format default keeps DigitalSourceFileType key", strpos($out, DSFT_KEY) !== false);
    check("$format default drops photoshop:Credit",  strpos($out, CREDIT_MARKER) === false);
    check("$format default drops private dc:description", strpos($out, PRIVATE_XMP) === false);
    check("$format default drops EXIF/GPS (Image.Make)",  strpos($out, PRIVATE_EXIF) === false);

    echo "== strip=none ($format): everything kept ==\n";
    $out = transform($fixture, true, $format);
    check("$format strip=none keeps DigitalSourceType key",     strpos($out, DST_KEY) !== false);
    check("$format strip=none keeps DigitalSourceFileType key", strpos($out, DSFT_KEY) !== false);
    check("$format strip=none keeps photoshop:Credit",  strpos($out, CREDIT_MARKER) !== false);
    check("$format strip=none keeps EXIF (Image.Make)",  strpos($out, PRIVATE_EXIF) !== false);
}

/**
 * Serve the fixture with no operations: the extension takes the no-re-encode
 * passthrough. In default mode it rewrites the container's metadata only (no
 * pixel work); strip=none returns the bytes verbatim.
 */
function passthrough($data, $keep_metadata) {
    $img = new Photon_OpenCV();
    $img->readimageblob($data, $keep_metadata);
    return $img->getimageblob();
}

echo "== default passthrough (no ops): provenance kept, private data stripped ==\n";
$out = passthrough($fixture, false);
check('passthrough default keeps DigitalSourceType key',     strpos($out, DST_KEY) !== false);
check('passthrough default keeps DigitalSourceFileType key', strpos($out, DSFT_KEY) !== false);
check('passthrough default drops photoshop:Credit',          strpos($out, CREDIT_MARKER) === false);
check('passthrough default drops private dc:description',    strpos($out, PRIVATE_XMP) === false);
check('passthrough default drops EXIF/GPS (Image.Make)',     strpos($out, PRIVATE_EXIF) === false);
// An already-filtered source has nothing left to strip, so it stays verbatim.
check('passthrough of already-filtered bytes is byte-identical', passthrough($out, false) === $out);

echo "== strip=none passthrough (no ops): bytes served verbatim ==\n";
check('passthrough strip=none is byte-identical', passthrough($fixture, true) === $fixture);

echo "\n" . ($failures === 0 ? "ALL CHECKS PASS\n" : "$failures CHECK(S) FAILED\n");
exit($failures === 0 ? 0 : 1);

/*
 * Fixture regeneration (needs Pillow + exiv2 CLI), for the record:
 *
 *   python3 -c 'from PIL import Image; w,h=64,48; im=Image.new("RGB",(w,h)); \
 *     px=im.load(); [px.__setitem__((x,y),((x*255)//w,(y*255)//h,((x+y)*255)//(w+h))) \
 *     for y in range(h) for x in range(w)]; im.save("small.png")'
 *   DST=http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia
 *   exiv2 -M"set Xmp.iptcExt.DigitalSourceType XmpText $DST" \
 *         -M"set Xmp.iptcExt.DigitalSourceFileType XmpText $DST" \
 *         -M"set Xmp.photoshop.Credit Made with Google AI" \
 *         -M"set Xmp.dc.description a private caption that must not survive default mode" \
 *         -M"set Exif.Image.Make TestCameraMake" \
 *         -M"set Exif.Image.Model TestCameraModel" \
 *         -M"set Exif.GPSInfo.GPSVersionID 2 3 0 0" \
 *         -M"set Exif.GPSInfo.GPSLatitudeRef N"  -M"set Exif.GPSInfo.GPSLatitude 51/1 30/1 0/1" \
 *         -M"set Exif.GPSInfo.GPSLongitudeRef W" -M"set Exif.GPSInfo.GPSLongitude 0/1 7/1 0/1" \
 *         small.png
 *   base64 small.png   # -> the packet above
 */
