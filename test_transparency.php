<?php
// Load the photon-opencv extension
if (!extension_loaded('photon-opencv')) {
    die("Error: photon-opencv extension not loaded\n");
}

echo "PNG Transparency Analysis using Photon OpenCV\n";
echo "============================================\n\n";

$images = [
    'images_to_examine/photon.png',
    'images_to_examine/welcome-peach.png',
    'images_to_examine/grayscale_trns.png',
    'images_to_examine/palette_trns.png',
    'images_to_examine/rgb_trns.png',
    'images_to_examine/sample-1.png'
];

foreach ($images as $imagePath) {
    if (!file_exists($imagePath)) {
        echo "File not found: $imagePath\n\n";
        continue;
    }
    
    echo "File: $imagePath\n";
    echo str_repeat('-', 50) . "\n";
    
    // Load image
    $photon = new Photon_OpenCV();
    $photon->readimage($imagePath);
    
    // Get basic info
    $width = $photon->getimagewidth();
    $height = $photon->getimageheight();
    $format = $photon->getimageformat();
    $imageType = $photon->getimagetype();
    $isTransparent = $photon->ispngtransparent();
    
    echo "Dimensions: {$width}x{$height}\n";
    echo "Format: $format\n";
    echo "Image type: $imageType\n";
    echo "Has actual transparency: " . ($isTransparent ? "true" : "false") . "\n";
    
    // Show image type constants for reference
    // if ($imagePath === $images[0]) {
    //     echo "\nImage type constants:\n";
    //     echo "IMGTYPE_GRAYSCALE = 2\n";
    //     echo "IMGTYPE_GRAYSCALEMATTE = 3\n";
    //     echo "IMGTYPE_PALETTE = 4\n";
    //     echo "IMGTYPE_PALETTEMATTE = 5\n";
    //     echo "IMGTYPE_TRUECOLOR = 6\n";
    //     echo "IMGTYPE_TRUECOLORMATTE = 7\n";
    //     echo "IMGTYPE_COLORSEPARATIONMATTE = 9\n";
    // }
    
    echo "\n";
}
?>
