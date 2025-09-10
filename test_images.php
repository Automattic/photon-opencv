<?php
// Enable dynamic loading for this script only
ini_set('enable_dl', '1');

// Load the photon-opencv extension
if (!extension_loaded('photon-opencv')) {
    // Try to load the extension from the current directory
    if (!dl('/home/dinika/code/work/photon-opencv/photon-opencv.so')) {
        die("Error: Could not load photon-opencv extension. Make sure it's compiled and available.\n");
    }
}

echo "Image Type Analysis using Photon OpenCV\n";
echo "=====================================\n\n";

$imageFiles = [
    'images_to_examine/photon.png',
    'images_to_examine/photon.webp'
];

foreach ($imageFiles as $filename) {
    echo "File: $filename\n";
    echo "----------------------------------------\n";
    
    if (!file_exists($filename)) {
        echo "Error: File does not exist\n\n";
        continue;
    }
    
    try {
        // Create Photon_OpenCV instance
        $photon = new Photon_OpenCV();
        
        // Read the image
        $photon->readimage($filename);
        
        // Get image information
        $width = $photon->getimagewidth();
        $height = $photon->getimageheight();
        $format = $photon->getimageformat();
        $imageType = $photon->getimagetype();
        
        echo "Dimensions: {$width}x{$height}\n";
        echo "Format: $format\n";
        echo "Image type: $imageType (";
        
        // Map numeric type to name
        switch ($imageType) {
            case Photon_OpenCV::IMGTYPE_GRAYSCALE:
                echo "GRAYSCALE";
                break;
            case Photon_OpenCV::IMGTYPE_GRAYSCALEMATTE:
                echo "GRAYSCALEMATTE";
                break;
            case Photon_OpenCV::IMGTYPE_PALETTE:
                echo "PALETTE";
                break;
            case Photon_OpenCV::IMGTYPE_PALETTEMATTE:
                echo "PALETTEMATTE";
                break;
            case Photon_OpenCV::IMGTYPE_TRUECOLOR:
                echo "TRUECOLOR";
                break;
            case Photon_OpenCV::IMGTYPE_TRUECOLORMATTE:
                echo "TRUECOLORMATTE";
                break;
            case Photon_OpenCV::IMGTYPE_COLORSEPARATIONMATTE:
                echo "COLORSEPARATIONMATTE";
                break;
            default:
                echo "UNKNOWN";
                break;
        }
        
        echo ")\n";
        
        // Get additional info if available
        $orientation = $photon->getimageorientation();
        if ($orientation > 0) {
            echo "Orientation: $orientation\n";
        }
        
        $compressionQuality = $photon->getcompressionquality();
        echo "Compression quality: $compressionQuality\n";
        
    } catch (Exception $e) {
        echo "Error: " . $e->getMessage() . "\n";
    }
    
    echo "\n";
}

echo "Image type constants:\n";
echo "IMGTYPE_GRAYSCALE = " . Photon_OpenCV::IMGTYPE_GRAYSCALE . "\n";
echo "IMGTYPE_GRAYSCALEMATTE = " . Photon_OpenCV::IMGTYPE_GRAYSCALEMATTE . "\n";
echo "IMGTYPE_PALETTE = " . Photon_OpenCV::IMGTYPE_PALETTE . "\n";
echo "IMGTYPE_PALETTEMATTE = " . Photon_OpenCV::IMGTYPE_PALETTEMATTE . "\n";
echo "IMGTYPE_TRUECOLOR = " . Photon_OpenCV::IMGTYPE_TRUECOLOR . "\n";
echo "IMGTYPE_TRUECOLORMATTE = " . Photon_OpenCV::IMGTYPE_TRUECOLORMATTE . "\n";
echo "IMGTYPE_COLORSEPARATIONMATTE = " . Photon_OpenCV::IMGTYPE_COLORSEPARATIONMATTE . "\n";
?>
