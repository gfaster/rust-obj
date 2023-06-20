use std::fmt::Display;

use image::{RgbaImage, ImageBuffer, GrayImage};
use image::imageops::colorops::grayscale_alpha;

#[derive(Debug)]
enum Error {
    MismatchedDimensions,
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error: {:?}", self)
    }
}

impl std::error::Error for Error { }

fn loss(img1: RgbaImage, img2: RgbaImage) -> Result<GrayImage, Box<dyn std::error::Error>> {
    if img1.dimensions() != img2.dimensions() {
        return Err(Error::MismatchedDimensions.into());
    }

    let gray_img1 = grayscale_alpha(&img1);
    let gray_img2 = grayscale_alpha(&img2);
    let (width, height) = img1.dimensions();

    let mut ret = ImageBuffer::new(width, height);
    for (x, y, pixel) in gray_img1.enumerate_pixels() {
        ret.put_pixel(x, y, pixel)
    }
    todo!()
}

fn spectral_loss(img1: RgbaImage, img2: RgbaImage) -> Result<GrayImage, Box<dyn std::error::Error>> {
    todo!()
}
