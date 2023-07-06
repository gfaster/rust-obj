use std::fmt::Display;

use image::{GrayImage, Pixel, DynamicImage, GenericImageView};

use fft2d::slice::fft_2d;
use nalgebra::Complex;

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

pub fn pixel_difference(img1: DynamicImage, img2: DynamicImage) -> Result<GrayImage, Box<dyn std::error::Error>> {
    if img1.dimensions() != img2.dimensions() {
        return Err(Error::MismatchedDimensions.into());
    }

    let gray_img1 = img1.to_luma8();
    let gray_img2 = img2.to_luma8();
    let (width, height) = img1.dimensions();

    let mut ret = GrayImage::new(width, height);
    for (x, y, pixel) in gray_img1.enumerate_pixels() {
        ret.put_pixel(x, y, pixel.map(|v| v.abs_diff(gray_img2.get_pixel(x, y).channels()[0])));    
    }
    
    Ok(ret)
}

pub fn spectral_loss(img1: DynamicImage, img2: DynamicImage) -> Result<GrayImage, Box<dyn std::error::Error>> {
    if img1.dimensions() != img2.dimensions() {
        return Err(Error::MismatchedDimensions.into());
    }

    let gray_img1 = img1.to_luma8();
    let gray_img2 = img2.to_luma8();
    let (width, height) = img1.dimensions();
    
    let mut img1_buffer = gray_img1.as_raw().iter().map(|&pix| Complex::new(pix as f64 / 255.0, 0.0)).collect::<Vec<_>>();
    let mut img2_buffer = gray_img2.as_raw().iter().map(|&pix| Complex::new(pix as f64 / 255.0, 0.0)).collect::<Vec<_>>();

    fft_2d(width.try_into().unwrap(), height.try_into().unwrap(), &mut img1_buffer);
    fft_2d(width.try_into().unwrap(), height.try_into().unwrap(), &mut img2_buffer);

    let img1_amps = img1_buffer.iter().map(|z| z.arg()).collect::<Vec<_>>();
    let img2_amps = img2_buffer.iter().map(|z| z.arg()).collect::<Vec<_>>();

    let spec_loss = img1_amps.iter().zip(img2_amps.iter()).map(|(a1, b1)| (a1 - b1).powi(2)).collect::<Vec<_>>();
    
    let mut ret = GrayImage::new(width, height);

    for idx in 0..spec_loss.len() {
        ret.put_pixel(u32::try_from(idx).unwrap() % width, u32::try_from(idx).unwrap() % height, image::Luma([(spec_loss[idx] * 255.0) as u8]));
    }

    Ok(ret)
}
