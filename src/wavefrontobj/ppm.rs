use std::{io::BufRead, error::Error, fmt::Display};

use crate::mesh::{image::Image, color::Color};

enum Variant {
    Ascii(VariantAscii),
    Binary(VariantBinary)
}

enum VariantAscii {
    P1,
    P2,
    P3,
}

enum VariantBinary {
    P4,
    P5,
    P6,
}

#[derive(Debug)]
enum PpmError {
    BadHeader,
    UnexpectedEof,
    ColorOutOfRange,
    TooManyPixels,
    BadScale,
}

impl Display for PpmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl Error for PpmError { }



/// read a Netpbm (ppm, pgm, pbm) into an image
///
/// Currently, pbm is unsupported
pub fn read_ppm(reader: &mut impl BufRead) -> Result<Image, Box<dyn Error>> {

    let mut header: [u8; 2] = [0; 2];
    reader.read_exact(&mut header)?;
    let variant = match &header {
        b"P6" => Ok(Variant::Binary(VariantBinary::P6)),
        b"P5" => Ok(Variant::Binary(VariantBinary::P5)),
        b"P4" => Ok(Variant::Binary(VariantBinary::P4)),
        b"P3" => Ok(Variant::Ascii(VariantAscii::P3)),
        b"P2" => Ok(Variant::Ascii(VariantAscii::P2)),
        b"P1" => Ok(Variant::Ascii(VariantAscii::P1)),
        _ => Err(PpmError::BadHeader)
    }?;

    match variant {
        Variant::Ascii(v) => read_ppm_ascii(reader, v),
        Variant::Binary(v) => read_ppm_binary(reader, v),
    }
}

/// reads ascii variants of Netpbm format. `reader` should have already read header and pass it as
/// `variant`
fn read_ppm_ascii(reader: &mut impl BufRead, variant_ascii: VariantAscii) -> Result<Image, Box<dyn Error>> {
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
    let mut tokens = lines.iter().map(|line| {
        line.split_once('#').map_or(line.as_str(), |(data, _)| data).split_whitespace()
    }).flatten();


    let width: usize = tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?;
    let height: usize = tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?;
    if width >= 65536 || height >= 65536 {
        return Err(PpmError::BadScale.into());
    }
    let total_pix = width * height;

    // remember that bitmap has no scale - really a whole different format
    let scale: usize = tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?;

    let mut data = Vec::<Color>::with_capacity(total_pix);

    match variant_ascii {
        VariantAscii::P3 => {
            for _ in 0..total_pix {
                data.push(Color {
                    r: apply_scale(scale, tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?)?,
                    g: apply_scale(scale, tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?)?,
                    b: apply_scale(scale, tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?)?,
                    a: 255,
                });
            }
        },
        VariantAscii::P2 => {
            let px = apply_scale(scale, tokens.next().ok_or(PpmError::UnexpectedEof)?.parse()?)?;
            for _ in 0..total_pix {
                data.push(Color {
                    r: px,
                    g: px,
                    b: px,
                    a: 255,
                });
            }
        },
        VariantAscii::P1 => unimplemented!(),
    }

    if tokens.next() != None {
        return Err(PpmError::TooManyPixels.into())
    }


    Ok(Image {
        width: width as u32,
        height: width as u32,
        data,
    })
}

fn read_ppm_binary(_reader: &mut impl BufRead, _variant_binary: VariantBinary) -> Result<Image, Box<dyn Error>> {
    todo!()
}

/// validate and scale pixel value to u8
fn apply_scale(scale: usize, val: usize) -> Result<u8, PpmError> {
    if val > scale {
        return Err(PpmError::ColorOutOfRange)
    }
    ((val * u8::MAX as usize) / scale).try_into().map_err(|_| PpmError::ColorOutOfRange)
}
