use glium::texture::RawImage2d;
use glium::texture::Texture2dDataSink;
use glium::texture::Texture2dDataSource;

use super::color::Color;

/// uncompressed image. 
///
/// using our own struct here in case we need to move away from glium
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<Color>
}

impl<'a> Texture2dDataSource<'a> for Image {
    type Data = Color;

    #[inline]
    fn into_raw(self) -> RawImage2d<'a, Color> {
        // this is really an unsafe construction
        // SAFTEY: Color has repr(C) and thus can be transmuted as u8u8u8u8
        RawImage2d {
            data: std::borrow::Cow::Owned(self.data),
            width: self.width,
            height: self.height,
            format: glium::texture::ClientFormat::U8U8U8U8,
        }
    }
}


impl<T: Into<Color> + Copy> Texture2dDataSink<T> for Image {
    fn from_raw(data: std::borrow::Cow<'_, [T]>, width: u32, height: u32) -> Self where [T]: ToOwned {
        Image {
            width,
            height,
            data: data.to_owned().into_iter().map(|c| (*c).into()).collect(),
        }
    }
}
