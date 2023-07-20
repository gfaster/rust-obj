use std::{io::BufRead, num::ParseIntError};

pub struct Font {
    bitmap: Vec<Option<Vec<u32>>>,

    /// `[w, h, off_x, off_y]`
    pub bbx: [i32; 4],
    pub size: [u32; 3],
    // s_size: u32
}

#[derive(Debug)]
#[non_exhaustive]
pub enum FontParseError {
    UnexpectedDirective,
    NoStartfontDirective,
    PrematureEndProperties,
    IncorrectArguments,
    PropertySetTwice,
    CruftAfterEnd,
    DifferentCharProperty,
    NecessaryPropertyNotSet,
    PrematureEndChar,
    ParseIntError(ParseIntError),
    IoError(std::io::Error),
}
impl_error!(FontParseError);
enum_variant_from!(FontParseError, IoError, std::io::Error);
enum_variant_from!(FontParseError, ParseIntError, ParseIntError);

#[derive(Debug)]
enum ReadState {
    Row(u32),
    CharProperties,
    FontProperties,
    ExtProperties(u32),
    Chars,
    Start,
    End,
}

#[inline]
#[must_use]
fn expect_unset<T>(item: &mut Option<T>) -> Result<&mut Option<T>, FontParseError> {
    match item {
        Some(_) => Err(FontParseError::PropertySetTwice),
        None => Ok(item),
    }
}

#[inline]
#[must_use]
fn expect_eq<T: Eq>(item: &Option<T>, other: &T) -> Result<(), FontParseError> {
    match item {
        Some(inner) if inner == other => Ok(()),
        Some(_) => Err(FontParseError::DifferentCharProperty),
        None => Err(FontParseError::NecessaryPropertyNotSet),
    }
}

#[inline]
#[must_use]
fn unwrap_property<T>(item: &Option<T>) -> Result<&T, FontParseError> {
    match item {
        Some(x) => Ok(x),
        None => Err(FontParseError::NecessaryPropertyNotSet),
    }
}

impl Font {
    pub fn parse_bdf(reader: &mut impl BufRead) -> Result<Self, FontParseError> {
        let mut line = String::new();
        let mut state = ReadState::Start;

        let mut bbx: Option<[i32; 4]> = None;
        let mut size: Option<[u32; 3]> = None;
        let mut encoding: Option<u8> = None;
        let mut chars: Option<u32> = None;
        let mut bitmaps: Vec<Option<Vec<u32>>> = vec![None; 256];

        while 0 != reader.read_line(&mut line)? {
            // dbg!((&line, &state, &chars));
            let tokens: Vec<_> = line.split_whitespace().collect();
            match state {
                ReadState::Row(r) => match &tokens[..] {
                    [] | ["COMMENT", ..] => (),
                    ["ENDCHAR"] if r == 0 => {
                        chars = Some(chars.ok_or(FontParseError::UnexpectedDirective)? - 1);
                        encoding = None;
                        state = ReadState::Chars;
                    }
                    [val] if r > 0 => {
                        let bitmap: &mut Vec<_> = bitmaps[*unwrap_property(&encoding)? as usize]
                            .as_mut()
                            .expect("bitmap should be previously initialized");
                        bitmap.push(u8::from_str_radix(val, 16)? as u32);
                        state = ReadState::Row(r - 1);
                    }
                    ["ENDCHAR"] => return Err(FontParseError::PrematureEndChar),
                    _ => return Err(FontParseError::UnexpectedDirective),
                },
                ReadState::Chars => match &tokens[..] {
                    [] | ["COMMENT", ..] => (),
                    ["STARTCHAR", _] => {
                        expect_unset(&mut encoding)?;
                        state = ReadState::CharProperties
                    }
                    ["ENDFONT"] if chars == Some(0) => {
                        break;
                    }
                    _ => return Err(FontParseError::UnexpectedDirective),
                },
                ReadState::CharProperties => match &tokens[..] {
                    [] | ["COMMENT", ..] => (),
                    ["SWIDTH", w, h] => {
                        let _: u32 = w.parse()?;
                        let _: u32 = h.parse()?;
                    }
                    ["DWIDTH", w, h] => {
                        let _: u32 = w.parse()?;
                        let _: u32 = h.parse()?;
                    }
                    ["BBX", w, h, x, y] => {
                        expect_eq(&bbx, &[w.parse()?, h.parse()?, x.parse()?, y.parse()?])?;
                    }
                    ["ENCODING", val] => {
                        *expect_unset(&mut encoding)? = Some(val.parse()?);
                    }
                    ["BITMAP"] => {
                        bitmaps[*unwrap_property(&encoding)? as usize] =
                            Some(Vec::with_capacity(unwrap_property(&bbx)?[1] as usize));
                        state = ReadState::Row(unwrap_property(&bbx)?[1] as u32)
                    }
                    _ => return Err(FontParseError::UnexpectedDirective),
                },
                ReadState::FontProperties => match &tokens[..] {
                    [] | ["COMMENT", ..] => (),
                    ["STARTPROPERTIES", count] => {
                        let count = count.parse()?;
                        state = ReadState::ExtProperties(count);
                    }
                    ["FONTBOUNDINGBOX", w, h, x, y] => {
                        *expect_unset(&mut bbx)? =
                            Some([w.parse()?, h.parse()?, x.parse()?, y.parse()?]);
                    }
                    ["FONT", ..] => (),
                    ["SIZE", points, x_res, y_res] => {
                        *expect_unset(&mut size)? =
                            Some([points.parse()?, x_res.parse()?, y_res.parse()?]);
                    }
                    ["CHARS", cnt] => {
                        state = ReadState::Chars;
                        *expect_unset(&mut chars)? = Some(cnt.parse()?);
                    }
                    _ => return Err(FontParseError::UnexpectedDirective),
                },
                ReadState::Start => match &tokens[..] {
                    [] | ["COMMENT", ..] => (),
                    ["STARTFONT", _] => state = ReadState::FontProperties,
                    ["STARTFONT", ..] => return Err(FontParseError::IncorrectArguments),
                    _ => return Err(FontParseError::NoStartfontDirective),
                },
                ReadState::End => return Err(FontParseError::CruftAfterEnd),
                // these are system dependant, and we can just ignore it
                ReadState::ExtProperties(cnt) => match &tokens[..] {
                    [] | ["COMMENT", ..] => (),
                    ["ENDPROPERTIES"] if cnt == 0 => {
                        state = ReadState::FontProperties;
                    }
                    ["ENDPROPERTIES", ..] if cnt == 0 => {
                        return Err(FontParseError::IncorrectArguments)
                    }
                    ["ENDPROPERTIES", ..] => return Err(FontParseError::PrematureEndProperties),
                    [_, ..] if cnt > 0 => state = ReadState::ExtProperties(cnt - 1),
                    [..] => {
                        return Err(FontParseError::UnexpectedDirective);
                    }
                },
            }

            line.clear();
        }

        Ok(Font {
            bitmap: bitmaps,
            bbx: *unwrap_property(&bbx)?,
            size: *unwrap_property(&size)?,
        })
    }

    pub fn flat_bitmaps(&self) -> Vec<u32> {
        // let blank = vec![0xF0F0F0; self.bbx[1] as usize];
        let blank = vec![0x0; self.bbx[1] as usize];
        // dbg!(&self.bitmap);

        self.bitmap
            .iter()
            .flat_map(|c| {
                match c {
                    Some(v) => &v[..],
                    None => &blank,
                }
                // &blank
            })
            .map(|b| *b)
            .collect()
    }
}
