#[allow(unused_macros)]
macro_rules! log {
    ($($tok:tt)*) => {
        eprintln!(
            "[INFO {}]: {}",
            module_path!().rsplit_once("::").unwrap().1,
            format!($($tok)*)
        )
    };
    () => {
        eprintln!(
            "[INFO {}]: Line {}",
            module_path!().rsplit_once("::").unwrap().1,
            line!()
        )
    };
}

#[allow(unused_macros)]
macro_rules! log_if_slow {
    ($cutoff_millis:expr => $expr:expr) => {
        {
            let cutoff: u128 = $cutoff_millis;
            let start = ::std::time::Instant::now();
            let res = $expr;
            let elapsed = start.elapsed().as_millis();
            if elapsed > cutoff {
                log!("Expr at line {} was slow: {elapsed} ms", line!());
            }
            res
        }
    };
    ($cutoff_millis:expr => $($tok:tt)*) => {
        let cutoff: u128 = $cutoff_millis;
        let start = ::std::time::Instant::now();
        {
            $($tok)*
        }
        let elapsed = start.elapsed().as_millis();
        if elapsed > cutoff {
            log!("Block at line {} was slow: {elapsed} ms", line!());
        }
    };
}

/// implement error for a type using its [`Debug`][debug] implementation. This macro also
/// implements [`Display`][display] for the type.
///
/// [debug]: std::fmt::Debug
/// [display]: std::fmt::Display
macro_rules! impl_error {
    ($item:ty) => {
        impl ::std::fmt::Display for $item {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                <Self as ::std::fmt::Debug>::fmt(&self, f)
            }
        }
        impl ::std::error::Error for $item {}
    };
}

macro_rules! enum_variant_from {
    ($ty:ty, $variant:ident, $other:ty) => {
        impl ::std::convert::From<$other> for $ty {
            fn from(val: $other) -> Self {
                <$ty>::$variant(val)
            }
        }
    };
}

/// apply a closure to a Cell
pub trait CellMap<T: Copy> {
    fn map_cell<F>(&self, f: F)
    where
        F: FnOnce(T) -> T;
}

impl<T: Copy> CellMap<T> for std::cell::Cell<T> {
    fn map_cell<F>(&self, f: F)
    where
        F: FnOnce(T) -> T,
    {
        self.set(f(self.get()));
    }
}
