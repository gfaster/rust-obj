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
