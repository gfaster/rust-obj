# Rust-OBJ

## Usage:
run with the path to an .obj file as the first argument (defaults to Stanford Bunny)

you probably need a few packages as well:
- opengl?
- mesa?
- shaderc? (this one tends to blow up hard on my Debian 12 install - try removing it, it'll be slower but not break)
- gcc?

## Keybinds:
- Q: exit
- W: draw as wireframe
- R: draw shaded (default)
- D: draw depth buffer, pixel value is from near to far plane
- S: Take a screenshot (Saved to `~/Pictures/rust_obj/[PID]/[TIME].ppm`)
- G: Take a greyscale screenshot (Saved to `~/Pictures/rust_obj/[PID]/[TIME].pgm`)
