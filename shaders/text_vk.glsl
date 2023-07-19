#version 460

layout(location = 0) out vec4 FragColor;

// layout(location = 0) in vec3 v_fragPos;
// layout(location = 1) in vec3 v_fragNorm;
// layout(location = 2) in vec2 v_texCoord;
// layout(location = 3) in float v_depth;


#define ROWS 30
#define COLS 80

layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord; 

layout(set = 0, binding = 1) uniform ShaderScreenInfo {
    vec2 dimensions;
} screenInfo;

layout(set = 0, binding = 2) uniform ShaderText {
    uint chars[2400];
} text;

layout(set = 0, binding = 0) readonly buffer ShaderFont {
    int bbx_x;
    int bbx_y;
    int bbx_off_x;
    int bbx_off_y;
    int dwidth_x;
    int dwidth_y;
    int pixel_width;
    int pixel_height;
    int bytes_per_row;
    int bytes_per_glyph;
    float size;
    vec2 resolution;
    int startchar;
    uint bitmap[];
} font;

void main()
{
    int pos_x = int(gl_FragCoord.x);
    int pos_y =  int(gl_FragCoord.y);
    int char_idx = (pos_x / font.pixel_width) + (COLS * pos_y / font.pixel_height);

    uint target_char = text.chars[char_idx];
    if (target_char == 0 || target_char < font.startchar) {
        discard;
    }
    int cell_off_y = pos_y % font.bytes_per_row;
    int cell_px_x = pos_x % font.pixel_width;
    uint bitmap_idx = ((target_char - font.startchar) * font.bytes_per_glyph)
                     + (font.bytes_per_row * cell_off_y)
                     + (cell_px_x / 8);

    int bitmap_shift = cell_px_x % 8;
    
    uint pixel = 0x1 & (font.bitmap[bitmap_idx] >> bitmap_shift);

    if (pixel == 1) {
        FragColor = vec4(1.0);
    } else {
        FragColor = vec4(0.0);
    }
}
