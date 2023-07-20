#version 460

layout(location = 0) out vec4 FragColor;

// layout(location = 0) in vec3 v_fragPos;
// layout(location = 1) in vec3 v_fragNorm;
// layout(location = 2) in vec2 v_texCoord;
// layout(location = 3) in float v_depth;


#define ROWS 30
#define COLS 80

layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord; 
//
// layout(set = 0, binding = 1) uniform ShaderScreenInfo {
//     vec2 dimensions;
// } screenInfo;

layout(set = 0, binding = 2) uniform ShaderText {
    uvec4 chars[600];
} text;

layout(set = 0, binding = 3) readonly buffer ShaderFontData {
    uint bitmap[];
} fontData;

layout(set = 0, binding = 0) uniform ShaderFontMeta {
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
} font;

void main()
{
    int pos_x = int(gl_FragCoord.x);
    int pos_y =  int(gl_FragCoord.y);
    int cell_x = pos_x / (font.pixel_width + 2);
    int cell_y = pos_y / font.pixel_height;
    if (cell_x >= COLS || cell_y >= ROWS) {
        discard;
    }
    int char_idx = cell_x + (COLS * cell_y);

    uint target_char = text.chars[char_idx / 4][char_idx % 4];
    // target_char = 65 + char_idx;
    if (target_char == 0) {
        discard;
    }  

    int cell_off_y = pos_y % font.pixel_height;
    int cell_off_x = pos_x % (font.pixel_width + 2);
    uint bitmap_idx = (target_char * font.bytes_per_glyph)
                     + (font.bytes_per_row * cell_off_y);
                     // + (cell_off_x / 32);

    int bitmap_shift = (2 +font.pixel_width - cell_off_x) % 32;
    
    uint pixel = 0x1 & (fontData.bitmap[bitmap_idx] >> bitmap_shift);

    if (pixel == 1) {
        FragColor = vec4(vec3(0.9), 1.0);
    } else {
        discard;
        // FragColor = vec4(vec3(0.0), 0.1);
    }
    // FragColor = vec4(1.0);
}
