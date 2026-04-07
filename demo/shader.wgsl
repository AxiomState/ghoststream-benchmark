// Ghost Stream SPAN Compute Shader — 48K parameter super-resolution
// Architecture: head(1→24) → 4×Block(conv→sigmoid_attn→conv+skip) → proj(24→24) → tail(24→4) → PixelShuffle(2)
// All convolutions are 3×3 with padding=1

struct ConvParams {
    in_ch: u32,
    out_ch: u32,
    height: u32,
    width: u32,
    weight_off: u32,   // offset into weights buffer (in f32 units)
    bias_off: u32,     // offset into weights buffer for bias
    input_off: u32,    // offset into feature buffer for input
    output_off: u32,   // offset into feature buffer for output
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> features: array<f32>;
@group(0) @binding(2) var<uniform> params: ConvParams;

// ─── 3×3 Convolution with padding=1 ───
@compute @workgroup_size(16, 16)
fn conv2d_3x3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.x;
    let x = gid.y;
    let oc = gid.z;

    let H = params.height;
    let W = params.width;
    let IC = params.in_ch;
    let OC = params.out_ch;

    if (y >= H || x >= W || oc >= OC) { return; }

    var sum: f32 = weights[params.bias_off + oc]; // bias

    for (var ic: u32 = 0u; ic < IC; ic++) {
        for (var ky: i32 = -1; ky <= 1; ky++) {
            for (var kx: i32 = -1; kx <= 1; kx++) {
                let iy = i32(y) + ky;
                let ix = i32(x) + kx;

                if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
                    let input_idx = params.input_off + ic * H * W + u32(iy) * W + u32(ix);
                    let weight_idx = params.weight_off + oc * IC * 9u + ic * 9u + u32(ky + 1) * 3u + u32(kx + 1);
                    sum += features[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    let output_idx = params.output_off + oc * H * W + y * W + x;
    features[output_idx] = sum;
}

// ─── Sigmoid Attention: out = sigmoid(x) * x ───
@compute @workgroup_size(256)
fn sigmoid_attention(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.out_ch * params.height * params.width;
    if (idx >= total) { return; }

    let feat_idx = params.input_off + idx;
    let val = features[feat_idx];
    let sig = 1.0 / (1.0 + exp(-val));
    features[params.output_off + idx] = sig * val;
}

// ─── Residual Add: output[i] = a[i] + b[i] ───
@compute @workgroup_size(256)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.out_ch * params.height * params.width;
    if (idx >= total) { return; }

    // input_off = source A, output_off = source B (also destination)
    features[params.output_off + idx] = features[params.input_off + idx] + features[params.output_off + idx];
}

// ─── PixelShuffle(2): rearrange [4, H, W] → [1, 2H, 2W] ───
@compute @workgroup_size(16, 16)
fn pixel_shuffle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oy = gid.x;  // output y
    let ox = gid.y;  // output x
    let OH = params.height * 2u;
    let OW = params.width * 2u;

    if (oy >= OH || ox >= OW) { return; }

    // Map output (oy, ox) back to input channel and position
    let iy = oy / 2u;
    let ix = ox / 2u;
    let sub_y = oy % 2u;
    let sub_x = ox % 2u;
    let ic = sub_y * 2u + sub_x;  // channel index (0-3 for scale=2)

    let input_idx = params.input_off + ic * params.height * params.width + iy * params.width + ix;
    let output_idx = params.output_off + oy * OW + ox;

    features[output_idx] = features[input_idx];
}
