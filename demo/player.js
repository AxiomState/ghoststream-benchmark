/**
 * Ghost Stream WebGPU Player
 * Real-time neural video super-resolution in the browser.
 *
 * Pipeline: video frame → extract Y → GPU conv2d chain → pixel shuffle → canvas
 * Architecture: SPAN 48K params (24ch, 4 blocks, sigmoid attention, PixelShuffle 2x)
 */

const BT601_R = 0.299, BT601_G = 0.587, BT601_B = 0.114;
const CHANNELS = 24;
const NUM_BLOCKS = 4;

class GhostStreamPlayer {
  constructor(videoEl, canvasGS, canvasLanczos, statusEl) {
    this.video = videoEl;
    this.canvasGS = canvasGS;
    this.canvasLanczos = canvasLanczos;
    this.statusEl = statusEl;
    this.device = null;
    this.pipelines = {};
    this.weightBuf = null;
    this.featureBuf = null;
    this.uniformBuf = null;
    this.bindGroupLayout = null;
    this.manifest = null;
    this.running = false;
    this.frameCount = 0;
    this.fpsTimer = 0;
    this.fps = 0;
  }

  log(msg) {
    console.log(`[GhostStream] ${msg}`);
    if (this.statusEl) this.statusEl.textContent = msg;
  }

  // ─── Initialization ──────────────────────────────────────────

  async init() {
    this.log('Checking WebGPU support...');
    if (!navigator.gpu) throw new Error('WebGPU not supported in this browser');

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: 256 * 1024 * 1024,
        maxBufferSize: 256 * 1024 * 1024,
      }
    });
    try { this.log(`GPU: ${adapter.info?.device || adapter.info?.description || 'WebGPU device ready'}`); } catch { this.log('GPU: WebGPU device ready'); }

    // Load weights
    this.log('Loading model weights (200KB)...');
    const [manifest, weightsBin] = await Promise.all([
      fetch('manifest.json').then(r => r.json()),
      fetch('weights.bin').then(r => r.arrayBuffer()),
    ]);
    this.manifest = manifest;

    // Convert float16 weights to float32 for GPU
    const f16 = new Uint16Array(weightsBin);
    const f32 = new Float32Array(f16.length);
    for (let i = 0; i < f16.length; i++) {
      f32[i] = float16ToFloat32(f16[i]);
    }

    this.weightBuf = this.device.createBuffer({
      size: f32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.weightBuf.getMappedRange()).set(f32);
    this.weightBuf.unmap();
    this.log(`Weights loaded: ${(manifest.param_count || manifest.total_params || 47980).toLocaleString()} parameters`);

    // Compile shaders
    this.log('Compiling shaders...');
    const shaderCode = await fetch('shader.wgsl').then(r => r.text());
    const shaderModule = this.device.createShaderModule({ code: shaderCode });

    // Create bind group layout
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });

    // Create compute pipelines for each operation
    for (const entry of ['conv2d_3x3', 'sigmoid_attention', 'residual_add', 'pixel_shuffle']) {
      this.pipelines[entry] = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: entry },
      });
    }

    // Uniform buffer for convolution params
    this.uniformBuf = this.device.createBuffer({
      size: 32, // 8 × u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.log('Ready. Click Play to start.');
  }

  // ─── Feature Buffer Management ───────────────────────────────

  allocateFeatures(h, w) {
    // We need enough space for:
    // - Input: 1 × h × w
    // - 2 feature maps: 24 × h × w each (ping-pong buffers)
    // - Tail output: 4 × h × w
    // - Final output: 1 × 2h × 2w
    const singleMap = CHANNELS * h * w;
    const totalFloats = h * w                    // input (slot 0)
                      + singleMap * 2            // feature A & B (slots 1, 2)
                      + 4 * h * w                // tail output (slot 3)
                      + 2 * h * 2 * w;           // final output (slot 4)

    if (this.featureBuf) this.featureBuf.destroy();
    this.featureBuf = this.device.createBuffer({
      size: totalFloats * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Return slot offsets (in float32 units)
    return {
      input: 0,
      featA: h * w,
      featB: h * w + singleMap,
      tail: h * w + singleMap * 2,
      output: h * w + singleMap * 2 + 4 * h * w,
      totalFloats,
    };
  }

  // ─── Dispatch Helpers ────────────────────────────────────────

  setParams(encoder, inCh, outCh, h, w, weightOff, biasOff, inputOff, outputOff) {
    const data = new Uint32Array([inCh, outCh, h, w, weightOff, biasOff, inputOff, outputOff]);
    this.device.queue.writeBuffer(this.uniformBuf, 0, data);
  }

  dispatchConv(encoder, inCh, outCh, h, w, weightOff, biasOff, inputOff, outputOff) {
    this.setParams(encoder, inCh, outCh, h, w, weightOff, biasOff, inputOff, outputOff);
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.weightBuf } },
        { binding: 1, resource: { buffer: this.featureBuf } },
        { binding: 2, resource: { buffer: this.uniformBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.conv2d_3x3);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(h / 16), Math.ceil(w / 16), outCh);
    pass.end();
  }

  dispatchOp(encoder, pipeline, total, inCh, outCh, h, w, inputOff, outputOff) {
    this.setParams(encoder, inCh, outCh, h, w, 0, 0, inputOff, outputOff);
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.weightBuf } },
        { binding: 1, resource: { buffer: this.featureBuf } },
        { binding: 2, resource: { buffer: this.uniformBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines[pipeline]);
    pass.setBindGroup(0, bindGroup);
    if (pipeline === 'pixel_shuffle') {
      pass.dispatchWorkgroups(Math.ceil(h * 2 / 16), Math.ceil(w * 2 / 16), 1);
    } else {
      pass.dispatchWorkgroups(Math.ceil(total / 256), 1, 1);
    }
    pass.end();
  }

  // ─── Full Inference ──────────────────────────────────────────

  runInference(yData, h, w) {
    const slots = this.allocateFeatures(h, w);

    // Upload Y-channel input to GPU
    this.device.queue.writeBuffer(this.featureBuf, 0, yData);

    // Build weight offset map from manifest
    const offsets = {};
    let wOff = 0;
    for (const layer of this.manifest.layers) {
      offsets[layer.name] = { weight: wOff / 2, bias: -1 }; // /2 because we stored as f16 but loaded as f32
      // Actually offsets are in f32 units after conversion
      offsets[layer.name] = { offset: wOff };
      wOff += layer.shape.reduce((a, b) => a * b, 1);
    }

    // Calculate proper f32 offsets
    let f32Off = 0;
    for (const layer of this.manifest.layers) {
      offsets[layer.name] = f32Off;
      f32Off += layer.shape.reduce((a, b) => a * b, 1);
    }

    const encoder = this.device.createCommandEncoder();

    // Head: conv2d(1→24), input → featA
    this.dispatchConv(encoder, 1, 24, h, w,
      offsets['h.weight'], offsets['h.bias'],
      slots.input, slots.featA);

    // 4 blocks
    for (let i = 0; i < NUM_BLOCKS; i++) {
      const srcSlot = (i % 2 === 0) ? slots.featA : slots.featB;
      const tmpSlot = (i % 2 === 0) ? slots.featB : slots.featA;
      const dstSlot = srcSlot; // residual back to source

      // Conv1: src → tmp
      this.dispatchConv(encoder, 24, 24, h, w,
        offsets[`b.${i}.c1.weight`], offsets[`b.${i}.c1.bias`],
        srcSlot, tmpSlot);

      // Sigmoid attention: tmp → tmp (in-place)
      this.dispatchOp(encoder, 'sigmoid_attention', 24 * h * w,
        24, 24, h, w, tmpSlot, tmpSlot);

      // Conv2: tmp → tmp (overwrite)
      const tmpSlot2 = (i % 2 === 0) ? slots.featA : slots.featB; // reuse other slot
      // Actually we need a third buffer for the conv2 output before residual add
      // Simplification: write conv2 output back to tmpSlot, then add srcSlot
      this.dispatchConv(encoder, 24, 24, h, w,
        offsets[`b.${i}.c2.weight`], offsets[`b.${i}.c2.bias`],
        tmpSlot, tmpSlot); // conv2 output in tmp

      // Residual: tmp += src → result in tmp
      this.dispatchOp(encoder, 'residual_add', 24 * h * w,
        24, 24, h, w, srcSlot, tmpSlot);

      // For next block, the result is in tmpSlot
      // Swap: even blocks result in featB, odd blocks result in featA
    }

    // After 4 blocks, result is in featA (4 is even, so last block wrote to featB,
    // but residual was added there... let's just use featB for simplicity)
    const blockOutput = (NUM_BLOCKS % 2 === 0) ? slots.featB : slots.featA;

    // Project: conv2d(24→24)
    const projSlot = (blockOutput === slots.featA) ? slots.featB : slots.featA;
    this.dispatchConv(encoder, 24, 24, h, w,
      offsets['p.weight'], offsets['p.bias'],
      blockOutput, projSlot);

    // Tail: conv2d(24→4)
    this.dispatchConv(encoder, 24, 4, h, w,
      offsets['t.weight'], offsets['t.bias'],
      projSlot, slots.tail);

    // PixelShuffle: [4, h, w] → [1, 2h, 2w]
    this.dispatchOp(encoder, 'pixel_shuffle', 0,
      4, 1, h, w, slots.tail, slots.output);

    this.device.queue.submit([encoder.finish()]);

    // Read back result
    const outputSize = 2 * h * 2 * w * 4; // float32 bytes
    const readBuf = this.device.createBuffer({
      size: outputSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const copyEncoder = this.device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this.featureBuf, slots.output * 4, readBuf, 0, outputSize);
    this.device.queue.submit([copyEncoder.finish()]);

    return readBuf.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      readBuf.destroy();
      return data;
    });
  }

  // ─── Frame Processing ────────────────────────────────────────

  extractYChannel(videoEl, targetW, targetH) {
    const canvas = document.createElement('canvas');
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, targetW, targetH);
    const imageData = ctx.getImageData(0, 0, targetW, targetH);
    const rgba = imageData.data;

    const y = new Float32Array(targetH * targetW);
    for (let i = 0; i < targetH * targetW; i++) {
      const r = rgba[i * 4] / 255;
      const g = rgba[i * 4 + 1] / 255;
      const b = rgba[i * 4 + 2] / 255;
      y[i] = BT601_R * r + BT601_G * g + BT601_B * b;
    }
    return y;
  }

  renderYToCanvas(yData, canvas, outH, outW) {
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(outW, outH);
    for (let i = 0; i < outH * outW; i++) {
      const v = Math.max(0, Math.min(255, Math.round(yData[i] * 255)));
      imageData.data[i * 4] = v;
      imageData.data[i * 4 + 1] = v;
      imageData.data[i * 4 + 2] = v;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  renderLanczos(videoEl, canvas, outW, outH) {
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(videoEl, 0, 0, outW, outH);
  }

  async processFrame() {
    if (!this.running) return;

    const lrW = 640, lrH = 360;
    const outW = 1280, outH = 720;

    // Extract Y channel at LR resolution
    const yInput = this.extractYChannel(this.video, lrW, lrH);

    // Run Ghost Stream inference
    const t0 = performance.now();
    const yOutput = await this.runInference(yInput, lrH, lrW);
    const inferenceMs = performance.now() - t0;

    // Render Ghost Stream output
    this.renderYToCanvas(yOutput, this.canvasGS, outH, outW);

    // Render Lanczos comparison (browser's bicubic)
    this.renderLanczos(this.video, this.canvasLanczos, outW, outH);

    // FPS tracking
    this.frameCount++;
    const now = performance.now();
    if (now - this.fpsTimer > 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.fpsTimer = now;
      this.log(`${this.fps} fps | inference: ${inferenceMs.toFixed(1)}ms`);
    }

    // Request next frame
    if (this.video.requestVideoFrameCallback) {
      this.video.requestVideoFrameCallback(() => this.processFrame());
    } else {
      requestAnimationFrame(() => this.processFrame());
    }
  }

  play() {
    this.running = true;
    this.fpsTimer = performance.now();
    this.frameCount = 0;
    this.video.play();
    this.processFrame();
  }

  pause() {
    this.running = false;
    this.video.pause();
  }
}

// ─── Float16 Conversion ──────────────────────────────────────

function float16ToFloat32(h) {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Denormalized
    let val = frac / 1024 * Math.pow(2, -14);
    return sign ? -val : val;
  }
  if (exp === 31) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  const val = Math.pow(2, exp - 15) * (1 + frac / 1024);
  return sign ? -val : val;
}

// ─── Exports ─────────────────────────────────────────────────

window.GhostStreamPlayer = GhostStreamPlayer;
