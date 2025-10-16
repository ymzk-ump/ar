// === 基本設定 ===
const MODEL_URL = "./model/model.json"; // モデルがない場合でも動く
const INPUT_SIZE = 256;

let video = document.getElementById("camera");
let canvas = document.getElementById("overlay");
let ctx = canvas.getContext("2d");
let model = null;
let currentStream = null;
let running = false;
let usingFront = false;

// === カメラ起動 ===
async function startCamera() {
  if (currentStream) currentStream.getTracks().forEach(t => t.stop());
  const constraints = {
    video: {
      facingMode: usingFront ? "user" : "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: false
  };
  try {
    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = currentStream;
    await video.play();
  } catch (err) {
    alert("カメラを起動できません: " + err);
  }
}

// === モデル読み込み ===
async function loadModel() {
  document.getElementById("status").innerText = "モデル読み込み中...";
  try {
    model = await tf.loadGraphModel(MODEL_URL);
    document.getElementById("status").innerText = "モデル読み込み完了";
  } catch (e) {
    document.getElementById("status").innerText = "モデル未検出（デモ動作中）";
    model = null;
  }
}

// === 前処理 ===
function preprocess() {
  return tf.tidy(() => {
    const frame = tf.browser.fromPixels(video);
    const resized = tf.image.resizeBilinear(frame, [INPUT_SIZE, INPUT_SIZE]);
    const normalized = resized.div(255.0).expandDims(0);
    return normalized;
  });
}

// === マスク描画 ===
function drawMask(maskArray, w, h) {
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const srcX = Math.floor(x * INPUT_SIZE / w);
      const srcY = Math.floor(y * INPUT_SIZE / h);
      const id = maskArray[srcY * INPUT_SIZE + srcX];
      const i = (y * w + x) * 4;
      if (id === 1) { // 可食部
        img.data[i] = 50;
        img.data[i + 1] = 220;
        img.data[i + 2] = 50;
        img.data[i + 3] = 100;
      } else if (id === 2) { // 不可食部
        img.data[i] = 220;
        img.data[i + 1] = 60;
        img.data[i + 2] = 50;
        img.data[i + 3] = 100;
      } else {
        img.data[i + 3] = 0;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

// === ダミー描画 ===
function drawFakeMask() {
  ctx.beginPath();
  ctx.arc(canvas.width / 2, canvas.height / 2, 80, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,255,0,0.3)";
  ctx.fill();
}

// === ループ ===
async function loop() {
  if (!running) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // スマホ縦画面調整
  if (video.videoHeight > video.videoWidth) {
    canvas.style.height = "100vh";
    canvas.style.width = "auto";
  } else {
    canvas.style.width = "100vw";
    canvas.style.height = "auto";
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (model) {
    const input = preprocess();
    const preds = await model.executeAsync(input);
    const seg = Array.isArray(preds) ? preds[0] : preds;
    const argmax = tf.argMax(seg.squeeze(), -1);
    const maskArray = await argmax.data();
    drawMask(maskArray, video.videoWidth, video.videoHeight);
    tf.dispose([input, preds, seg, argmax]);
  } else {
    drawFakeMask();
  }

  requestAnimationFrame(loop);
}

// === イベント ===
document.getElementById("toggleBtn").addEventListener("click", async () => {
  if (!running) {
    await startCamera();
    await loadModel();
    running = true;
    document.getElementById("toggleBtn").innerText = "■ 停止";
    loop();
  } else {
    running = false;
    document.getElementById("toggleBtn").innerText = "▶ 開始";
    if (currentStream) currentStream.getTracks().forEach(t => t.stop());
  }
});

document.getElementById("switchCam").addEventListener("click", async () => {
  usingFront = !usingFront;
  await startCamera();
});
