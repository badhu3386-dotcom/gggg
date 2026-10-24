// ============================================================
// Sign to Voice: webcam -> hand landmarks -> sign -> text -> speech
// Everything runs in the browser. No server, no video upload.
// ============================================================

const STORAGE_KEY = "sign_to_voice_samples_v3";
const RECORD_FRAMES = 60;      // how many frames one recording captures
const FEATURE_LEN = 128;        // 2 hands x 63 numbers + 2 numbers for how far apart the hands are
const K = 5;                   // neighbours used by the classifier

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const videoEl = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const statusEl = $("status");
const liveLabel = $("liveLabel").querySelector("b");
const sentenceEl = $("sentence");
const signListEl = $("signList");
const progressWrap = $("recordProgress");
const progressBar = progressWrap.firstElementChild;

// ---------- State ----------
let samples = loadSamples();   // [{label, f:[63 numbers]}]
let words = [];                // the translated sentence
let recording = null;          // {label, count} while recording
let candidate = null;          // label currently being held
let candidateCount = 0;
let lastAdded = null;          // stops the same sign repeating while held
let countingDown = false;      // true during the 3-2-1 before a guided recording
let guided = null;             // {list, index} while guided setup is active
let holdFrames = 15;
let maxDistance = 1.2;

// ---------- Storage ----------
function loadSamples() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
  } catch (e) {
    return [];
  }
}
function saveSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  } catch (e) {
    setStatus("Could not save signs (browser storage full or blocked).");
  }
}

// ---------- Helpers ----------
function setStatus(msg) { statusEl.textContent = msg; }

// Describe ONE hand so it does not depend on where the hand is or how big it looks:
// move the wrist to (0,0,0) and scale so the farthest point is at distance 1.
function handVector(lm, aspect) {
  const base = lm[0];
  // x is multiplied by the video's width/height so the hand is not stretched
  const pts = lm.map((p) => [(p.x - base.x) * aspect, p.y - base.y, p.z - base.z]);
  let maxD = 0;
  for (const p of pts) maxD = Math.max(maxD, Math.hypot(p[0], p[1], p[2]));
  const scale = maxD || 1;
  const vec = [];
  for (const p of pts) for (const v of p) vec.push(Math.round((v / scale) * 10000) / 10000);
  return { vec, scale, wx: base.x * aspect, wy: base.y };
}

// Two-hand features: [ "Left" hand 63 numbers ][ "Right" hand 63 numbers ][ offset x, offset y ].
// A missing hand is filled with zeros, so one-hand and two-hand signs look different.
// (Python script tools/make_signs_file.py uses exactly the same maths.)
function toFeatures(handLms, labels, aspect) {
  const items = handLms.slice(0, 2).map((lm, i) => ({ ...handVector(lm, aspect), label: labels[i] }));
  let A = null, B = null;                       // A = left slot, B = right slot
  if (items.length === 2) {
    if (items[0].label !== items[1].label) {
      A = items[0].label === "Left" ? items[0] : items[1];
      B = A === items[0] ? items[1] : items[0];
    } else {                                    // both got the same label: order by position
      [A, B] = items[0].wx <= items[1].wx ? [items[0], items[1]] : [items[1], items[0]];
    }
  } else {
    if (items[0].label === "Left") A = items[0]; else B = items[0];
  }
  const zeros = new Array(63).fill(0);
  let offX = 0, offY = 0;
  if (A && B) {
    const s = Math.max(A.scale, B.scale) || 1;
    offX = Math.round(((B.wx - A.wx) / s) * 10000) / 10000;
    offY = Math.round(((B.wy - A.wy) / s) * 10000) / 10000;
  }
  return [...(A ? A.vec : zeros), ...(B ? B.vec : zeros), offX, offY];
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// k-Nearest-Neighbours: find the K closest saved samples and let them vote.
function predict(features) {
  if (samples.length === 0) return null;
  const scored = samples.map((s) => ({ label: s.label, d: distance(features, s.f) }));
  scored.sort((a, b) => a.d - b.d);
  const top = scored.slice(0, Math.min(K, scored.length));
  const votes = {};
  for (const t of top) votes[t.label] = (votes[t.label] || 0) + 1;
  let best = null, bestVotes = 0;
  for (const label in votes) {
    if (votes[label] > bestVotes) { best = label; bestVotes = votes[label]; }
  }
  return { label: best, confidence: bestVotes / top.length, nearest: top[0].d };
}

// ---------- Speech ----------
function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = $("langSelect").value;
  window.speechSynthesis.speak(u);
}

// ---------- Sentence UI ----------
function renderSentence() {
  sentenceEl.textContent = words.length ? words.join(" ") : "(nothing yet)";
}
function addWord(w) {
  words.push(w);
  renderSentence();
  if ($("autoSpeak").checked) speak(w);
}

// ---------- Saved signs UI ----------
function renderSigns() {
  signListEl.innerHTML = "";
  const counts = {};
  for (const s of samples) counts[s.label] = (counts[s.label] || 0) + 1;
  const labels = Object.keys(counts);
  if (!labels.length) {
    signListEl.textContent = "No signs saved yet.";
    return;
  }
  for (const label of labels) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${label} (${counts[label]})`;
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "Delete this sign";
    del.onclick = () => {
      samples = samples.filter((s) => s.label !== label);
      saveSamples();
      renderSigns();
    };
    chip.appendChild(del);
    signListEl.appendChild(chip);
  }
}

// ---------- MediaPipe results (runs on every camera frame) ----------
function onResults(results) {
  canvas.width = results.image.width;
  canvas.height = results.image.height;
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const handList = results.multiHandLandmarks;
  if (!handList || handList.length === 0) {
    ctx.restore();
    liveLabel.textContent = "–";
    candidate = null;
    candidateCount = 0;
    lastAdded = null;           // hand lowered -> the same sign may be used again
    if (!recording && !countingDown) setStatus("Show your hand to the camera");
    return;
  }

  const colours = ["#4f46e5", "#f97316"];
  handList.forEach((lm, i) => {
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: colours[i % 2], lineWidth: 4 });
    drawLandmarks(ctx, lm, { color: "#ffffff", fillColor: "#22c55e", lineWidth: 1, radius: 4 });
  });
  ctx.restore();

  const labels = (results.multiHandedness || []).map((h) => h.label);
  const features = toFeatures(handList, labels, canvas.width / canvas.height);
  const handsText = handList.length === 2 ? "2 hands" : "1 hand";

  // --- Recording mode: save frames as training samples ---
  if (recording) {
    samples.push({ label: recording.label, f: features });
    recording.count++;
    progressBar.style.width = `${(recording.count / RECORD_FRAMES) * 100}%`;
    setStatus(`Recording "${recording.label}" (${handsText})… ${recording.count}/${RECORD_FRAMES}`);
    if (recording.count >= RECORD_FRAMES) finishRecording();
    return;
  }

  // --- Translate mode ---
  if (countingDown) return;   // do not translate while counting down
  const p = predict(features);
  if (!p || p.confidence < 0.6 || p.nearest > maxDistance) {
    liveLabel.textContent = p ? "not sure" : "no signs saved";
    candidate = null;
    candidateCount = 0;
    setStatus(p ? `${handsText} detected, sign not recognised` : "Teach at least one sign first");
    return;
  }

  liveLabel.textContent = p.label;
  if (p.label === candidate) {
    candidateCount++;
  } else {
    candidate = p.label;
    candidateCount = 1;
  }

  if (candidateCount >= holdFrames && candidate !== lastAdded) {
    addWord(candidate);
    lastAdded = candidate;
  }
  setStatus(`Recognising "${p.label}" (${handsText})`);
}

function beginRecording(label) {
  recording = { label, count: 0 };
  progressBar.style.width = "0%";
  progressWrap.classList.remove("hidden");
  $("recordBtn").disabled = true;
  $("guidedRecord").disabled = true;
}

function startRecording() {
  const label = $("labelInput").value.trim();
  if (!label) { alert("Type the word for this sign first."); return; }
  if (!cameraRunning) { alert("Start the camera first."); return; }
  if (recording || countingDown) return;
  beginRecording(label);
}

function finishRecording() {
  recording = null;
  saveSamples();
  renderSigns();
  progressWrap.classList.add("hidden");
  $("recordBtn").disabled = false;
  setStatus("Saved! Record again for more accuracy, or try the sign.");
  if (guided) { guided.index++; renderGuided(); }   // move on to the next sign
}

// ---------- Guided setup ----------
const PACKS = {
  mywords: ["Hello", "Thank you", "Love you", "Help", "Welcome"],
  words: ["Hello", "Thank you", "Yes", "No", "Please", "Sorry", "Help", "Water", "Food", "Good"],
  numbers: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
  letters: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split(""),
};

function renderGuided() {
  const box = $("guidedBox");
  if (!guided) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  if (guided.index >= guided.list.length) {
    $("guidedSign").textContent = "All done! 🎉";
    $("guidedCount").textContent = "Press \"Export signs\" below to download your file.";
    $("guidedRecord").disabled = true;
    return;
  }
  $("guidedSign").textContent = guided.list[guided.index];
  $("guidedCount").textContent = `Sign ${guided.index + 1} of ${guided.list.length}`;
  $("guidedRecord").disabled = false;
}

function guidedRecord() {
  if (!guided || guided.index >= guided.list.length) return;
  if (!cameraRunning) { alert("Start the camera first."); return; }
  if (recording || countingDown) return;
  const label = guided.list[guided.index];
  countingDown = true;
  $("guidedRecord").disabled = true;
  let n = 3;
  const tick = () => {
    if (n > 0) {
      setStatus(`Get ready to sign "${label}"… ${n}`);
      n--;
      setTimeout(tick, 1000);
    } else {
      countingDown = false;
      beginRecording(label);
    }
  };
  tick();
}

// ---------- Camera ----------
let cameraRunning = false;
let camera = null;
let hands = null;

async function startCamera() {
  if (cameraRunning) return;
  if (typeof Hands === "undefined") {
    setStatus("MediaPipe failed to load. Check your internet connection.");
    return;
  }
  $("startBtn").disabled = true;
  setStatus("Loading hand model…");

  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onResults);

  camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width: 640,
    height: 480,
  });

  try {
    await camera.start();
    cameraRunning = true;
    $("startBtn").textContent = "Camera running";
    setStatus("Show your hand to the camera");
  } catch (err) {
    console.error(err);
    $("startBtn").disabled = false;
    setStatus("Could not open the camera. Allow camera access and use https or localhost.");
  }
}

// ---------- Wire up buttons ----------
$("startBtn").onclick = startCamera;
$("recordBtn").onclick = startRecording;
$("guidedStart").onclick = () => {
  guided = { list: PACKS[$("packSelect").value], index: 0 };
  renderGuided();
};
$("guidedRecord").onclick = guidedRecord;
$("guidedSkip").onclick = () => { if (guided && !countingDown && !recording) { guided.index++; renderGuided(); } };
$("guidedBack").onclick = () => { if (guided && !countingDown && !recording) { guided.index = Math.max(0, guided.index - 1); renderGuided(); } };
$("guidedStop").onclick = () => { if (!countingDown && !recording) { guided = null; renderGuided(); } };
$("speakBtn").onclick = () => speak(words.join(" "));
$("undoBtn").onclick = () => { words.pop(); renderSentence(); };
$("clearBtn").onclick = () => { words = []; renderSentence(); };

$("holdRange").oninput = (e) => {
  holdFrames = Number(e.target.value);
  $("holdVal").textContent = holdFrames;
};
$("strictRange").oninput = (e) => {
  maxDistance = Number(e.target.value) / 10;
  $("strictVal").textContent = maxDistance.toFixed(1);
};

$("exportBtn").onclick = () => {
  const blob = new Blob([JSON.stringify(samples)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "my-signs.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
$("importBtn").onclick = () => $("importFile").click();
$("importFile").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (!Array.isArray(data) || !data.every((s) => s.label && Array.isArray(s.f) && s.f.length === FEATURE_LEN)) {
      throw new Error("bad format");
    }
    samples = samples.concat(data);
    saveSamples();
    renderSigns();
  } catch (err) {
    alert("That file is not valid for this version. Files made before two-hand support cannot be imported; please record or rebuild them.");
  }
  e.target.value = "";
};
$("wipeBtn").onclick = () => {
  if (confirm("Delete all saved signs?")) {
    samples = [];
    saveSamples();
    renderSigns();
  }
};

// ---------- Init ----------
renderSentence();
renderSigns();
