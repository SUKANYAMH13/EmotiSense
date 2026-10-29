/**
 * app.js — EmotiSense
 */
const API = ""; // Using relative path to avoid CORS issues

// ── Tabs ───────────────────────────────────────────────────────────
document.querySelectorAll(".module-tab").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".module-tab").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("pane-" + btn.dataset.tab).classList.add("active");
    });
});

// ── Toast ─────────────────────────────────────────────────────────
function toast(msg, type = "") {
    const box = document.getElementById("toastBox");
    const t = document.createElement("div");
    t.className = "toast " + type;
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity 0.3s"; setTimeout(() => t.remove(), 300); }, 3000);
}

// ── Spinner HTML ──────────────────────────────────────────────────
function spinner(text = "Analyzing…") {
    return `<div class="spinner-wrap"><div class="spinner"></div><span class="spinner-text">${text}</span></div>`;
}

// ── Render bars ───────────────────────────────────────────────────
function bars(items, nameKey, valKey) {
    let h = `<div class="bars-label">All Scores</div>`;
    items.forEach(it => {
        const v = parseFloat(it[valKey]) || 0;
        h += `<div class="bar-row">
      <span class="bar-name">${it[nameKey]}</span>
      <div class="bar-track"><div class="bar-fill" data-val="${v}"></div></div>
      <span class="bar-val">${v.toFixed(1)}%</span>
    </div>`;
    });
    return h;
}
function animateBars(container) {
    container.querySelectorAll(".bar-fill").forEach(b => {
        requestAnimationFrame(() => { b.style.width = b.dataset.val + "%"; });
    });
}

// ══════════════════════════════════════════════════════════════════
// FACIAL — WEBCAM
// ══════════════════════════════════════════════════════════════════
const video = document.getElementById("webcamVideo");
const camPH = document.getElementById("camPlaceholder");
const startBtn = document.getElementById("startCamBtn");
const stopBtn = document.getElementById("stopCamBtn");
const livePill = document.getElementById("livePill");
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
let stream = null, timer = null, busy = false;

startBtn.addEventListener("click", async () => {
    // mediaDevices is only available in secure contexts (https or localhost)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        toast("Camera not available. Please open the app via http://localhost:5000 (not 127.0.0.1).", "error");
        return;
    }
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        camPH.style.display = "none";
        livePill.style.display = "inline-flex";
        startBtn.disabled = true; stopBtn.disabled = false;
        timer = setInterval(snapAndSend, 1500);
    } catch (e) { toast("Camera access denied: " + e.message, "error"); }
});

stopBtn.addEventListener("click", () => {
    clearInterval(timer);
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; video.srcObject = null;
    camPH.style.display = "flex";
    livePill.style.display = "none";
    startBtn.disabled = false; stopBtn.disabled = true;
});

async function snapAndSend() {
    if (busy || !stream || video.readyState < 2) return;
    busy = true;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    canvas.toBlob(async blob => {
        try {
            const fd = new FormData(); fd.append("image", blob, "frame.jpg");
            const r = await fetch(`${API}/predict/facial`, { method: "POST", body: fd });
            renderFacial(await r.json());
        } catch (err) { 
            console.error("API Fetch Error:", err);
            toast("Cannot reach API. Check if api.py is running and console for details.", "error"); 
            clearInterval(timer); 
            stopBtn.click(); 
        }
        finally { busy = false; }
    }, "image/jpeg", 0.85);
}

// ── Render facial result ──────────────────────────────────────────
const EMOJIS = { Angry: "😡", Disgust: "🤮", Fear: "😱", Happy: "😄", Neutral: "😐", Sad: "😭", Surprise: "😮" };

function renderFacial(data) {
    const panel = document.getElementById("facialResult");
    if (data.error) { panel.innerHTML = `<p style="color:#dc2626">${data.error}</p>`; return; }
    if (!data.emotion) {
        panel.innerHTML = `<div class="result-empty"><span class="big-icon">🔍</span>${data.message || "No face detected"}</div>`;
        return;
    }
    panel.innerHTML = `
    <div class="result-emotion">
      <span class="big-emoji">${EMOJIS[data.emotion] || "🙂"}</span>
      <div class="emotion-name">${data.emotion}</div>
      <div class="emotion-conf">${data.confidence.toFixed(1)}% confident</div>
    </div>`;
    updateFusionVote('facial', data.emotion, data.confidence);
}


// ── Upload image ───────────────────────────────────────────────────
const dz = document.getElementById("dropZone");
const inp = document.getElementById("uploadInput");
const prev = document.getElementById("previewImg");
const uBtns = document.getElementById("uploadBtns");
const analyzeUpBtn = document.getElementById("analyzeUploadBtn");
const clearUpBtn = document.getElementById("clearUploadBtn");
let uploadFile = null;

dz.addEventListener("click", () => inp.click());
dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", e => { e.preventDefault(); dz.classList.remove("dragover"); loadFile(e.dataTransfer.files[0]); });
inp.addEventListener("change", () => loadFile(inp.files[0]));

function loadFile(f) {
    if (!f || !f.type.startsWith("image/")) { toast("Please select an image file.", "error"); return; }
    uploadFile = f;
    const rd = new FileReader();
    rd.onload = e => {
        prev.src = e.target.result; prev.style.display = "block";
        dz.style.display = "none"; uBtns.style.display = "flex";
    };
    rd.readAsDataURL(f);
}

analyzeUpBtn.addEventListener("click", async () => {
    if (!uploadFile) return;
    const panel = document.getElementById("facialResult");
    panel.innerHTML = spinner("Detecting emotion…");
    analyzeUpBtn.disabled = true;
    try {
        const fd = new FormData(); fd.append("image", uploadFile);
        const r = await fetch(`${API}/predict/facial`, { method: "POST", body: fd });
        renderFacial(await r.json());
    } catch { toast("Cannot reach API. Is api.py running?", "error"); }
    finally { analyzeUpBtn.disabled = false; }
});

clearUpBtn.addEventListener("click", () => {
    uploadFile = null; inp.value = "";
    prev.style.display = "none"; prev.src = "";
    dz.style.display = "block"; uBtns.style.display = "none";
    document.getElementById("facialResult").innerHTML =
        `<div class="result-empty"><span class="big-icon">🙂</span>Start camera or upload an image to detect emotions</div>`;
});

// ══════════════════════════════════════════════════════════════════
// TEXT EMOTION
// ══════════════════════════════════════════════════════════════════
const txtInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeTxt = document.getElementById("analyzeTextBtn");
const clearTxt = document.getElementById("clearTextBtn");

txtInput.addEventListener("input", () => { charCount.textContent = txtInput.value.length; });
txtInput.addEventListener("keydown", e => { if (e.key === "Enter" && e.ctrlKey) analyzeTxt.click(); });

analyzeTxt.addEventListener("click", async () => {
    const text = txtInput.value.trim();
    if (!text) { toast("Please type something first.", "error"); return; }
    const panel = document.getElementById("textResult");
    panel.innerHTML = spinner("BERT is thinking…");
    analyzeTxt.disabled = true;
    try {
        const r = await fetch(`${API}/predict/text`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        renderText(await r.json());
        toast("Done!", "success");
    } catch { toast("Cannot reach API. Is api.py running?", "error"); }
    finally { analyzeTxt.disabled = false; }
});

clearTxt.addEventListener("click", () => {
    txtInput.value = ""; charCount.textContent = "0";
    document.getElementById("textResult").innerHTML =
        `<div class="result-empty"><span class="big-icon">💬</span>Type something to analyze your emotions</div>`;
});

function renderText(data) {
    const panel = document.getElementById("textResult");
    if (data.error) { panel.innerHTML = `<p style="color:#dc2626">${data.error}</p>`; return; }
    panel.innerHTML = `
    <div class="result-emotion">
      <span class="big-emoji">${data.best_emoji || "🙂"}</span>
      <div class="emotion-name">${data.best_label}</div>
    </div>
    ${data.best_quote ? `<div class="quote-box">${data.best_quote}</div>` : ""}
    ${data.emotions && data.emotions.length > 1 ? bars(data.emotions, "label", "probability") : ""}`;
    animateBars(panel);
    // confidence = top emotion probability
    const conf = data.emotions && data.emotions[0] ? data.emotions[0].probability : 50;
    updateFusionVote('text', data.best_label, conf);
}

// ══════════════════════════════════════════════════════════════════
// SPEECH / AUDIO EMOTION
// ══════════════════════════════════════════════════════════════════
const startMicBtn   = document.getElementById("startMicBtn");
const stopMicBtn    = document.getElementById("stopMicBtn");
const clearMicBtn   = document.getElementById("clearMicBtn");
const micPill       = document.getElementById("micPill");
const audioPlayback = document.getElementById("audioPlayback");
const micPlaceholder = document.getElementById("micPlaceholder");

let mediaRecorder = null, audioChunks = [], recordedBlob = null;

startMicBtn.addEventListener("click", async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        toast("Microphone not available. Open via http://localhost:5000", "error"); return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            recordedBlob = new Blob(audioChunks, { type: "audio/wav" });
            audioPlayback.src = URL.createObjectURL(recordedBlob);
            audioPlayback.style.display = "block";
            micPlaceholder.style.display = "none";
            clearMicBtn.style.display = "inline-flex";
            startMicBtn.disabled = false;
            micPill.style.display = "none";
            // Auto-analyze
            await sendAudioBlob(recordedBlob);
        };
        mediaRecorder.start();
        startMicBtn.disabled = true; stopMicBtn.disabled = false;
        micPill.style.display = "inline-flex";
        toast("Recording… click Stop & Analyze when done.", "");
    } catch (e) { toast("Microphone access denied: " + e.message, "error"); }
});

stopMicBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        stopMicBtn.disabled = true;
    }
});

clearMicBtn.addEventListener("click", () => {
    recordedBlob = null;
    audioPlayback.src = ""; audioPlayback.style.display = "none";
    micPlaceholder.style.display = "flex";
    clearMicBtn.style.display = "none";
    startMicBtn.disabled = false; stopMicBtn.disabled = true;
    document.getElementById("audioResult").innerHTML =
        `<div class="result-empty"><span class="empty-icon">🎤</span><span class="empty-title">Awaiting Audio</span><span class="empty-desc">Record or upload audio to detect emotion.</span></div>`;
});

async function sendAudioBlob(blob) {
    const panel = document.getElementById("audioResult");
    panel.innerHTML = spinner("Analyzing speech…");
    try {
        const fd = new FormData(); fd.append("audio", blob, "recording.wav");
        const r = await fetch(`${API}/predict/audio`, { method: "POST", body: fd });
        renderAudio(await r.json());
        toast("Done!", "success");
    } catch { toast("Cannot reach API. Is api.py running?", "error"); }
}

// ── Audio File Upload ─────────────────────────────────────────────
const audioDZ        = document.getElementById("audioDZ");
const audioInp       = document.getElementById("audioUploadInput");
const audioUpBtns    = document.getElementById("audioUploadBtns");
const analyzeAuBtn   = document.getElementById("analyzeAudioBtn");
const clearAuBtn     = document.getElementById("clearAudioBtn");
const audioFileName  = document.getElementById("audioFileName");
let audioUploadFile  = null;

audioDZ.addEventListener("click", () => audioInp.click());
audioDZ.addEventListener("dragover", e => { e.preventDefault(); audioDZ.classList.add("dragover"); });
audioDZ.addEventListener("dragleave", () => audioDZ.classList.remove("dragover"));
audioDZ.addEventListener("drop", e => {
    e.preventDefault(); audioDZ.classList.remove("dragover");
    loadAudioFile(e.dataTransfer.files[0]);
});
audioInp.addEventListener("change", () => loadAudioFile(audioInp.files[0]));

const uploadedAudioPlayer = document.getElementById("uploadedAudioPlayer");
let uploadedAudioURL = null;

function loadAudioFile(f) {
    if (!f) return;
    if (!f.type.startsWith("audio/")) { toast("Please select an audio file (WAV or MP3).", "error"); return; }
    audioUploadFile = f;
    audioFileName.textContent = "📎 " + f.name;
    // Revoke previous blob URL to free memory
    if (uploadedAudioURL) URL.revokeObjectURL(uploadedAudioURL);
    uploadedAudioURL = URL.createObjectURL(f);
    uploadedAudioPlayer.src = uploadedAudioURL;
    uploadedAudioPlayer.style.display = "block";
    audioDZ.style.display = "none";
    audioUpBtns.style.display = "flex";
}

analyzeAuBtn.addEventListener("click", async () => {
    if (!audioUploadFile) return;
    analyzeAuBtn.disabled = true;
    const fd = new FormData(); fd.append("audio", audioUploadFile);
    const panel = document.getElementById("audioResult");
    panel.innerHTML = spinner("Analyzing audio…");
    try {
        const r = await fetch(`${API}/predict/audio`, { method: "POST", body: fd });
        renderAudio(await r.json());
        toast("Done!", "success");
    } catch { toast("Cannot reach API. Is api.py running?", "error"); }
    finally { analyzeAuBtn.disabled = false; }
});

clearAuBtn.addEventListener("click", () => {
    audioUploadFile = null; audioInp.value = "";
    audioDZ.style.display = "block"; audioUpBtns.style.display = "none";
    audioFileName.textContent = "";
    if (uploadedAudioURL) { URL.revokeObjectURL(uploadedAudioURL); uploadedAudioURL = null; }
    uploadedAudioPlayer.src = ""; uploadedAudioPlayer.style.display = "none";
    document.getElementById("audioResult").innerHTML =
        `<div class="result-empty"><span class="empty-icon">🎤</span><span class="empty-title">Awaiting Audio</span><span class="empty-desc">Record or upload audio to detect emotion.</span></div>`;
});

// ── Render audio result ───────────────────────────────────────────
function renderAudio(data) {
    const panel = document.getElementById("audioResult");
    if (data.error) { panel.innerHTML = `<p style="color:#dc2626">${data.error}</p>`; return; }

    const scores = data.all_scores || {};
    const scoresArr = Object.entries(scores).map(([k, v]) => ({ label: k, value: v }))
                            .sort((a, b) => b.value - a.value);

    let barsHtml = `<div class="bars-label">All Scores</div>`;
    scoresArr.forEach(({ label, value }) => {
        barsHtml += `<div class="bar-row">
          <span class="bar-name">${label}</span>
          <div class="bar-track"><div class="bar-fill" data-val="${value}"></div></div>
          <span class="bar-val">${value.toFixed(1)}%</span>
        </div>`;
    });

    panel.innerHTML = `
    <div class="result-emotion">
      <span class="big-emoji">${data.emoji || "🎤"}</span>
      <div class="emotion-name">${data.emotion ? data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1) : "Unknown"}</div>
      <div class="emotion-conf">${(data.confidence || 0).toFixed(1)}% confident</div>
    </div>
    ${data.quote ? `<div class="quote-box">${data.quote}</div>` : ""}
    ${scoresArr.length > 1 ? barsHtml : ""}`;
    animateBars(panel);
    updateFusionVote('speech', data.emotion, data.confidence || 0);
}

// ══════════════════════════════════════════════════════════════════
// MULTIMODAL FUSION — MAJORITY VOTING
// ══════════════════════════════════════════════════════════════════
const COMMON_EMOJIS = {
    angry:"😡", disgust:"🤮", fear:"😱",
    happy:"😄", neutral:"😶", sad:"😭", surprise:"😮"
};

// Facial (title-case) → common 7
const FACIAL_MAP = {
    Angry:"angry", Disgust:"disgust", Fear:"fear",
    Happy:"happy", Neutral:"neutral", Sad:"sad", Surprise:"surprise"
};

// BERT GoEmotions 27 labels → common 7
const TEXT_MAP = {
    anger:"angry",      annoyance:"angry",
    disgust:"disgust",  disapproval:"disgust",
    fear:"fear",        nervousness:"fear",
    joy:"happy",        amusement:"happy",    excitement:"happy",
    admiration:"happy", approval:"happy",     caring:"happy",
    gratitude:"happy",  love:"happy",         optimism:"happy",
    pride:"happy",      realization:"happy",  relief:"happy",
    neutral:"neutral",  confusion:"neutral",  curiosity:"neutral", desire:"neutral",
    sadness:"sad",      grief:"sad",          disappointment:"sad",
    remorse:"sad",      embarrassment:"sad",
    surprise:"surprise"
};

// Speech labels → common 7  (ps = pleasant surprise)
const SPEECH_MAP = {
    angry:"angry", disgust:"disgust", fear:"fear",
    happy:"happy", neutral:"neutral", ps:"surprise", sad:"sad"
};

// Vote store — all must be non-null before verdict
const votes = { facial: null, text: null, speech: null };

const verdictBtn = document.getElementById("getFinalVerdictBtn");
verdictBtn.addEventListener("click", showFinalVerdict);

function normaliseEmotion(source, raw) {
    if (!raw) return null;
    const l = raw.toLowerCase().trim();
    if (source === 'facial') return FACIAL_MAP[raw] || l;
    if (source === 'text')   return TEXT_MAP[l]     || l;
    if (source === 'speech') return SPEECH_MAP[l]   || l;
    return l;
}

function computeFusion() {
    // Requires ALL THREE votes
    if (!votes.facial || !votes.text || !votes.speech) return null;

    const tally = {}, confSum = {};
    Object.values(votes).forEach(v => {
        tally[v.common]   = (tally[v.common]   || 0) + 1;
        confSum[v.common] = (confSum[v.common]  || 0) + v.confidence;
    });

    // Max votes; ties broken by total confidence
    const winner = Object.entries(tally).sort((a, b) =>
        b[1] !== a[1] ? b[1] - a[1] : confSum[b[0]] - confSum[a[0]]
    )[0][0];

    return { winner, tally };
}

function updateFusionVote(source, rawLabel, confidence) {
    const common = normaliseEmotion(source, rawLabel);
    votes[source] = { common, raw: rawLabel, confidence };

    // Update the individual vote card
    const IDS = {
        card:  { facial:'voteCardFacial',  text:'voteCardText',  speech:'voteCardSpeech'  },
        emoji: { facial:'vr-facial-emoji', text:'vr-text-emoji', speech:'vr-speech-emoji' },
        label: { facial:'vr-facial-label', text:'vr-text-label', speech:'vr-speech-label' },
        conf:  { facial:'vr-facial-conf',  text:'vr-text-conf',  speech:'vr-speech-conf'  }
    };
    document.getElementById(IDS.card[source]).classList.add('voted');
    document.getElementById(IDS.emoji[source]).textContent = COMMON_EMOJIS[common] || "❓";
    document.getElementById(IDS.label[source]).textContent = common || rawLabel || "—";
    document.getElementById(IDS.conf[source]).textContent  = confidence.toFixed(1) + "% conf";

    // Update progress bar
    const ready = Object.values(votes).filter(v => v !== null).length;
    document.getElementById("fusionProgressLabel").textContent = `${ready} / 3 models ready`;
    document.getElementById("fusionProgressFill").style.width  = (ready / 3 * 100) + "%";

    // Enable button only when all 3 are ready
    verdictBtn.disabled = (ready < 3);
    if (ready === 3) {
        verdictBtn.textContent = "🧠 Get Final Verdict ✓";
        toast("All 3 models ready — click Get Final Verdict!", "success");
    }
}

function showFinalVerdict() {
    const result = computeFusion();
    if (!result) {
        toast("Please run all 3 models first.", "error");
        return;
    }

    const { winner, tally } = result;
    const vc = document.getElementById("verdictCard");

    document.getElementById("verdictEmoji").textContent   = COMMON_EMOJIS[winner] || "🙂";
    document.getElementById("verdictEmotion").textContent = winner.charAt(0).toUpperCase() + winner.slice(1);

    const totalVotes   = Object.values(tally).reduce((s, c) => s + c, 0);
    const winnerVotes  = tally[winner];
    document.getElementById("verdictInfo").textContent =
        `${winnerVotes} of ${totalVotes} models agree`;

    document.getElementById("verdictTally").innerHTML = Object.entries(tally)
        .sort((a, b) => b[1] - a[1])
        .map(([em, cnt]) =>
            `<span class="tally-chip ${em === winner ? 'winner' : ''}">
               ${COMMON_EMOJIS[em] || ""} ${em} &times;${cnt}
             </span>`)
        .join("");

    // Pulse animation on verdict card
    vc.classList.remove("active");
    requestAnimationFrame(() => vc.classList.add("active"));

    // Scroll into view
    document.getElementById("fusionSection").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
