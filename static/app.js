const startSessionButtons = [
  document.getElementById("startBtn"),
  document.getElementById("startBtnSecondary"),
];
const startCameraButtons = [
  document.getElementById("heroStartBtn"),
];
const stopCameraButtons = [
  document.getElementById("stopBtn"),
  document.getElementById("stopBtnSecondary"),
];
const endSessionButtons = [
  document.getElementById("endSessionBtn"),
  document.getElementById("endSessionBtnSecondary"),
];

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const sessionId = document.getElementById("sessionId");
const videoFeed = document.getElementById("videoFeed");
const videoWrap = document.getElementById("videoWrap");
const captionText = document.getElementById("captionText");
const objectList = document.getElementById("objectList");
const objectDescription = document.getElementById("objectDescription");
const sessionList = document.getElementById("sessionList");

let sessionActive = false;
let cameraActive = false;
let currentSessionId = "";
let pollingInterval = null;

function setActiveState(isSessionActive, isCameraActive, session) {
  sessionActive = isSessionActive;
  cameraActive = isCameraActive;
  currentSessionId = session || "";
  statusDot.style.background = isCameraActive ? "#10b981" : (isSessionActive ? "#f59e0b" : "#9ca3af");
  statusText.textContent = isCameraActive
    ? "Live — camera on"
    : (isSessionActive ? "Session active — camera off" : "Idle — camera off");
  sessionId.textContent = session ? `Session: ${session}` : "";

  if (isCameraActive) {
    videoFeed.style.display = "block";
    videoFeed.src = "/stream";
    videoWrap.querySelector(".video-placeholder").style.display = "none";
  } else {
    videoFeed.style.display = "none";
    videoFeed.src = "";
    videoWrap.querySelector(".video-placeholder").style.display = "block";
    if (!isSessionActive) {
      captionText.textContent = "Start a session to see the description.";
    }
  }
}

async function startSession() {
  const res = await fetch("/session/start", { method: "POST" });
  const data = await res.json();
  setActiveState(data.session_active, data.camera_active, data.session_id);
  startPolling();
}

async function endSession() {
  await fetch("/session/end", { method: "POST" });
  setActiveState(false, false, "");
  objectList.innerHTML = "";
  objectDescription.textContent = "Select an object to describe it.";
  stopPolling();
}

async function startCamera() {
  const res = await fetch("/camera/start", { method: "POST" });
  const data = await res.json();
  setActiveState(data.session_active, data.camera_active, data.session_id);
  startPolling();
}

async function stopCamera() {
  const res = await fetch("/camera/stop", { method: "POST" });
  const data = await res.json();
  setActiveState(data.session_active, data.camera_active, data.session_id);
}

async function pollStatus() {
  const res = await fetch("/status");
  const data = await res.json();
  setActiveState(data.session_active, data.camera_active, data.session_id);
  if (data.caption) {
    captionText.textContent = data.caption;
  } else if (data.session_active) {
    captionText.textContent = "Waiting for description...";
  } else {
    captionText.textContent = "Start a session to see the description.";
  }
}

async function pollObjects() {
  if (!sessionActive) return;
  const res = await fetch("/objects");
  const data = await res.json();
  renderObjects(data.objects || []);
}

function startPolling() {
  if (pollingInterval) return;
  pollStatus();
  pollObjects();
  pollSessions();
  pollingInterval = setInterval(() => {
    pollStatus();
    pollObjects();
    pollSessions();
  }, 1500);
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
}

async function pollSessions() {
  const res = await fetch("/sessions");
  const data = await res.json();
  renderSessions(data.sessions || []);
}

function renderSessions(sessions) {
  if (!sessionList) return;
  sessionList.innerHTML = "";
  if (!sessions.length) {
    sessionList.innerHTML = "<span class='object-detail-text'>No sessions yet.</span>";
    return;
  }

  sessions.forEach((session) => {
    const pill = document.createElement("div");
    pill.className = "session-pill" + (session.session_id === currentSessionId ? " active" : "");
    pill.textContent = `Session ${session.number}`;
    sessionList.appendChild(pill);
  });
}

function renderObjects(objects) {
  objectList.innerHTML = "";
  if (!objects.length) {
    objectList.innerHTML = "<div class='object-detail-text'>No objects detected yet.</div>";
    return;
  }

  objects.forEach((obj) => {
    const card = document.createElement("div");
    card.className = "object-card";
    card.innerHTML = `
      <div class="object-name">${obj.name}</div>
      <div class="object-meta">Count: ${obj.count}</div>
      <div class="object-meta">Last seen: ${obj.last_seen}</div>
    `;

    card.addEventListener("click", async () => {
      objectDescription.textContent = "Generating description...";
      const res = await fetch(`/describe?object=${encodeURIComponent(obj.name)}`);
      const data = await res.json();
      objectDescription.textContent = data.description || "No description available.";
    });

    objectList.appendChild(card);
  });
}

startSessionButtons.forEach((btn) => btn && btn.addEventListener("click", startSession));
startCameraButtons.forEach((btn) => btn && btn.addEventListener("click", startCamera));
stopCameraButtons.forEach((btn) => btn && btn.addEventListener("click", stopCamera));
endSessionButtons.forEach((btn) => btn && btn.addEventListener("click", endSession));

window.addEventListener("beforeunload", stopPolling);
