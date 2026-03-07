/**
 * admin.js
 * Handles the logic for the Aeon Forge Admin Dashboard
 */

// ─── State ─────────────────────────────────────────────────────────────────
let currentSessionId = null;
let facesData = [];
let chartInstance = null;
let adminToken = localStorage.getItem("forge_admin_token") || null;

const API_BASE = "http://localhost:8000/api";

// ─── DOM Elements ──────────────────────────────────────────────────────────
const elGate = document.getElementById("sessionGate");
const elApp = document.getElementById("dashboardApp");
const elLoginGate = document.getElementById("loginGate");
const elSessionsList = document.getElementById("sessionsList");
const elSessionsLoading = document.getElementById("sessionsLoading");
const elSessionPill = document.getElementById("activeSessionPill");
const elLogoutBtn = document.getElementById("btnLogout");

// Metrics Map
const elStatFaces = document.getElementById("statTotalFaces");
const elStatEmbeds = document.getElementById("statTotalEmbeddings");
const elStatAge = document.getElementById("statSessionAge");

// Table
const elFacesTableBody = document.getElementById("facesTableBody");

// Modal
const elMergeModal = document.getElementById("mergeModal");
const elMergeTargetSelect = document.getElementById("mergeTargetSelect");

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // Nav bindings
    document.querySelectorAll(".nav-item[data-target]").forEach(btn => {
        btn.addEventListener("click", (e) => switchView(e.currentTarget));
    });

    // Action bindings
    document.getElementById("navBack").addEventListener("click", () => window.location.href = "/");
    document.getElementById("btnCreateNew").addEventListener("click", createNewSession);
    document.getElementById("btnDeleteSession").addEventListener("click", deleteActiveSession);
    document.getElementById("btnRefreshFaces").addEventListener("click", loadIdentities);

    // Auth bindings
    document.getElementById("btnLogin").addEventListener("click", login);
    document.getElementById("adminPassword").addEventListener("keypress", (e) => {
        if (e.key === "Enter") login();
    });
    elLogoutBtn.addEventListener("click", logout);

    // Merge Modal bindings
    document.getElementById("btnCancelMerge").addEventListener("click", () => {
        elMergeModal.classList.add("hidden");
    });
    document.getElementById("btnConfirmMerge").addEventListener("click", executeMerge);

    // Initial load: Check auth then fetch sessions
    checkAuth();
});

// ─── Navigation ────────────────────────────────────────────────────────────
function switchView(selectedBtn) {
    // Update active class on nav
    document.querySelectorAll(".nav-item").forEach(btn => btn.classList.remove("active"));
    selectedBtn.classList.add("active");

    // Hide all views, show targeted
    const targetId = selectedBtn.getAttribute("data-target");
    document.querySelectorAll(".admin-view").forEach(view => {
        view.classList.add("hidden");
        view.classList.remove("active-view");
    });

    const targetView = document.getElementById(targetId);
    if (targetView) {
        targetView.classList.remove("hidden");
        targetView.classList.add("active-view");
    }

    // Trigger data load based on view
    if (targetId === "view-overview") loadMetrics();
    if (targetId === "view-identities") loadIdentities();
    if (targetId === "view-map") loadPCAChart();
}

// ─── API Helpers ───────────────────────────────────────────────────────────
async function secureFetch(url, options = {}) {
    try {
        // Add auth token if available
        if (adminToken) {
            options.headers = {
                ...options.headers,
                'X-Admin-Token': adminToken
            };
        }

        const res = await fetch(url, options);
        if (res.status === 401) {
            // Token expired or invalid
            logout();
            throw new Error("Unauthorized");
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error(`Fetch error ${url}:`, err);
        return { error: err.message };
    }
}

// ─── Authentication ───────────────────────────────────────────────────────

function checkAuth() {
    if (adminToken) {
        showSessionGate();
    } else {
        showLoginGate();
    }
}

async function login() {
    const passwordInput = document.getElementById("adminPassword");
    const errorMsg = document.getElementById("loginError");
    const password = passwordInput.value;

    errorMsg.classList.add("hidden");

    try {
        const res = await fetch(`${API_BASE}/admin/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password })
        });

        const data = await res.json();
        if (res.ok && data.status === "success") {
            adminToken = data.token;
            localStorage.setItem("forge_admin_token", adminToken);
            showSessionGate();
            passwordInput.value = "";
        } else {
            errorMsg.classList.remove("hidden");
        }
    } catch (err) {
        console.error("Login error:", err);
        errorMsg.classList.remove("hidden");
    }
}

function logout() {
    adminToken = null;
    localStorage.removeItem("forge_admin_token");
    currentSessionId = null;
    showLoginGate();
}

function showLoginGate() {
    elLoginGate.classList.remove("hidden");
    elGate.classList.add("hidden");
    elApp.classList.add("hidden");
    elLogoutBtn.classList.add("hidden");
}

function showSessionGate() {
    elLoginGate.classList.add("hidden");
    elGate.classList.remove("hidden");
    elApp.classList.add("hidden");
    elLogoutBtn.classList.remove("hidden");
    fetchSessionsList();
}

// ─── Session Management (Gatekeeper) ───────────────────────────────────────

async function fetchSessionsList() {
    elSessionsLoading.classList.remove("hidden");
    elSessionsList.innerHTML = "";

    const data = await secureFetch(`${API_BASE}/admin/sessions`);
    elSessionsLoading.classList.add("hidden");

    if (data.sessions && data.sessions.length > 0) {
        data.sessions.forEach(session => {
            const date = new Date(session.updated_at * 1000).toLocaleString();

            const card = document.createElement("div");
            card.className = `session-card ${session.is_active ? 'active-card' : ''}`;
            card.innerHTML = `
                <div class="session-info">
                    <strong>${session.id.split('-')[0]}...</strong>
                    <small>Last active: ${date}</small>
                </div>
                ${session.is_active ? '<span class="status-pill">Active</span>' : ''}
            `;

            card.addEventListener("click", () => activateSession(session.id));
            elSessionsList.appendChild(card);
        });
    } else {
        elSessionsList.innerHTML = "<p style='color:#8892b0'>No historical sessions found.</p>";
    }
}

async function createNewSession() {
    const data = await secureFetch(`${API_BASE}/session/new`);
    if (data.status === "success") {
        activateSession(data.session_id);
    }
}

async function activateSession(sessionId) {
    // Ensure backend has it active
    const data = await secureFetch(`${API_BASE}/session/load/${sessionId}`);
    if (data.status === "success" || data.status === "ok") {
        // Unlock Dashboard
        currentSessionId = sessionId;
        elSessionPill.innerText = `Active: ${sessionId.split('-')[0]}`;

        elGate.classList.add("hidden");
        elApp.classList.remove("hidden");

        // Load initial data
        loadMetrics();
    } else {
        alert("Failed to load session data.");
    }
}

async function deleteActiveSession() {
    if (!confirm("Are you sure? This will permanently delete all faces and embeddings in this session.")) return;

    const data = await secureFetch(`${API_BASE}/session/${currentSessionId}`, { method: 'DELETE' });
    if (data.status === "success") {
        // Return to gatekeeper
        currentSessionId = null;
        elApp.classList.add('hidden');
        elGate.classList.remove('hidden');
        fetchSessionsList();
    }
}

// ─── Metrics View ──────────────────────────────────────────────────────────

async function loadMetrics() {
    if (!currentSessionId) return;

    // Also load the DB info from faces
    const facesDataTemp = await secureFetch(`${API_BASE}/admin/session/${currentSessionId}/faces`);
    if (facesDataTemp.faces) {
        facesData = facesDataTemp.faces;
    }

    const data = await secureFetch(`${API_BASE}/admin/session/${currentSessionId}/metrics`);

    if (!data.error) {
        elStatFaces.innerText = data.total_faces;
        elStatEmbeds.innerText = data.total_embeddings;
        elStatAge.innerText = "Active"; // You can enhance this with actual duration logic if desired
    }
}

// ─── Identities View ───────────────────────────────────────────────────────

async function loadIdentities() {
    if (!currentSessionId) return;

    elFacesTableBody.innerHTML = "<tr><td colspan='5' style='text-align:center'>Loading Identities...</td></tr>";

    const data = await secureFetch(`${API_BASE}/admin/session/${currentSessionId}/faces`);

    if (data.faces) {
        facesData = data.faces; // Update local state
        renderFacesTable();
    }
}

function renderFacesTable() {
    elFacesTableBody.innerHTML = "";

    if (facesData.length === 0) {
        elFacesTableBody.innerHTML = "<tr><td colspan='5' style='text-align:center'>No identities recorded in this session yet.</td></tr>";
        return;
    }

    facesData.forEach(face => {
        const tr = document.createElement("tr");

        // Determine Thumbnail
        let thumbHtml = `<div class="no-img">No Image</div>`;
        if (face.images && face.images.length > 0) {
            // Using the primary image if it exists, else the first expansion
            // The route maps to the StaticFiles mount /sessions
            const imgPath = `http://localhost:8000/sessions/${currentSessionId}/faces/${face.reid}/${face.images[0]}`;
            thumbHtml = `<img src="${imgPath}" alt="Face Crop" loading="lazy" />`;
        }

        tr.innerHTML = `
            <td class="thumbnail-cell">${thumbHtml}</td>
            <td><strong>#${face.reid}</strong></td>
            <td>
                <span id="name-display-${face.reid}">${face.name}</span>
                <input type="text" id="name-input-${face.reid}" class="inline-edit-input hidden" value="${face.name}">
            </td>
            <td><small>Has ${face.images ? face.images.length : 0} visual variants recorded</small></td>
            <td class="action-btns">
                <button class="btn secondary" onclick="toggleEdit(${face.reid})" id="btn-edit-${face.reid}">Rename</button>
                <button class="btn secondary" onclick="openMergeModal(${face.reid}, '${face.name}')">Merge</button>
                <button class="btn danger" onclick="deleteFace(${face.reid})">Delete</button>
            </td>
        `;
        elFacesTableBody.appendChild(tr);
    });
}

// Rename Logic
window.toggleEdit = async function (reid) {
    const displayEl = document.getElementById(`name-display-${reid}`);
    const inputEl = document.getElementById(`name-input-${reid}`);
    const btnEl = document.getElementById(`btn-edit-${reid}`);

    if (inputEl.classList.contains("hidden")) {
        // Switch to Edit Mode
        displayEl.classList.add("hidden");
        inputEl.classList.remove("hidden");
        inputEl.focus();
        btnEl.innerText = "Save";
        btnEl.classList.add("primary");
        btnEl.classList.remove("secondary");
    } else {
        // Save Mode
        const newName = inputEl.value.trim();

        // Optimistic UI Update
        displayEl.innerText = newName;
        displayEl.classList.remove("hidden");
        inputEl.classList.add("hidden");
        btnEl.innerText = "Rename";
        btnEl.classList.add("secondary");
        btnEl.classList.remove("primary");

        // Fire Backend Update
        await fetch(`${API_BASE}/face/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reid: reid, name: newName })
        });

        // Update local state quietly
        const face = facesData.find(f => f.reid === reid);
        if (face) face.name = newName;
    }
};

// Delete Logic
window.deleteFace = async function (reid) {
    if (!confirm(`Are you sure you want to completely erase ReID #${reid} from memory?`)) return;

    const res = await secureFetch(`${API_BASE}/admin/faces/${reid}`, { method: 'DELETE' });
    if (res.status === "success") {
        // Locally remove and re-render
        facesData = facesData.filter(f => f.reid !== reid);
        renderFacesTable();

        // If chart is active, refresh it
        if (chartInstance) loadPCAChart();
    }
}

// ─── Merge Modal Logic ─────────────────────────────────────────────────────

let mergeFromIdState = null;

window.openMergeModal = function (fromReid, fromName) {
    mergeFromIdState = fromReid;
    document.getElementById("mergeFromName").innerText = fromName;
    document.getElementById("mergeFromId").innerText = fromReid;

    // Populate Select Options (excluding the one we are merging from)
    elMergeTargetSelect.innerHTML = "";
    const options = facesData.filter(f => f.reid !== fromReid);

    if (options.length === 0) {
        alert("There are no other identities to merge into!");
        return;
    }

    options.forEach(f => {
        const opt = document.createElement("option");
        opt.value = f.reid;
        opt.text = `${f.name} (ReID ${f.reid})`;
        elMergeTargetSelect.appendChild(opt);
    });

    elMergeModal.classList.remove("hidden");
};

async function executeMerge() {
    const targetReid = elMergeTargetSelect.value;
    if (!targetReid || !mergeFromIdState) return;

    const res = await fetch(`${API_BASE}/admin/faces/merge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ merge_from: mergeFromIdState, merge_to: targetReid })
    });

    const data = await res.json();
    if (data.status === "success") {
        elMergeModal.classList.add("hidden");
        loadIdentities(); // Full refresh
    } else {
        alert("Merge failed.");
    }
}


// ─── 2D Map (PCA Chart) ────────────────────────────────────────────────────

async function loadPCAChart() {
    if (!currentSessionId) return;

    const data = await secureFetch(`${API_BASE}/admin/session/${currentSessionId}/map`);

    if (data.points) {
        renderChart(data.points);
    }
}

function renderChart(points) {
    const ctx = document.getElementById('pcaChart').getContext('2d');

    // Group points by ReID to give them different colors using Chart.js datasets
    const grouped = {};
    points.forEach(p => {
        if (!grouped[p.reid]) {
            grouped[p.reid] = {
                label: p.name,
                data: [],
                pointRadius: 6,
                pointHoverRadius: 9,
                // Generate consistent color based on reid
                backgroundColor: `hsl(${(p.reid * 137.5) % 360}, 70%, 60%)`,
                borderColor: `hsl(${(p.reid * 137.5) % 360}, 70%, 40%)`,
            };
        }
        grouped[p.reid].data.push({ x: p.x, y: p.y, meta: p });
    });

    const datasets = Object.values(grouped);

    if (chartInstance) {
        chartInstance.destroy();
    }

    // Modern Chart.js config mapping perfectly to the dark mode UI
    Chart.defaults.color = '#8892b0';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    chartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#f0f0f4', padding: 20 }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 17, 21, 0.9)',
                    titleFont: { size: 14 },
                    padding: 12,
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function (context) {
                            return `${context.dataset.label} (ReID: ${context.raw.meta.reid})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'Principal Component 1' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: { display: true, text: 'Principal Component 2' }
                }
            },
            animation: {
                duration: 800,
                easing: 'easeOutQuart'
            }
        }
    });
}
