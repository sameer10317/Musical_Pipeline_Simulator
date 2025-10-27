const WS_URL = 'ws://localhost:8765';
let ws = null;
let audioEngine = null;
let currentState = null;

// Audio Engine
class AudioEngine {
    constructor() {
        this.audioContext = null;
        this.initialized = false;
    }

    async init() {
        if (this.initialized) return;
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.initialized = true;
    }

    playNote(frequency, duration = 0.5, volume = 0.6) {  // Changed from 0.3 to 0.6
        if (!this.initialized) return;
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';
        gainNode.gain.setValueAtTime(volume, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration);
    }

    getFrequency(note) {
        const noteFrequencies = {
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
            'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
            'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99,
            'A5': 880.00, 'B5': 987.77
        };
        return noteFrequencies[note] || 440;
    }
}

// WebSocket functions
function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('Connected to backend');
        updateConnectionStatus(true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleBackendEvent(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus(false);
    };

    ws.onclose = () => {
        console.log('Disconnected from backend');
        updateConnectionStatus(false);
        setTimeout(connectWebSocket, 3000);
    };
}

function sendCommand(command, params = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command, params }));
    }
}

function updateConnectionStatus(connected) {
    const status = document.getElementById('connectionStatus');
    status.className = 'connection-status ' + (connected ? 'connected' : 'disconnected');
    status.textContent = connected ? '🟢 Connected' : '⚫ Disconnected';
}

// Event handlers
function handleBackendEvent(event) {
    console.log('Event:', event);

    switch (event.event_type) {
        case 'note_play':
            playNoteFromEvent(event.data);
            break;
        case 'cycle_tick':
            updateUI(event.data);
            break;
        case 'instruction_completed':
            console.log('Instruction completed:', event.data);
            break;
        case 'simulation_complete':
            console.log('Simulation complete!', event.data);
            updateButtons(false, false);
            break;
    }
}

async function playNoteFromEvent(data) {
    if (!audioEngine) {
        audioEngine = new AudioEngine();
        await audioEngine.init();
    }
    const freq = audioEngine.getFrequency(data.note);
    const duration = data.duration || 0.5;
    audioEngine.playNote(freq, duration, 0.9);  // Changed from 0.2 to 0.6
}

function updateUI(state) {
    currentState = state;
    document.getElementById('cycleCount').textContent = state.cycle;

    const stats = state.statistics;
    document.getElementById('completedCount').textContent = stats.instructions_completed;
    document.getElementById('stallCount').textContent = stats.total_stalls;
    document.getElementById('flushCount').textContent = stats.total_flushes;
    document.getElementById('efficiency').textContent = stats.efficiency + '%';

    renderPipelines(state.pipelines);

    const totalInstructions = stats.instructions_completed + state.instructions_remaining;
    const progress = totalInstructions > 0 ? (stats.instructions_completed / totalInstructions) * 100 : 0;
    const progressBar = document.getElementById('timelineProgress');
    progressBar.style.width = progress + '%';
    progressBar.textContent = Math.round(progress) + '%';
}

function renderPipelines(pipelines) {
    const container = document.getElementById('pipelinesContainer');
    if (container.children.length === 0) {
        createPipelineUI(pipelines.length);
    }

    pipelines.forEach((pipeline, pIdx) => {
        pipeline.forEach((instr, sIdx) => {
            const stageContent = document.getElementById(`stage-content-${pIdx}-${sIdx}`);
            if (instr) {
                const classes = ['instruction', `inst-${instr.id % 8}`];
                if (instr.stalled) classes.push('stalled');
                stageContent.innerHTML = `<div class="${classes.join(' ')}"><div>${instr.name}</div><small>${instr.note}</small></div>`;
            } else {
                stageContent.innerHTML = '';
            }
        });
    });
}

function createPipelineUI(numPipelines) {
    const container = document.getElementById('pipelinesContainer');
    container.innerHTML = '';
    const stages = ['IF', 'ID', 'EX', 'MEM', 'WB'];
    const stageNames = ['Instruction Fetch', 'Instruction Decode', 'Execute', 'Memory Access', 'Write Back'];

    for (let p = 0; p < numPipelines; p++) {
        const pipelineDiv = document.createElement('div');
        pipelineDiv.className = 'pipeline-container';
        pipelineDiv.innerHTML = `
            <div class="pipeline-title">
                ${numPipelines > 1 ? 'Superscalar Pipeline' : 'Classic 5-Stage Pipeline'}
                ${numPipelines > 1 ? `<span class="pipeline-type-badge">Pipeline ${p + 1}</span>` : ''}
            </div>
            <div class="pipeline" id="pipeline-${p}">
                ${stages.map((stage, i) => `
                    <div class="stage" id="stage-${p}-${i}">
                        <div class="stage-header">${stage}<br><small>${stageNames[i]}</small></div>
                        <div class="stage-content" id="stage-content-${p}-${i}"></div>
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(pipelineDiv);
    }
}

function updateButtons(running, paused) {
    document.getElementById('startBtn').disabled = running && !paused;
    document.getElementById('pauseBtn').disabled = !running || paused;
    document.getElementById('stallBtn').disabled = !running || paused;
    document.getElementById('flushBtn').disabled = !running || paused;
}

// Event listeners
document.getElementById('startBtn').addEventListener('click', async () => {
    if (!audioEngine) {
        audioEngine = new AudioEngine();
        await audioEngine.init();
    }
    sendCommand('start', {
        melody: document.getElementById('melodySelect').value,
        speed: parseFloat(document.getElementById('speedSelect').value),
        pipeline_type: document.getElementById('pipelineType').value
    });
    updateButtons(true, false);
});

document.getElementById('pauseBtn').addEventListener('click', () => {
    sendCommand('pause');
    updateButtons(false, true);
});

document.getElementById('resetBtn').addEventListener('click', () => {
    sendCommand('reset');
    updateButtons(false, false);
    document.getElementById('cycleCount').textContent = '0';
    document.getElementById('completedCount').textContent = '0';
    document.getElementById('stallCount').textContent = '0';
    document.getElementById('flushCount').textContent = '0';
    document.getElementById('efficiency').textContent = '100%';
    document.getElementById('pipelinesContainer').innerHTML = '';
});

document.getElementById('stallBtn').addEventListener('click', () => {
    sendCommand('stall');
});

document.getElementById('flushBtn').addEventListener('click', () => {
    sendCommand('flush');
});

// Initialize
connectWebSocket();

