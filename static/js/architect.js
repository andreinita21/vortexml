/**
 * Vortex ML — Architecture Builder
 */

(function () {
    const archGrid = document.getElementById('arch-grid');
    const layerConfigSection = document.getElementById('layer-config-section');
    const hyperSection = document.getElementById('hyper-section');
    const previewSection = document.getElementById('preview-arch-section');
    const actionRow = document.getElementById('action-row');
    const layerList = document.getElementById('layer-list');
    const btnAddLayer = document.getElementById('btn-add-layer');
    const btnStartTraining = document.getElementById('btn-start-training');
    const canvas = document.getElementById('arch-preview-canvas');

    // Weight upload elements
    const weightUploadZone = document.getElementById('weight-upload-zone');
    const weightFileInput = document.getElementById('weight-file-input');
    const weightUploadStatus = document.getElementById('weight-upload-status');
    const weightStatusText = document.getElementById('weight-status-text');

    // Early stopping elements
    const esEnabledCheckbox = document.getElementById('inp-es-enabled');
    const esParams = document.getElementById('es-params');

    let architectures = [];
    let selectedArch = null;
    let layers = [128, 64];  // default layers
    let weightsLoaded = false;

    // ─── Early Stopping Toggle ───
    esEnabledCheckbox.addEventListener('change', () => {
        if (esEnabledCheckbox.checked) {
            esParams.classList.remove('hidden');
        } else {
            esParams.classList.add('hidden');
        }
    });

    // ─── Weight Upload ───
    weightUploadZone.addEventListener('click', () => weightFileInput.click());

    weightUploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        weightUploadZone.classList.add('drag-over');
    });

    weightUploadZone.addEventListener('dragleave', () => {
        weightUploadZone.classList.remove('drag-over');
    });

    weightUploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        weightUploadZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.pt')) {
            uploadWeightFile(file);
        } else {
            showToast('Only .pt weight files are supported', 'error');
        }
    });

    weightFileInput.addEventListener('change', () => {
        const file = weightFileInput.files[0];
        if (file) uploadWeightFile(file);
    });

    async function uploadWeightFile(file) {
        weightUploadZone.classList.add('uploading');
        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/weights/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (data.error) {
                showToast(data.error, 'error');
                weightUploadZone.classList.remove('uploading');
                return;
            }

            // Auto-fill architecture from parsed config
            const cfg = data.config;
            weightsLoaded = true;

            // Select architecture card
            selectedArch = cfg.arch_type;
            archGrid.querySelectorAll('.arch-card').forEach(c => c.classList.remove('selected'));
            const card = archGrid.querySelector(`.arch-card[data-key="${selectedArch}"]`);
            if (card) card.classList.add('selected');

            // Set layers
            layers = cfg.layer_sizes || [128, 64];

            // Show all sections
            layerConfigSection.classList.remove('hidden');
            hyperSection.classList.remove('hidden');
            previewSection.classList.remove('hidden');
            actionRow.classList.remove('hidden');

            // Fill hyperparameters
            document.getElementById('inp-epochs').value = cfg.epochs || 50;
            document.getElementById('inp-lr').value = cfg.lr || 0.001;
            document.getElementById('inp-batch').value = cfg.batch_size || 32;
            document.getElementById('inp-optimizer').value = cfg.optimizer || 'adam';
            document.getElementById('inp-activation').value = cfg.activation || 'relu';
            document.getElementById('inp-project-name').value = cfg.project_name || 'VortexProject';

            renderLayers();
            drawPreview();

            // Show upload success
            weightUploadZone.classList.add('hidden');
            weightUploadStatus.classList.remove('hidden');
            weightStatusText.textContent = `Loaded: ${file.name} — ${cfg.arch_type.toUpperCase()}, ${cfg.layer_sizes.join('→')} neurons`;

            showToast(`Weights loaded! Architecture auto-configured as ${cfg.arch_type.toUpperCase()}`, 'success');
        } catch (e) {
            showToast('Error uploading weights: ' + e.message, 'error');
        }
        weightUploadZone.classList.remove('uploading');
    }

    // ─── Load architectures ───
    async function loadArchitectures() {
        try {
            architectures = await apiGet('/api/architectures');
            renderArchGrid();
        } catch (e) {
            archGrid.innerHTML = '<p class="text-muted">Failed to load architectures.</p>';
        }
    }

    function renderArchGrid() {
        archGrid.innerHTML = architectures.map(a => `
            <div class="arch-card" data-key="${a.key}">
                <div class="arch-icon">${a.icon}</div>
                <div class="arch-name">${a.name}</div>
                <div class="arch-short">${a.short}</div>
                <div class="arch-desc">${a.desc}</div>
            </div>
        `).join('');

        archGrid.addEventListener('click', e => {
            const card = e.target.closest('.arch-card');
            if (!card) return;

            archGrid.querySelectorAll('.arch-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            selectedArch = card.dataset.key;

            layerConfigSection.classList.remove('hidden');
            hyperSection.classList.remove('hidden');
            previewSection.classList.remove('hidden');
            actionRow.classList.remove('hidden');

            renderLayers();
            drawPreview();
        });
    }

    // ─── Layer Configurator ───
    function renderLayers() {
        layerList.innerHTML = layers.map((neurons, i) => `
            <div class="layer-item" data-index="${i}">
                <span class="layer-label">Layer ${i + 1}</span>
                <input type="number" value="${neurons}" min="1" max="2048"
                       class="layer-neurons" data-index="${i}">
                <span class="text-muted" style="font-size:0.8rem">neurons</span>
                <button class="remove-layer" data-index="${i}" ${layers.length <= 1 ? 'disabled style="opacity:0.3"' : ''}>×</button>
            </div>
        `).join('');

        // Neuron change
        layerList.querySelectorAll('.layer-neurons').forEach(input => {
            input.addEventListener('change', () => {
                const i = parseInt(input.dataset.index);
                layers[i] = Math.max(1, Math.min(2048, parseInt(input.value) || 64));
                drawPreview();
            });
        });

        // Remove layer
        layerList.querySelectorAll('.remove-layer').forEach(btn => {
            btn.addEventListener('click', () => {
                if (layers.length <= 1) return;
                const i = parseInt(btn.dataset.index);
                layers.splice(i, 1);
                renderLayers();
                drawPreview();
            });
        });
    }

    btnAddLayer.addEventListener('click', () => {
        const last = layers[layers.length - 1] || 64;
        layers.push(Math.max(16, Math.floor(last / 2)));
        renderLayers();
        drawPreview();
    });

    // ─── Architecture Preview (Canvas) ───
    function drawPreview() {
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        const W = rect.width;
        const H = rect.height;

        ctx.clearRect(0, 0, W, H);

        // Build layer sizes array: [input, ...hidden, output]
        const allLayers = [4, ...layers, 2]; // placeholder input=4, output=2
        const numLayers = allLayers.length;
        const layerSpacing = W / (numLayers + 1);
        const maxNeurons = Math.max(...allLayers);
        const maxDisplay = 8; // max neurons to display per layer

        const positions = []; // positions[layerIdx] = [{x, y}, ...]

        for (let l = 0; l < numLayers; l++) {
            const x = layerSpacing * (l + 1);
            const n = Math.min(allLayers[l], maxDisplay);
            const neuronSpacing = Math.min(40, (H - 60) / (n + 1));
            const startY = H / 2 - (neuronSpacing * (n - 1)) / 2;
            const layerPositions = [];

            for (let i = 0; i < n; i++) {
                layerPositions.push({ x, y: startY + i * neuronSpacing });
            }
            positions.push(layerPositions);
        }

        // Draw connections
        for (let l = 0; l < positions.length - 1; l++) {
            for (const from of positions[l]) {
                for (const to of positions[l + 1]) {
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = 'rgba(99, 102, 241, 0.12)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        // Draw neurons
        for (let l = 0; l < positions.length; l++) {
            const isInput = l === 0;
            const isOutput = l === positions.length - 1;

            for (const pos of positions[l]) {
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);

                if (isInput) {
                    ctx.fillStyle = 'rgba(6, 182, 212, 0.6)';
                    ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
                } else if (isOutput) {
                    ctx.fillStyle = 'rgba(249, 115, 22, 0.6)';
                    ctx.strokeStyle = 'rgba(249, 115, 22, 0.8)';
                } else {
                    ctx.fillStyle = 'rgba(139, 92, 246, 0.5)';
                    ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
                }

                ctx.lineWidth = 2;
                ctx.fill();
                ctx.stroke();
            }

            // Label
            const x = positions[l][0].x;
            const bottomY = positions[l][positions[l].length - 1].y + 28;
            ctx.fillStyle = 'rgba(157, 157, 186, 0.8)';
            ctx.font = '11px "Space Grotesk", sans-serif';
            ctx.textAlign = 'center';

            if (isInput) {
                ctx.fillText('Input', x, bottomY);
            } else if (isOutput) {
                ctx.fillText('Output', x, bottomY);
            } else {
                ctx.fillText(`${allLayers[l]}n`, x, bottomY);
            }

            // Truncation indicator
            if (allLayers[l] > maxDisplay) {
                ctx.fillStyle = 'rgba(139, 92, 246, 0.6)';
                ctx.fillText('⋮', x, H / 2 + 4);
            }
        }
    }

    // Redraw on resize
    window.addEventListener('resize', () => {
        if (selectedArch) drawPreview();
    });

    // ─── Start Training ───
    btnStartTraining.addEventListener('click', async () => {
        if (!selectedArch) {
            showToast('Please select a network architecture.', 'error');
            return;
        }

        const esEnabled = esEnabledCheckbox.checked;
        const config = {
            arch_type: selectedArch,
            layer_sizes: layers.slice(),
            epochs: parseInt(document.getElementById('inp-epochs').value) || 50,
            lr: parseFloat(document.getElementById('inp-lr').value) || 0.001,
            batch_size: parseInt(document.getElementById('inp-batch').value) || 32,
            optimizer: document.getElementById('inp-optimizer').value,
            activation: document.getElementById('inp-activation').value,
            project_name: document.getElementById('inp-project-name').value || 'VortexProject',
            early_stopping: {
                enabled: esEnabled,
                patience: parseInt(document.getElementById('inp-es-patience').value) || 10,
                min_delta: parseFloat(document.getElementById('inp-es-delta').value) || 0.0001,
            },
        };

        try {
            btnStartTraining.disabled = true;
            btnStartTraining.textContent = 'Configuring...';

            const res = await apiPost('/api/model/configure', config);
            if (res.error) {
                showToast(res.error, 'error');
                btnStartTraining.disabled = false;
                btnStartTraining.textContent = 'Start Training ⚡';
                return;
            }

            showToast('Model configured! Redirecting to training...', 'success');
            setTimeout(() => window.location.href = '/training', 800);
        } catch (e) {
            showToast('Error: ' + e.message, 'error');
            btnStartTraining.disabled = false;
            btnStartTraining.textContent = 'Start Training ⚡';
        }
    });

    // ─── Check state ───
    (async function init() {
        await loadArchitectures();

        try {
            const state = await apiGet('/api/state');
            if (state.has_model && state.model_config) {
                const cfg = state.model_config;
                // Restore selection
                selectedArch = cfg.arch_type;
                layers = cfg.layer_sizes || [128, 64];

                const card = archGrid.querySelector(`.arch-card[data-key="${selectedArch}"]`);
                if (card) card.classList.add('selected');

                layerConfigSection.classList.remove('hidden');
                hyperSection.classList.remove('hidden');
                previewSection.classList.remove('hidden');
                actionRow.classList.remove('hidden');

                document.getElementById('inp-epochs').value = cfg.epochs || 50;
                document.getElementById('inp-lr').value = cfg.lr || 0.001;
                document.getElementById('inp-batch').value = cfg.batch_size || 32;
                document.getElementById('inp-optimizer').value = cfg.optimizer || 'adam';
                document.getElementById('inp-activation').value = cfg.activation || 'relu';
                document.getElementById('inp-project-name').value = cfg.project_name || 'VortexProject';

                // Restore early stopping
                if (cfg.early_stopping && cfg.early_stopping.enabled) {
                    esEnabledCheckbox.checked = true;
                    esParams.classList.remove('hidden');
                    document.getElementById('inp-es-patience').value = cfg.early_stopping.patience || 10;
                    document.getElementById('inp-es-delta').value = cfg.early_stopping.min_delta || 0.0001;
                }

                renderLayers();
                drawPreview();
            }
        } catch (e) {
            // ignore
        }
    })();
})();
