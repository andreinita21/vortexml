/**
 * Vortex ML — Training Dashboard
 */

(function () {
    const socket = io();

    // DOM refs
    const statEpoch = document.getElementById('stat-epoch');
    const statLoss = document.getElementById('stat-loss');
    const statAcc = document.getElementById('stat-acc');
    const statEta = document.getElementById('stat-eta');
    const progressBar = document.getElementById('training-progress');
    const progressPct = document.getElementById('progress-pct');
    const trainingLog = document.getElementById('training-log');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnSaveWeights = document.getElementById('btn-save-weights');
    const esBadge = document.getElementById('es-badge');
    const chartCanvas = document.getElementById('training-chart');
    const networkCanvas = document.getElementById('network-canvas');

    let chart = null;
    let isTraining = false;
    let modelConfig = null;
    let lastWeightFilename = null;

    // ─── Chart Setup ───
    function initChart() {
        const ctx = chartCanvas.getContext('2d');
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Train Loss',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                    },
                    {
                        label: 'Val Loss',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.05)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                    },
                    {
                        label: 'Val Accuracy',
                        data: [],
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.05)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        yAxisID: 'y1',
                        hidden: true,  // reveal when we get accuracy data
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        labels: {
                            color: '#9d9dba',
                            font: { family: "'Space Grotesk', sans-serif", size: 12 },
                        },
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true, text: 'Epoch', color: '#9d9dba',
                            font: { family: "'Space Grotesk'" }
                        },
                        ticks: { color: '#5a5a7a', font: { family: "'JetBrains Mono'", size: 10 } },
                        grid: { color: 'rgba(100,100,200,0.07)' },
                    },
                    y: {
                        title: {
                            display: true, text: 'Loss', color: '#9d9dba',
                            font: { family: "'Space Grotesk'" }
                        },
                        ticks: { color: '#5a5a7a', font: { family: "'JetBrains Mono'", size: 10 } },
                        grid: { color: 'rgba(100,100,200,0.07)' },
                        position: 'left',
                    },
                    y1: {
                        title: {
                            display: true, text: 'Accuracy %', color: '#9d9dba',
                            font: { family: "'Space Grotesk'" }
                        },
                        ticks: { color: '#5a5a7a', font: { family: "'JetBrains Mono'", size: 10 } },
                        grid: { drawOnChartArea: false },
                        position: 'right',
                        min: 0,
                        max: 100,
                    },
                },
            },
        });
    }

    // ─── Network Visualization ───
    function drawNetwork(layerSizes, activeEpoch) {
        const ctx = networkCanvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = networkCanvas.getBoundingClientRect();
        networkCanvas.width = rect.width * dpr;
        networkCanvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        const W = rect.width;
        const H = rect.height;

        ctx.clearRect(0, 0, W, H);

        if (!layerSizes || layerSizes.length === 0) return;

        const numLayers = layerSizes.length;
        const layerSpacing = W / (numLayers + 1);
        const maxDisplay = 8;

        const positions = [];
        for (let l = 0; l < numLayers; l++) {
            const x = layerSpacing * (l + 1);
            const n = Math.min(layerSizes[l], maxDisplay);
            const spacing = Math.min(36, (H - 80) / (n + 1));
            const startY = H / 2 - (spacing * (n - 1)) / 2;
            const lp = [];
            for (let i = 0; i < n; i++) {
                lp.push({ x, y: startY + i * spacing });
            }
            positions.push(lp);
        }

        // Animated connections
        const pulse = (Math.sin(Date.now() / 400) + 1) / 2;
        for (let l = 0; l < positions.length - 1; l++) {
            for (const from of positions[l]) {
                for (const to of positions[l + 1]) {
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    const alpha = isTraining ? 0.06 + pulse * 0.08 : 0.08;
                    ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        // Draw data flow particles when training
        if (isTraining) {
            const t = (Date.now() % 2000) / 2000;
            for (let l = 0; l < positions.length - 1; l++) {
                const fromLayer = positions[l];
                const toLayer = positions[l + 1];
                // Draw a few particles
                for (let p = 0; p < 3; p++) {
                    const fi = (p * 37 + l * 13) % fromLayer.length;
                    const ti = (p * 23 + l * 7) % toLayer.length;
                    const from = fromLayer[fi];
                    const to = toLayer[ti];
                    const pt = (t + p * 0.33 + l * 0.1) % 1;
                    const px = from.x + (to.x - from.x) * pt;
                    const py = from.y + (to.y - from.y) * pt;

                    ctx.beginPath();
                    ctx.arc(px, py, 3, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(139, 92, 246, ${0.4 + pulse * 0.4})`;
                    ctx.fill();
                }
            }
        }

        // Draw neurons
        for (let l = 0; l < positions.length; l++) {
            const isInput = l === 0;
            const isOutput = l === positions.length - 1;

            for (const pos of positions[l]) {
                // Glow effect when training
                if (isTraining) {
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 16, 0, Math.PI * 2);
                    const g = ctx.createRadialGradient(pos.x, pos.y, 4, pos.x, pos.y, 16);
                    if (isInput) {
                        g.addColorStop(0, `rgba(6, 182, 212, ${0.2 + pulse * 0.15})`);
                    } else if (isOutput) {
                        g.addColorStop(0, `rgba(249, 115, 22, ${0.2 + pulse * 0.15})`);
                    } else {
                        g.addColorStop(0, `rgba(139, 92, 246, ${0.2 + pulse * 0.15})`);
                    }
                    g.addColorStop(1, 'transparent');
                    ctx.fillStyle = g;
                    ctx.fill();
                }

                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);

                if (isInput) {
                    ctx.fillStyle = 'rgba(6, 182, 212, 0.7)';
                    ctx.strokeStyle = 'rgba(6, 182, 212, 0.9)';
                } else if (isOutput) {
                    ctx.fillStyle = 'rgba(249, 115, 22, 0.7)';
                    ctx.strokeStyle = 'rgba(249, 115, 22, 0.9)';
                } else {
                    ctx.fillStyle = 'rgba(139, 92, 246, 0.6)';
                    ctx.strokeStyle = 'rgba(139, 92, 246, 0.9)';
                }

                ctx.lineWidth = 2;
                ctx.fill();
                ctx.stroke();
            }

            // Labels
            const x = positions[l][0].x;
            const bottomY = positions[l][positions[l].length - 1].y + 24;
            ctx.fillStyle = 'rgba(157, 157, 186, 0.7)';
            ctx.font = '10px "Space Grotesk", sans-serif';
            ctx.textAlign = 'center';
            if (isInput) ctx.fillText('Input', x, bottomY);
            else if (isOutput) ctx.fillText('Output', x, bottomY);
            else ctx.fillText(`${layerSizes[l]}n`, x, bottomY);

            if (layerSizes[l] > maxDisplay) {
                ctx.fillStyle = 'rgba(139, 92, 246, 0.5)';
                ctx.fillText('⋮', x, H / 2 + 3);
            }
        }
    }

    let animFrame = null;
    function startNetworkAnimation(layerSizes) {
        function animate() {
            drawNetwork(layerSizes);
            animFrame = requestAnimationFrame(animate);
        }
        animate();
    }

    function stopNetworkAnimation() {
        if (animFrame) cancelAnimationFrame(animFrame);
    }

    // ─── Socket.IO Events ───
    socket.on('training_update', (data) => {
        const { epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, eta_seconds } = data;

        // Stats
        statEpoch.textContent = `${epoch}/${total_epochs}`;
        statLoss.textContent = train_loss.toFixed(4);
        statEta.textContent = formatTime(eta_seconds);

        if (val_acc !== null && val_acc !== undefined) {
            statAcc.textContent = val_acc.toFixed(1) + '%';
            // Unhide accuracy dataset
            if (chart.data.datasets[2].hidden) {
                chart.data.datasets[2].hidden = false;
            }
        } else {
            statAcc.textContent = val_loss.toFixed(4);
            document.querySelector('.stat-card:nth-child(3) .stat-label').textContent = 'Val Loss';
        }

        // Progress
        const pct = Math.round((epoch / total_epochs) * 100);
        progressBar.style.width = pct + '%';
        progressPct.textContent = pct + '%';

        // Early stopping badge
        if (data.es_patience !== undefined) {
            esBadge.classList.remove('hidden');
            esBadge.title = `Early Stopping: ${data.es_patience_counter}/${data.es_patience} patience`;
        }

        // Chart
        chart.data.labels.push(epoch.toString());
        chart.data.datasets[0].data.push(train_loss);
        chart.data.datasets[1].data.push(val_loss);
        chart.data.datasets[2].data.push(val_acc);
        chart.update('none');

        // Log
        let logLine = `<span class="log-epoch">Epoch ${epoch}/${total_epochs}</span>`;
        logLine += ` — <span class="log-loss">loss: ${train_loss.toFixed(4)}</span>`;
        logLine += ` · <span class="log-loss">val_loss: ${val_loss.toFixed(4)}</span>`;
        if (val_acc !== null && val_acc !== undefined) {
            logLine += ` · <span class="log-acc">acc: ${val_acc.toFixed(1)}%</span>`;
        }
        if (data.es_patience !== undefined) {
            logLine += ` · <span class="log-es">patience: ${data.es_patience_counter}/${data.es_patience}</span>`;
        }
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = logLine;
        trainingLog.appendChild(entry);
        trainingLog.scrollTop = trainingLog.scrollHeight;
    });

    socket.on('training_complete', (data) => {
        isTraining = false;
        stopNetworkAnimation();
        btnStop.classList.add('hidden');
        btnStart.classList.remove('hidden');
        btnStart.textContent = '✓ Complete';
        btnStart.disabled = true;

        // Store weight filename for download
        lastWeightFilename = data.weight_filename || null;

        // Show Save Weights button
        if (lastWeightFilename) {
            btnSaveWeights.classList.remove('hidden');
        }

        // Show message
        if (data.early_stopped) {
            showToast(`Early stopping at epoch ${data.stopped_epoch}! 🛑`, 'warning');
            // Add early stop log entry
            const esEntry = document.createElement('div');
            esEntry.className = 'log-entry log-entry-es';
            esEntry.innerHTML = `<span class="log-es-msg">🛑 Early stopping triggered at epoch ${data.stopped_epoch} (patience: ${data.message.match(/patience: (\d+)/)?.[1] || '?'}). Best weights restored.</span>`;
            trainingLog.appendChild(esEntry);
        } else {
            showToast('Training complete! 🎉', 'success');
        }

        // Final log entry
        const finalEntry = document.createElement('div');
        finalEntry.className = 'log-entry log-entry-final';
        finalEntry.innerHTML = `<span class="log-final">✅ ${data.message} — Final loss: ${data.final_train_loss.toFixed(4)} · Val loss: ${data.final_val_loss.toFixed(4)}</span>`;
        trainingLog.appendChild(finalEntry);
        trainingLog.scrollTop = trainingLog.scrollHeight;

        // Final network static draw
        if (modelConfig) {
            const ls = [modelConfig.input_dim || 4, ...(modelConfig.layer_sizes || [128, 64]), modelConfig.output_dim || 2];
            drawNetwork(ls);
        }
    });

    socket.on('training_stopped', () => {
        isTraining = false;
        stopNetworkAnimation();
        btnStop.classList.add('hidden');
        btnStart.classList.remove('hidden');
        btnStart.textContent = '▶ Restart';
        btnStart.disabled = false;
        showToast('Training stopped.', 'warning');
    });

    socket.on('training_error', (data) => {
        isTraining = false;
        stopNetworkAnimation();
        showToast('Training error: ' + data.message, 'error');
        btnStop.classList.add('hidden');
        btnStart.classList.remove('hidden');
        btnStart.disabled = false;
    });

    // ─── Save Weights ───
    btnSaveWeights.addEventListener('click', () => {
        if (!lastWeightFilename) {
            showToast('No weights file available.', 'error');
            return;
        }
        // Direct download via URL with filename in path — browser always gets the right name
        window.location.href = '/api/weights/file/' + encodeURIComponent(lastWeightFilename);
        showToast('Downloading: ' + lastWeightFilename, 'success');
    });

    // ─── Start / Stop ───
    btnStart.addEventListener('click', async () => {
        try {
            btnStart.disabled = true;
            btnStart.textContent = '⏳ Starting...';
            btnSaveWeights.classList.add('hidden');
            esBadge.classList.add('hidden');

            // Clear chart
            chart.data.labels = [];
            chart.data.datasets.forEach(ds => ds.data = []);
            chart.data.datasets[2].hidden = true;
            chart.update();

            // Clear log
            trainingLog.innerHTML = '<div class="log-entry text-muted">Starting training...</div>';

            // Reset stats
            statEpoch.textContent = '0/0';
            statLoss.textContent = '—';
            statAcc.textContent = '—';
            statEta.textContent = '—';
            progressBar.style.width = '0%';
            progressPct.textContent = '0%';

            const res = await fetch('/api/training/start', { method: 'POST' });
            const data = await res.json();

            if (data.error) {
                showToast(data.error, 'error');
                btnStart.disabled = false;
                btnStart.textContent = '▶ Start Training';
                return;
            }

            isTraining = true;
            modelConfig = data.info;

            // Show early stopping badge if enabled
            if (data.info.early_stopping && data.info.early_stopping.enabled) {
                esBadge.classList.remove('hidden');
                esBadge.title = `Early Stopping enabled (patience: ${data.info.early_stopping.patience})`;
            }

            btnStart.classList.add('hidden');
            btnStop.classList.remove('hidden');

            // Start network animation
            const layerSizes = [
                data.info.input_dim || 4,
                ...(modelConfig.layer_sizes || [128, 64]),
                data.info.output_dim || 2,
            ];
            startNetworkAnimation(layerSizes);

        } catch (e) {
            showToast('Error: ' + e.message, 'error');
            btnStart.disabled = false;
            btnStart.textContent = '▶ Start Training';
        }
    });

    btnStop.addEventListener('click', async () => {
        try {
            await fetch('/api/training/stop', { method: 'POST' });
        } catch (e) {
            showToast('Error: ' + e.message, 'error');
        }
    });

    // ─── Init ───
    initChart();

    // Draw initial empty network from state
    (async function loadState() {
        try {
            const state = await apiGet('/api/state');
            if (state.has_model && state.model_config) {
                modelConfig = state.model_config;
                const ls = [4, ...modelConfig.layer_sizes, 2];
                drawNetwork(ls);
            }
            // Show save button if weights already available
            if (state.last_weights_file) {
                lastWeightFilename = state.last_weights_file;
            }
        } catch (e) {
            // ignore
        }
    })();
})();
