/**
 * Vortex ML — Dataset Designer
 */

(function () {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadBar = document.getElementById('upload-bar');
    const uploadStatus = document.getElementById('upload-status');
    const previewSection = document.getElementById('preview-section');
    const pickerSection = document.getElementById('picker-section');
    const fileBadge = document.getElementById('file-badge');
    const dataSummary = document.getElementById('data-summary');
    const previewThead = document.getElementById('preview-thead');
    const previewTbody = document.getElementById('preview-tbody');
    const columnPicker = document.getElementById('column-picker');
    const btnContinue = document.getElementById('btn-continue');

    const steps = [
        document.getElementById('step-1'),
        document.getElementById('step-2'),
        document.getElementById('step-3'),
    ];

    let datasetInfo = null;
    let selectedFeatures = new Set();
    let selectedTarget = null;

    // ─── Step management ───
    function setStep(n) {
        steps.forEach((s, i) => {
            s.classList.remove('active', 'completed');
            if (i < n - 1) s.classList.add('completed');
            if (i === n - 1) s.classList.add('active');
        });
    }

    // ─── Drag & Drop ───
    ['dragover', 'dragenter'].forEach(evt => {
        uploadZone.addEventListener(evt, e => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        uploadZone.addEventListener(evt, () => {
            uploadZone.classList.remove('dragover');
        });
    });

    uploadZone.addEventListener('drop', e => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file) uploadFile(file);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) uploadFile(fileInput.files[0]);
    });

    // ─── Upload ───
    async function uploadFile(file) {
        const valid = /\.(csv|xlsx|xls)$/i.test(file.name);
        if (!valid) {
            showToast('Only CSV and Excel files are supported.', 'error');
            return;
        }

        uploadProgress.classList.add('active');
        uploadBar.style.width = '30%';
        uploadStatus.textContent = 'Uploading ' + file.name + '...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            uploadBar.style.width = '60%';
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.error) {
                showToast(data.error, 'error');
                uploadProgress.classList.remove('active');
                return;
            }

            uploadBar.style.width = '100%';
            uploadStatus.textContent = 'Upload complete!';

            setTimeout(() => {
                datasetInfo = data.info;
                showPreview(data.filename, data.info);
                showColumnPicker(data.info);
                setStep(2);
            }, 400);

        } catch (err) {
            showToast('Upload failed: ' + err.message, 'error');
            uploadProgress.classList.remove('active');
        }
    }

    // ─── Data Preview ───
    function showPreview(filename, info) {
        previewSection.classList.remove('hidden');

        fileBadge.innerHTML = `<span class="badge badge-success">✓ ${filename}</span>`;

        dataSummary.innerHTML = `
            <div class="summary-item"><div class="si-label">Rows</div><div class="si-value">${info.rows.toLocaleString()}</div></div>
            <div class="summary-item"><div class="si-label">Columns</div><div class="si-value">${info.cols}</div></div>
            <div class="summary-item"><div class="si-label">Numeric</div><div class="si-value">${info.columns.filter(c => c.is_numeric).length}</div></div>
            <div class="summary-item"><div class="si-label">Categorical</div><div class="si-value">${info.columns.filter(c => !c.is_numeric).length}</div></div>
        `;

        // Table header
        previewThead.innerHTML = '<tr>' + info.columns.map(c => `<th>${c.name}</th>`).join('') + '</tr>';

        // Table rows
        previewTbody.innerHTML = info.preview.map(row =>
            '<tr>' + info.columns.map(c => `<td>${row[c.name] ?? ''}</td>`).join('') + '</tr>'
        ).join('');
    }

    // ─── Column Picker ───
    function showColumnPicker(info) {
        pickerSection.classList.remove('hidden');
        setStep(3);

        columnPicker.innerHTML = info.columns.map(col => `
            <div class="col-item" data-col="${col.name}">
                <div>
                    <div class="col-name">${col.name}</div>
                    <div class="col-dtype">${col.dtype}${col.is_numeric ? ' · ' + (col.unique) + ' unique' : ' · ' + (col.unique) + ' categories'}</div>
                </div>
                <div class="col-controls">
                    <button class="col-btn btn-feat" data-col="${col.name}">Feature</button>
                    <button class="col-btn btn-tgt" data-col="${col.name}">Target</button>
                </div>
            </div>
        `).join('');

        // Event delegation
        columnPicker.addEventListener('click', e => {
            const btn = e.target.closest('.col-btn');
            if (!btn) return;

            const colName = btn.dataset.col;
            const item = btn.closest('.col-item');

            if (btn.classList.contains('btn-feat')) {
                // Toggle feature
                if (selectedFeatures.has(colName)) {
                    selectedFeatures.delete(colName);
                    btn.classList.remove('active-feature');
                    item.classList.remove('selected-feature');
                } else {
                    // Remove target if it was target
                    if (selectedTarget === colName) {
                        selectedTarget = null;
                        item.querySelector('.btn-tgt').classList.remove('active-target');
                        item.classList.remove('selected-target');
                    }
                    selectedFeatures.add(colName);
                    btn.classList.add('active-feature');
                    item.classList.add('selected-feature');
                }
            } else if (btn.classList.contains('btn-tgt')) {
                // Clear previous target
                if (selectedTarget) {
                    const prevItem = columnPicker.querySelector(`.col-item[data-col="${selectedTarget}"]`);
                    if (prevItem) {
                        prevItem.querySelector('.btn-tgt').classList.remove('active-target');
                        prevItem.classList.remove('selected-target');
                    }
                }

                // Remove from features if was feature
                if (selectedFeatures.has(colName)) {
                    selectedFeatures.delete(colName);
                    item.querySelector('.btn-feat').classList.remove('active-feature');
                    item.classList.remove('selected-feature');
                }

                selectedTarget = colName;
                btn.classList.add('active-target');
                item.classList.add('selected-target');
            }

            updateContinueBtn();
        });
    }

    function updateContinueBtn() {
        const ready = selectedFeatures.size > 0 && selectedTarget !== null;
        btnContinue.style.opacity = ready ? '1' : '0.4';
        btnContinue.style.pointerEvents = ready ? 'auto' : 'none';
    }

    // ─── Continue button ───
    btnContinue.addEventListener('click', async (e) => {
        if (selectedFeatures.size === 0 || !selectedTarget) {
            e.preventDefault();
            showToast('Select at least one feature and one target column.', 'error');
            return;
        }

        try {
            const res = await apiPost('/api/dataset/configure', {
                feature_cols: Array.from(selectedFeatures),
                target_col: selectedTarget,
            });

            if (res.error) {
                e.preventDefault();
                showToast(res.error, 'error');
                return;
            }

            showToast('Dataset configured! Redirecting...', 'success');
            // allow navigation
        } catch (err) {
            e.preventDefault();
            showToast('Error: ' + err.message, 'error');
        }
    });

    // ─── Check if dataset already loaded ───
    (async function init() {
        try {
            const state = await apiGet('/api/state');
            if (state.has_dataset) {
                const data = await apiGet('/api/dataset/analyze');
                if (data.info) {
                    datasetInfo = data.info;
                    showPreview(data.filename, data.info);
                    showColumnPicker(data.info);

                    // Restore selections
                    if (state.has_features) {
                        state.feature_cols.forEach(col => {
                            selectedFeatures.add(col);
                            const item = columnPicker.querySelector(`.col-item[data-col="${col}"]`);
                            if (item) {
                                item.querySelector('.btn-feat').classList.add('active-feature');
                                item.classList.add('selected-feature');
                            }
                        });

                        if (state.target_col) {
                            selectedTarget = state.target_col;
                            const item = columnPicker.querySelector(`.col-item[data-col="${state.target_col}"]`);
                            if (item) {
                                item.querySelector('.btn-tgt').classList.add('active-target');
                                item.classList.add('selected-target');
                            }
                        }

                        updateContinueBtn();
                    }
                }
            }
        } catch (e) {
            // ignore
        }
    })();
})();
