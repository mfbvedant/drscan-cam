/**
 * DRScan Cam — Dashboard Application Logic
 * Handles: modality selection, file upload, API calls, result rendering
 */

// =============================================================================
// STATE
// =============================================================================
const state = {
    selectedModality: 'chest_xray',
    selectedFile: null,
    previewUrl: null,
    isAnalyzing: false,
    results: null,
};

// =============================================================================
// DOM REFERENCES
// =============================================================================
const dom = {
    // Modality
    modalityTabs: () => document.querySelectorAll('.modality-tab'),

    // Upload
    uploadZone: () => document.getElementById('upload-zone'),
    fileInput: () => document.getElementById('file-input'),
    uploadContent: () => document.getElementById('upload-content'),
    previewContainer: () => document.getElementById('preview-container'),
    previewImage: () => document.getElementById('preview-image'),
    previewFilename: () => document.getElementById('preview-filename'),
    btnClear: () => document.getElementById('btn-clear'),
    btnAnalyze: () => document.getElementById('btn-analyze'),

    // Results
    resultsEmpty: () => document.getElementById('results-empty'),
    resultsLoading: () => document.getElementById('results-loading'),
    resultsContent: () => document.getElementById('results-content'),
    diagnosisLabel: () => document.getElementById('diagnosis-label'),
    severityBadge: () => document.getElementById('severity-badge'),
    severityLabel: () => document.getElementById('severity-label-text'),
    gaugeValue: () => document.getElementById('gauge-value'),
    gaugeFill: () => document.getElementById('gauge-fill'),
    predictionsList: () => document.getElementById('predictions-list'),
    recommendationText: () => document.getElementById('recommendation-text'),
    inferenceTime: () => document.getElementById('inference-time'),
    modelVersion: () => document.getElementById('model-version'),

    // Heatmap
    heatmapCard: () => document.getElementById('heatmap-card'),
    originalImage: () => document.getElementById('original-image'),
    heatmapImage: () => document.getElementById('heatmap-image'),
    regionsList: () => document.getElementById('regions-list'),

    // Report
    reportCard: () => document.getElementById('report-card'),
    reportContent: () => document.getElementById('report-content'),
    reportDisclaimer: () => document.getElementById('report-disclaimer'),
};

// =============================================================================
// MODALITY SELECTION
// =============================================================================
function initModalityTabs() {
    dom.modalityTabs().forEach(tab => {
        tab.addEventListener('click', () => {
            const modality = tab.dataset.modality;
            selectModality(modality);
        });
    });
}

function selectModality(modality) {
    state.selectedModality = modality;
    dom.modalityTabs().forEach(tab => {
        tab.classList.toggle('active', tab.dataset.modality === modality);
    });
}

// =============================================================================
// FILE UPLOAD
// =============================================================================
function initUpload() {
    const zone = dom.uploadZone();
    const input = dom.fileInput();

    // Click to upload
    zone.addEventListener('click', (e) => {
        if (!state.selectedFile && !e.target.closest('.preview-actions')) {
            input.click();
        }
    });

    // File selected via input
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & drop
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clear button
    dom.btnClear().addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    // Analyze button
    dom.btnAnalyze().addEventListener('click', analyzeImage);
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff',
                        'application/dicom', 'application/octet-stream'];
    const ext = file.name.toLowerCase().split('.').pop();
    const validExts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'dcm', 'dicom'];

    if (!validExts.includes(ext)) {
        showToast('Unsupported file format. Please upload a medical image (JPEG, PNG, DICOM, TIFF).', 'error');
        return;
    }

    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showToast('File too large. Maximum size is 50MB.', 'error');
        return;
    }

    state.selectedFile = file;
    state.previewUrl = URL.createObjectURL(file);

    // Show preview
    const zone = dom.uploadZone();
    zone.classList.add('has-file');
    dom.uploadContent().style.display = 'none';
    dom.previewContainer().style.display = 'block';
    dom.previewImage().src = state.previewUrl;
    dom.previewFilename().textContent = file.name;
    dom.btnAnalyze().disabled = false;

    // Reset results
    resetResults();
}

function clearFile() {
    if (state.previewUrl) {
        URL.revokeObjectURL(state.previewUrl);
    }
    state.selectedFile = null;
    state.previewUrl = null;

    const zone = dom.uploadZone();
    zone.classList.remove('has-file');
    dom.uploadContent().style.display = 'flex';
    dom.previewContainer().style.display = 'none';
    dom.previewImage().src = '';
    dom.fileInput().value = '';
    dom.btnAnalyze().disabled = true;

    resetResults();
}

// =============================================================================
// ANALYSIS
// =============================================================================
async function analyzeImage() {
    if (!state.selectedFile || state.isAnalyzing) return;

    state.isAnalyzing = true;
    dom.btnAnalyze().disabled = true;
    dom.btnAnalyze().textContent = 'Analyzing...';

    // Show loading
    dom.resultsEmpty().style.display = 'none';
    dom.resultsContent().classList.remove('active');
    dom.resultsLoading().classList.add('active');
    dom.heatmapCard().classList.remove('active');
    dom.reportCard().classList.remove('active');

    try {
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        formData.append('modality', state.selectedModality);

        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const data = await response.json();
        state.results = data;
        renderResults(data);

    } catch (error) {
        console.error('Analysis error:', error);
        showToast(error.message || 'Analysis failed. Please try again.', 'error');
        dom.resultsLoading().classList.remove('active');
        dom.resultsEmpty().style.display = 'flex';
    } finally {
        state.isAnalyzing = false;
        dom.btnAnalyze().disabled = false;
        dom.btnAnalyze().textContent = '🔬 Analyze Image';
    }
}

// =============================================================================
// RENDER RESULTS
// =============================================================================
function renderResults(data) {
    // Hide loading, show content
    dom.resultsLoading().classList.remove('active');
    dom.resultsContent().classList.add('active');

    // Diagnosis
    dom.diagnosisLabel().textContent = data.diagnosis;

    // Severity badge
    const badge = dom.severityBadge();
    badge.className = `severity-badge severity-${data.severity}`;
    badge.textContent = data.severity;

    // Severity label
    const sevLabel = dom.severityLabel();
    sevLabel.className = `severity-label severity-${data.severity}`;
    sevLabel.textContent = `${data.severity_label} — Grade ${data.severity}/4`;

    // Confidence gauge
    const confidence = data.confidence;
    dom.gaugeValue().textContent = `${(confidence * 100).toFixed(1)}`;
    const circumference = 2 * Math.PI * 40; // radius = 40
    const offset = circumference * (1 - confidence);
    setTimeout(() => {
        dom.gaugeFill().style.strokeDashoffset = offset;
        // Color based on confidence
        const hue = confidence > 0.8 ? 170 : confidence > 0.6 ? 45 : 0;
        dom.gaugeFill().style.stroke = `hsl(${hue}, 80%, 55%)`;
    }, 100);

    // Top predictions
    renderPredictions(data.top_predictions);

    // Recommendation
    dom.recommendationText().textContent = data.recommendation;

    // Inference meta
    dom.inferenceTime().textContent = `${data.inference_time_ms.toFixed(0)}ms`;
    dom.modelVersion().textContent = data.model_version;

    // Heatmap
    renderHeatmap(data);

    // Report
    renderReport(data);
}

function renderPredictions(predictions) {
    const container = dom.predictionsList();
    container.innerHTML = '';

    const topN = predictions.slice(0, 5);
    topN.forEach((pred, idx) => {
        const row = document.createElement('div');
        row.className = 'prediction-row';

        const pct = (pred.confidence * 100).toFixed(1);
        row.innerHTML = `
            <span class="prediction-name" title="${pred.class_name}">${pred.class_name}</span>
            <div class="prediction-bar-bg">
                <div class="prediction-bar-fill ${idx > 0 ? 'secondary' : ''}" 
                     style="width: 0%" data-width="${pct}%"></div>
            </div>
            <span class="prediction-value">${pct}%</span>
        `;
        container.appendChild(row);
    });

    // Animate bars
    requestAnimationFrame(() => {
        setTimeout(() => {
            container.querySelectorAll('.prediction-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 150);
    });
}

function renderHeatmap(data) {
    if (data.original_base64 && data.heatmap_base64) {
        dom.originalImage().src = `data:image/jpeg;base64,${data.original_base64}`;
        dom.heatmapImage().src = `data:image/jpeg;base64,${data.heatmap_base64}`;
        dom.heatmapCard().classList.add('active');

        // Regions of concern
        const regionsList = dom.regionsList();
        regionsList.innerHTML = '';
        (data.regions_of_concern || []).forEach(region => {
            const item = document.createElement('div');
            item.className = 'region-item';
            item.innerHTML = `
                <span class="region-dot"></span>
                <span>${region}</span>
            `;
            regionsList.appendChild(item);
        });
    }
}

function renderReport(data) {
    // Build report text from available data
    const lines = [
        `**Modality:** ${formatModality(data.modality)}`,
        `**Primary Finding:** ${data.diagnosis}`,
        `**Severity Grade:** ${data.severity}/4 (${data.severity_label})`,
        `**Confidence Score:** ${(data.confidence * 100).toFixed(1)}%`,
        '',
        '**Top Differential Diagnoses:**',
    ];

    (data.top_predictions || []).slice(0, 4).forEach(pred => {
        lines.push(`  • ${pred.class_name}: ${(pred.confidence * 100).toFixed(1)}%`);
    });

    lines.push('');
    lines.push('**Clinical Recommendation:**');
    lines.push(data.recommendation);

    // Render with bold formatting
    const reportEl = dom.reportContent();
    reportEl.innerHTML = lines.join('\n').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    dom.reportDisclaimer().textContent =
        '⚠️ This AI-generated report is intended as a clinical decision support tool only. ' +
        'It does not constitute a medical diagnosis. All findings must be reviewed and confirmed ' +
        'by a qualified healthcare professional.';

    dom.reportCard().classList.add('active');
}

// =============================================================================
// UTILITIES
// =============================================================================
function resetResults() {
    dom.resultsEmpty().style.display = 'flex';
    dom.resultsLoading().classList.remove('active');
    dom.resultsContent().classList.remove('active');
    dom.heatmapCard().classList.remove('active');
    dom.reportCard().classList.remove('active');

    // Reset gauge
    dom.gaugeFill().style.strokeDashoffset = 251;
    dom.gaugeValue().textContent = '--';

    state.results = null;
}

function formatModality(modality) {
    const map = {
        chest_xray: 'Chest X-Ray',
        brain_mri: 'Brain MRI',
        lung_ct: 'Lung CT',
    };
    return map[modality] || modality;
}

function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 10000;
        padding: 14px 20px;
        max-width: 400px;
        background: ${type === 'error' ? 'rgba(239, 68, 68, 0.95)' : 'rgba(6, 214, 224, 0.95)'};
        color: #fff;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        font-weight: 500;
        border-radius: 12px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function downloadReport() {
    if (!state.results) return;

    const data = state.results;
    const text = [
        '═══════════════════════════════════════════════════════',
        '    DRScan Cam — AI Medical Image Analysis Report',
        '═══════════════════════════════════════════════════════',
        '',
        `Date: ${new Date().toLocaleString()}`,
        `Modality: ${formatModality(data.modality)}`,
        '',
        '─── DIAGNOSIS ───',
        `Primary Finding: ${data.diagnosis}`,
        `Severity: Grade ${data.severity}/4 (${data.severity_label})`,
        `Confidence: ${(data.confidence * 100).toFixed(1)}%`,
        '',
        '─── TOP PREDICTIONS ───',
        ...(data.top_predictions || []).map(p =>
            `  ${p.class_name}: ${(p.confidence * 100).toFixed(1)}%`
        ),
        '',
        '─── REGIONS OF CONCERN ───',
        ...(data.regions_of_concern || []).map(r => `  • ${r}`),
        '',
        '─── RECOMMENDATION ───',
        data.recommendation,
        '',
        '─── DISCLAIMER ───',
        'This AI-generated report is intended as a clinical decision',
        'support tool only. It does not constitute a medical diagnosis.',
        'All findings must be reviewed and confirmed by a qualified',
        'healthcare professional.',
        '',
        `Inference Time: ${data.inference_time_ms.toFixed(0)}ms`,
        `Model Version: ${data.model_version}`,
        '',
        '═══════════════════════════════════════════════════════',
    ].join('\n');

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drscan-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Report downloaded successfully.');
}

function newAnalysis() {
    clearFile();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initModalityTabs();
    initUpload();

    // Select default modality
    selectModality('chest_xray');

    console.log('🏥 DRScan Cam dashboard initialized');
});
