/**
 * lexprep Web Application - Frontend JavaScript
 */

//  State Management 
const state = {
    selectedLanguage: null,
    selectedTool: null,
    tools: {},
    results: [],
    uploadedFile: null,
    wordColumn: null,
    availableColumns: [],
    isProcessing: false,
    warmupStatus: {
        started: false,
        completed: false,
        loaded: [],
        errors: []
    },
    currentJobId: null
};

// DOM Elements
const elements = {
    languageGrid: document.getElementById('languageGrid'),
    toolGrid: document.getElementById('toolGrid'),
    processBtn: document.getElementById('processBtn'),
    toast: document.getElementById('toast'),
    navToggle: document.querySelector('.nav-toggle'),
    navMenu: document.querySelector('.nav-menu'),
    // File upload elements
    fileUploadArea: document.getElementById('fileUploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadPreview: document.getElementById('uploadPreview'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    removeFile: document.getElementById('removeFile'),
    wordColumnSelect: document.getElementById('wordColumnSelect'),
    wordColumnDropdown: document.getElementById('wordColumnDropdown'),
    cancelProcessBtn: document.getElementById('cancelProcessBtn'),
    accuracyNote: document.getElementById('accuracyNote'),

    // Sampling Elements
    stratifiedForm: document.getElementById('stratifiedForm'),
    stratFileArea: document.getElementById('stratFileArea'),
    stratFile: document.getElementById('stratFile'),
    stratFilePreview: document.getElementById('stratFilePreview'),
    stratFileName: document.getElementById('stratFileName'),
    stratUploadContent: document.getElementById('stratUploadContent'),
    removeStratFile: document.getElementById('removeStratFile'),
    stratFileInfo: document.getElementById('stratFileInfo'),
    stratColumnSection: document.getElementById('stratColumnSection'),
    stratColumn: document.getElementById('stratColumn'),
    stratConfigSection: document.getElementById('stratConfigSection'),
    stratButtonGroup: document.getElementById('stratButtonGroup'),
    stratifiedBtn: document.getElementById('stratifiedBtn'),
    stratifiedResults: document.getElementById('stratifiedResults'),
    stratifiedWarningsList: document.getElementById('stratifiedWarningsList'),
    bins: document.getElementById('bins'),
    allocation: document.getElementById('allocation'),
    shuffleForm: document.getElementById('shuffleForm'),
    shuffleFiles: document.getElementById('shuffleFiles'),
    shuffleBtn: document.getElementById('shuffleBtn'),
};

// ============== Initialization ==============
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    loadTools();
    initFileUpload();
    initEventListeners();
    initSamplingUI();
    initTypingAnimation();
    triggerWarmup();
});

// ============== Model Warmup ==============
async function triggerWarmup() {
    try {
        
        const response = await fetch('/api/warmup', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'started') {
            console.log('[Warmup] Model warmup started in background');
            pollWarmupStatus();
        } else if (data.status === 'already_started') {
            state.warmupStatus = {
                started: true,
                completed: data.completed,
                loaded: data.loaded || [],
                errors: data.errors || []
            };
            if (!data.completed) {
                pollWarmupStatus();
            } else {
                console.log('[Warmup] Already complete:', data.loaded);
            }
        }
    } catch (error) {
        console.log('[Warmup] Failed to trigger warmup:', error);
    }
}

async function pollWarmupStatus() {
    const maxAttempts = 60; // Poll for up to 5 minutes
    let attempts = 0;

    const poll = async () => {
        try {
            const response = await fetch('/api/warmup/status');
            const data = await response.json();

            state.warmupStatus = data;

            // Stop polling if completed is true (even if there are errors)
            if (data.completed === true) {
                console.log('[Warmup] Complete! Loaded:', data.loaded);
                if (data.errors && data.errors.length > 0) {
                    console.log('[Warmup] Errors:', data.errors);
                }
                return; // STOP polling
            }

            attempts++;
            if (attempts < maxAttempts) {
                setTimeout(poll, 5000); // Poll every 5 seconds
            }
        } catch (error) {
            console.log('[Warmup] Poll error:', error);
        }
    };

    poll();
}

// ============== Elite Typewriter Animation ==============
function initTypingAnimation() {
    const lexEl = document.querySelector('.typewriter-lex');
    const prepEl = document.querySelector('.typewriter-prep');
    const spaceEl = document.querySelector('.typewriter-space');

    if (!lexEl || !prepEl) return;

    const lexShort = lexEl.dataset.short || 'lex';
    const lexFull = lexEl.dataset.full || 'lexical';
    const prepShort = prepEl.dataset.short || 'prep';
    const prepFull = prepEl.dataset.full || 'preparation';

    const typeSpeed = 80;  // ms per character
    const deleteSpeed = 50; // ms per character
    const pauseBetween = 2000; // pause before switching

    async function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async function typeText(element, text, startFrom = 0) {
        for (let i = startFrom; i <= text.length; i++) {
            element.textContent = text.slice(0, i);
            await sleep(typeSpeed);
        }
    }

    async function deleteText(element, targetText) {
        const currentText = element.textContent;
        for (let i = currentText.length; i >= targetText.length; i--) {
            element.textContent = currentText.slice(0, i);
            await sleep(deleteSpeed);
        }
        // Ensure final state is exactly the target
        element.textContent = targetText;
    }

    async function expandAnimation() {
        // Type out "ical" after "lex" → "lexical"
        await typeText(lexEl, lexFull, lexShort.length);
        await sleep(100);
        // Show space between words
        if (spaceEl) spaceEl.textContent = ' ';
        await sleep(typeSpeed);
        // Type out "aration" after "prep" → "preparation"
        await typeText(prepEl, prepFull, prepShort.length);
    }

    async function collapseAnimation() {
        // Delete back to "prep"
        await deleteText(prepEl, prepShort);
        await sleep(100);
        // Hide space
        if (spaceEl) spaceEl.textContent = '';
        await sleep(deleteSpeed);
        // Delete back to "lex"
        await deleteText(lexEl, lexShort);
    }

    async function runAnimation() {
        while (true) {
            await sleep(pauseBetween);

            // Expand: lex → lexical, prep → preparation (with space)
            await expandAnimation();

            await sleep(pauseBetween);

            // Collapse: lexical → lex, preparation → prep (remove space)
            await collapseAnimation();
        }
    }

    // Start the animation loop
    runAnimation();
}

// ============== Navigation ==============
function initNavigation() {
    const navToggle = document.getElementById('navToggle') || document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');

    if (!navToggle || !navMenu) return;

    // Toggle menu on button click
    navToggle.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        const isOpen = navMenu.classList.contains('active');

        if (isOpen) {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
        } else {
            navToggle.classList.add('active');
            navMenu.classList.add('active');
        }

        return false;
    }, true);

    // Prevent any bubbling from the toggle button
    navToggle.addEventListener('mousedown', (e) => e.stopPropagation());

    // Close menu on link click
    navMenu.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // Close menu on outside click
    document.addEventListener('click', (e) => {
        if (navMenu.classList.contains('active')) {
            if (!navToggle.contains(e.target) && !navMenu.contains(e.target)) {
                navToggle.classList.remove('active');
                navMenu.classList.remove('active');
            }
        }
    });

    // Close menu on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && navMenu.classList.contains('active')) {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
        }
    });
}

// ============== Load Tools from API ==============
async function loadTools() {
    try {
        const response = await fetch('/api/tools');
        state.tools = await response.json();
        renderLanguages();
    } catch (error) {
        console.error('Failed to load tools:', error);
        showToast('Failed to load tools. Please refresh the page.', 'error');
    }
}

// ============== Render Languages ==============
function renderLanguages() {
    if (!elements.languageGrid) return;

    elements.languageGrid.innerHTML = Object.entries(state.tools)
        .map(([code, lang]) => `
            <button class="language-btn" data-lang="${escapeHtml(code)}">
                <img src="${escapeHtml(lang.flag)}" alt="${escapeHtml(code)}" class="flag-icon">
                <span class="name">${escapeHtml(lang.name)}</span>
            </button>
        `)
        .join('');

    // Add click handlers
    elements.languageGrid.querySelectorAll('.language-btn').forEach(btn => {
        btn.addEventListener('click', () => selectLanguage(btn.dataset.lang));
    });
}

// ============== Select Language ==============
function selectLanguage(langCode) {
    state.selectedLanguage = langCode;
    state.selectedTool = null;

    // Update UI
    elements.languageGrid.querySelectorAll('.language-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === langCode);
    });

    renderTools();
    updateProcessButton();
    updateAccuracyNote();
}

// ============== Render Tools ==============
function renderTools() {
    if (!elements.toolGrid) return;

    if (!state.selectedLanguage) {
        elements.toolGrid.innerHTML = `
            <div class="placeholder-message">
                Please select a language first
            </div>
        `;
        return;
    }

    const lang = state.tools[state.selectedLanguage];
    elements.toolGrid.innerHTML = Object.entries(lang.tools)
        .map(([code, tool]) => `
            <button class="tool-btn" data-tool="${escapeHtml(code)}">
                <span class="name">${escapeHtml(tool.name)}</span>
                <span class="description">${escapeHtml(tool.description)}</span>
            </button>
        `)
        .join('');

    // Add click handlers
    elements.toolGrid.querySelectorAll('.tool-btn').forEach(btn => {
        btn.addEventListener('click', () => selectTool(btn.dataset.tool));
    });

    // Animate in
    elements.toolGrid.querySelectorAll('.tool-btn').forEach((btn, i) => {
        btn.style.opacity = '0';
        btn.style.transform = 'translateY(10px)';
        setTimeout(() => {
            btn.style.transition = 'all 0.3s ease';
            btn.style.opacity = '1';
            btn.style.transform = 'translateY(0)';
        }, i * 50);
    });
}

// ============== Select Tool ==============
function selectTool(toolCode) {
    state.selectedTool = toolCode;

    // Update UI
    elements.toolGrid.querySelectorAll('.tool-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tool === toolCode);
    });

    updateProcessButton();
    updateAccuracyNote();
}

// ============== Update Accuracy Note ==============
function updateAccuracyNote() {
    if (!elements.accuracyNote) return;

    // Show note only for Persian syllables tool
    if (state.selectedLanguage === 'fa' && state.selectedTool === 'syllables') {
        elements.accuracyNote.style.display = 'flex';
    } else {
        elements.accuracyNote.style.display = 'none';
    }
}

// ============== File Upload ==============
function initFileUpload() {
    if (!elements.fileUploadArea || !elements.fileInput) return;

    // Click to upload
    elements.fileUploadArea.addEventListener('click', (e) => {
        if (elements.removeFile && (e.target === elements.removeFile || elements.removeFile.contains(e.target))) {
            return;
        }
        elements.fileInput.click();
    });

    // Drag and drop
    elements.fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.fileUploadArea.classList.add('dragover');
    });

    elements.fileUploadArea.addEventListener('dragleave', () => {
        elements.fileUploadArea.classList.remove('dragover');
    });

    elements.fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.fileUploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Remove file
    if (elements.removeFile) {
        elements.removeFile.addEventListener('click', (e) => {
            e.stopPropagation();
            clearFile();
        });
    }

    // Word column dropdown change
    if (elements.wordColumnDropdown) {
        elements.wordColumnDropdown.addEventListener('change', (e) => {
            state.wordColumn = e.target.value;
        });
    }
}

async function handleFileSelect(file) {
    const validExtensions = ['.xlsx', '.xls', '.csv', '.tsv', '.txt'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
        showToast('Please upload Excel, CSV, or TXT files only', 'error');
        return;
    }

    state.uploadedFile = file;

    // GA4: track file upload
    if (typeof gtag === 'function') {
        gtag('event', 'file_uploaded', {
            file_size: file.size,
            file_type: ext,
            language: state.selectedLanguage || undefined,
            analysis_type: state.selectedTool || undefined
        });
    }

    // Update UI
    elements.fileUploadArea.classList.add('has-file');
    const uploadContent = elements.fileUploadArea.querySelector('.upload-content');
    if (uploadContent) uploadContent.style.display = 'none';
    if (elements.uploadPreview) elements.uploadPreview.style.display = 'flex';

    if (elements.fileName) elements.fileName.textContent = file.name;
    if (elements.fileSize) elements.fileSize.textContent = formatFileSize(file.size);

    // Parse file to get columns (for Excel/CSV)
    if (ext !== '.txt') {
        await parseFileForColumns(file);
    } else {
        state.availableColumns = ['word'];
        state.wordColumn = 'word';
        if (elements.wordColumnSelect) elements.wordColumnSelect.style.display = 'none';
    }

    updateProcessButton();
}

async function parseFileForColumns(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/parse-columns', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.columns && data.columns.length > 0) {
            state.availableColumns = data.columns;
            showColumnSelector(data.columns, data.suggested);
        }
    } catch (error) {
        console.error('Error parsing file:', error);
        showToast('Error reading file columns', 'error');
    }
}

function showColumnSelector(columns, suggested) {
    if (!elements.wordColumnSelect || !elements.wordColumnDropdown) return;

    elements.wordColumnDropdown.innerHTML = columns.map(col =>
        `<option value="${escapeHtml(col)}" ${col === suggested ? 'selected' : ''}>${escapeHtml(col)}</option>`
    ).join('');

    state.wordColumn = suggested || columns[0];
    elements.wordColumnSelect.style.display = 'flex';
}

function clearFile() {
    state.uploadedFile = null;
    state.wordColumn = null;
    state.availableColumns = [];

    elements.fileUploadArea.classList.remove('has-file');
    const uploadContent = elements.fileUploadArea.querySelector('.upload-content');
    if (uploadContent) uploadContent.style.display = 'block';
    if (elements.uploadPreview) elements.uploadPreview.style.display = 'none';
    if (elements.fileInput) elements.fileInput.value = '';
    if (elements.wordColumnSelect) elements.wordColumnSelect.style.display = 'none';

    updateProcessButton();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============== Process Button ==============
function updateProcessButton() {
    if (!elements.processBtn) return;

    const canProcess = state.selectedLanguage &&
                       state.selectedTool &&
                       state.uploadedFile !== null;

    elements.processBtn.disabled = !canProcess;
}

// ============== Event Listeners ==============
function initEventListeners() {
    // Process button
    if (elements.processBtn) {
        elements.processBtn.addEventListener('click', processFile);
    }

    // Cancel button
    if (elements.cancelProcessBtn) {
        elements.cancelProcessBtn.addEventListener('click', () => {
            cancelProcessing();
        });
    }
}

// ============== Sampling Tools JavaScript ==============
// This code is kept in sync with the inline script in sampling.html
// The initSamplingUI function sets up all event handlers when the sampling page loads

let stratifiedFileData = null;
let currentMode = 'quantile';

// Advanced options toggle
function toggleAdvanced() {
    const toggle = document.querySelector('.advanced-toggle');
    const options = document.getElementById('advancedOptions');
    toggle.classList.toggle('open');
    options.classList.toggle('open');
}

// Add range row
function addRangeRow() {
    const builder = document.getElementById('rangeBuilder');
    const row = document.createElement('div');
    const isFixed = document.getElementById('allocation').value === 'fixed';
    row.className = 'range-row' + (isFixed ? ' show-fixed' : '');
    row.innerHTML = `
        <div class="range-input-wrap"><input type="number" step="any" placeholder="Min" class="range-min"></div>
        <div class="range-input-wrap"><input type="number" step="any" placeholder="Max" class="range-max"></div>
        <div class="range-count-wrap"><input type="number" min="0" placeholder="Count" class="range-fixed-count" title="Fixed count for this range"></div>
        <button type="button" class="remove-range-btn" onclick="this.closest('.range-row').remove(); updateWeightsGrid();" title="Remove">&times;</button>
    `;
    builder.appendChild(row);
    updateWeightsGrid();
}

// Handle allocation method change
function handleAllocationChange() {
    const allocation = document.getElementById('allocation').value;
    const allocationInfo = document.getElementById('allocationInfo');
    const weightsCheckboxWrap = document.getElementById('weightsCheckboxWrap');
    const nTotalCustomGroup = document.getElementById('nTotalCustomGroup');
    const fixedCountHeader = document.getElementById('fixedCountHeader');

    // Update info text
    const infoTexts = {
        'equal': 'Each stratum gets the same number of samples: n/k where n=total samples, k=number of groups.',
        'proportional': 'Larger strata get more samples, proportional to their size: n_h = n * (N_h / N).',
        'optimal': 'Allocates more to high-variance strata (Neyman): n_h = n * (N_h * S_h) / sum(N_j * S_j). Minimizes overall variance.',
        'fixed': 'You specify the exact count for each stratum in Custom Ranges mode. In Automatic Bins, enter total samples.'
    };
    allocationInfo.textContent = infoTexts[allocation] || '';

    // Show/hide weights checkbox (not for fixed)
    const isFixed = allocation === 'fixed';
    weightsCheckboxWrap.style.display = isFixed ? 'none' : 'flex';
    if (isFixed) {
        document.getElementById('useWeights').checked = false;
        document.getElementById('weightsPanel').classList.remove('open');
    }

    // Show/hide fixed count inputs in custom mode
    document.querySelectorAll('.range-row').forEach(row => {
        if (isFixed) {
            row.classList.add('show-fixed');
        } else {
            row.classList.remove('show-fixed');
        }
    });

    // Show/hide n_total and fixed count header in custom mode
    if (currentMode === 'custom') {
        nTotalCustomGroup.style.display = isFixed ? 'none' : 'block';
        fixedCountHeader.style.display = isFixed ? 'inline' : 'none';
    }
}

// Toggle weights panel
function toggleWeightsPanel() {
    const checkbox = document.getElementById('useWeights');
    const panel = document.getElementById('weightsPanel');
    if (checkbox.checked) {
        panel.classList.add('open');
        updateWeightsGrid();
    } else {
        panel.classList.remove('open');
    }
}

// Update weights grid based on number of bins/ranges
function updateWeightsGrid() {
    const grid = document.getElementById('weightsGrid');
    const allocation = document.getElementById('allocation').value;

    if (allocation === 'fixed') {
        grid.innerHTML = '';
        return;
    }

    let numGroups = 0;
    if (currentMode === 'quantile') {
        numGroups = parseInt(document.getElementById('bins').value) || 3;
    } else {
        numGroups = document.querySelectorAll('.range-row').length;
    }

    grid.innerHTML = '';
    for (let i = 0; i < numGroups; i++) {
        const item = document.createElement('div');
        item.className = 'weight-item';
        item.innerHTML = `
            <label>Group ${i + 1}:</label>
            <input type="number" step="0.1" min="0" value="1.0" class="weight-input" data-group="${i}">
        `;
        grid.appendChild(item);
    }
}

// Modal functions
function openModal(id) {
    document.getElementById(id).classList.add('open');
}
function closeModal(id) {
    document.getElementById(id).classList.remove('open');
}

// Stratified file handler
async function handleStratifiedFile() {
    const stratFileInput = document.getElementById('stratFile');
    const stratFileName = document.getElementById('stratFileName');
    const stratFilePreview = document.getElementById('stratFilePreview');
    const stratUploadContent = document.getElementById('stratUploadContent');
    const stratFileArea = document.getElementById('stratFileArea');

    if (!stratFileInput.files || stratFileInput.files.length === 0) return;

    const file = stratFileInput.files[0];
    stratFileName.textContent = file.name;
    stratFilePreview.style.display = 'flex';
    stratUploadContent.style.display = 'none';
    stratFileArea.classList.add('has-file');

    // GA4: track sampling file upload
    if (typeof gtag === 'function') {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        gtag('event', 'file_uploaded', {
            file_size: file.size,
            file_type: ext,
            context: 'sampling'
        });
    }

    // Parse file to get columns
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/sampling/parse-file', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Failed to parse file');

        stratifiedFileData = await response.json();

        // Show file info
        const fileInfo = document.getElementById('stratFileInfo');
        fileInfo.innerHTML = `<div class="file-info-badge">${stratifiedFileData.row_count} rows detected</div>`;
        fileInfo.style.display = 'block';

        // Populate column dropdown
        const columnSelect = document.getElementById('stratColumn');
        columnSelect.innerHTML = '<option value="">Select a numeric column...</option>';

        stratifiedFileData.columns.forEach(col => {
            if (col.is_numeric) {
                const option = document.createElement('option');
                option.value = col.name;
                option.textContent = col.name;
                if (col.name === stratifiedFileData.suggested_column) {
                    option.selected = true;
                }
                columnSelect.appendChild(option);
            }
        });

        // Show column section
        document.getElementById('stratColumnSection').style.display = 'block';

        // If column is pre-selected, show config
        if (stratifiedFileData.suggested_column) {
            document.getElementById('stratConfigSection').style.display = 'block';
            document.getElementById('stratButtonGroup').style.display = 'flex';
        }

    } catch (error) {
        alert('Error parsing file: ' + error.message);
    }
}

// Shuffle files handler
function handleShuffleFiles() {
    const shuffleFilesInput = document.getElementById('shuffleFiles');
    const shuffleFilesPreview = document.getElementById('shuffleFilesPreview');
    const shuffleFilesList = document.getElementById('shuffleFilesList');
    const shuffleUploadContent = document.getElementById('shuffleUploadContent');
    const shuffleFilesArea = document.getElementById('shuffleFilesArea');

    if (!shuffleFilesInput.files || shuffleFilesInput.files.length < 2) {
        if (shuffleFilesInput.files.length === 1) {
            alert('Please select at least 2 files');
        }
        return;
    }

    const fileNames = Array.from(shuffleFilesInput.files).map(f => f.name);
    shuffleFilesList.innerHTML = `<span class="file-name">${shuffleFilesInput.files.length} files selected:</span><br><span style="color: #94a3b8; font-size: 0.85rem;">${fileNames.join(', ')}</span>`;
    shuffleFilesPreview.style.display = 'flex';
    shuffleUploadContent.style.display = 'none';
    shuffleFilesArea.classList.add('has-file');
    document.getElementById('shuffleConfigSection').style.display = 'block';
    document.getElementById('shuffleButtonGroup').style.display = 'flex';
}

function initSamplingUI() {
    // Only run on sampling page
    if (!document.getElementById('stratifiedForm')) return;

    // Tab switching
    document.querySelectorAll('.tool-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tool-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tool-panel').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab + 'Panel').classList.add('active');
        });
    });

    // Mode toggle for stratified sampling
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;

            document.getElementById('quantileConfig').style.display = currentMode === 'quantile' ? 'block' : 'none';
            document.getElementById('customConfig').style.display = currentMode === 'custom' ? 'block' : 'none';
        });
    });

    // Update weights when bins change
    document.getElementById('bins').addEventListener('change', updateWeightsGrid);

    // Close modal on overlay click
    document.querySelectorAll('.modal-overlay').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.remove('open');
        });
    });

    //  Stratified File Upload
    const stratFileArea = document.getElementById('stratFileArea');
    const stratFileInput = document.getElementById('stratFile');
    const stratFilePreview = document.getElementById('stratFilePreview');
    const stratFileName = document.getElementById('stratFileName');
    const stratUploadContent = document.getElementById('stratUploadContent');

    stratFileArea.addEventListener('click', () => stratFileInput.click());
    stratFileArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        stratFileArea.style.borderColor = '#3b82f6';
    });
    stratFileArea.addEventListener('dragleave', () => {
        stratFileArea.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    });
    stratFileArea.addEventListener('drop', (e) => {
        e.preventDefault();
        stratFileArea.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        if (e.dataTransfer.files.length > 0) {
            stratFileInput.files = e.dataTransfer.files;
            handleStratifiedFile();
        }
    });

    stratFileInput.addEventListener('change', handleStratifiedFile);

    document.getElementById('removeStratFile').addEventListener('click', (e) => {
        e.stopPropagation();
        stratFileInput.value = '';
        stratifiedFileData = null;
        stratFilePreview.style.display = 'none';
        stratUploadContent.style.display = 'block';
        stratFileArea.classList.remove('has-file');
        document.getElementById('stratFileInfo').style.display = 'none';
        document.getElementById('stratColumnSection').style.display = 'none';
        document.getElementById('stratConfigSection').style.display = 'none';
        document.getElementById('stratButtonGroup').style.display = 'none';
    });

    // Column selection change
    document.getElementById('stratColumn').addEventListener('change', function() {
        if (this.value) {
            document.getElementById('stratConfigSection').style.display = 'block';
            document.getElementById('stratButtonGroup').style.display = 'flex';
        } else {
            document.getElementById('stratConfigSection').style.display = 'none';
            document.getElementById('stratButtonGroup').style.display = 'none';
        }
    });

    //  Stratified Form Submit
    document.getElementById('stratifiedForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('stratifiedBtn');
        const btnText = btn.querySelector('.btn-text');
        btn.disabled = true;
        btnText.textContent = 'Processing...';

        try {
            const formData = new FormData();
            formData.append('file', stratFileInput.files[0]);
            formData.append('score_col', document.getElementById('stratColumn').value);
            formData.append('mode', currentMode);
            formData.append('allocation', document.getElementById('allocation').value);
            formData.append('random_state', document.getElementById('randomSeed').value);

            // Collect weights if checkbox is checked and not fixed allocation
            const allocation = document.getElementById('allocation').value;
            const useWeights = document.getElementById('useWeights').checked;
            if (allocation !== 'fixed' && useWeights) {
                const weights = {};
                document.querySelectorAll('.weight-input').forEach(input => {
                    const group = parseInt(input.dataset.group);
                    const weight = parseFloat(input.value) || 1.0;
                    weights[group] = weight;
                });
                formData.append('weights', JSON.stringify(weights));
            }

            if (currentMode === 'quantile') {
                formData.append('n_total', document.getElementById('nTotal').value);
                formData.append('bins', document.getElementById('bins').value);
            } else {
                // Custom ranges mode
                const ranges = [];
                const fixedCounts = {};
                let rangeIdx = 0;

                document.querySelectorAll('.range-row').forEach(row => {
                    const min = row.querySelector('.range-min').value;
                    const max = row.querySelector('.range-max').value;
                    const fixedCount = row.querySelector('.range-fixed-count')?.value;

                    if (min !== '' || max !== '') {
                        ranges.push({
                            lower: min !== '' ? parseFloat(min) : -Infinity,
                            upper: max !== '' ? parseFloat(max) : Infinity,
                            lower_inclusive: true,
                            upper_inclusive: true
                        });

                        // Collect fixed count if using fixed allocation
                        if (allocation === 'fixed' && fixedCount !== '') {
                            fixedCounts[rangeIdx] = parseInt(fixedCount) || 0;
                        }
                        rangeIdx++;
                    }
                });

                formData.append('ranges', JSON.stringify(ranges));

                if (allocation === 'fixed') {
                    formData.append('fixed_counts', JSON.stringify(fixedCounts));
                    formData.append('n_total', '0'); // Will be calculated from fixed_counts
                } else {
                    formData.append('n_total', document.getElementById('nTotalCustom').value || '0');
                }
            }

            const response = await fetch('/api/sampling/stratified', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Processing failed');
            }

            // Download the file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.headers.get('Content-Disposition')?.split('filename=')[1]?.replace(/"/g, '') || 'stratified_sample.zip';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

            // GA4: track wordlist generation
            if (typeof gtag === 'function') {
                const nTotal = document.getElementById('nTotal')?.value || document.getElementById('nTotalCustom')?.value || '0';
                gtag('event', 'wordlist_generated', {
                    word_count: parseInt(nTotal) || 0
                });
            }

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            btn.disabled = false;
            btnText.textContent = 'Generate Sample';
        }
    });

    // Shuffle File Upload
    const shuffleFilesArea = document.getElementById('shuffleFilesArea');
    const shuffleFilesInput = document.getElementById('shuffleFiles');
    const shuffleFilesPreview = document.getElementById('shuffleFilesPreview');
    const shuffleFilesList = document.getElementById('shuffleFilesList');
    const shuffleUploadContent = document.getElementById('shuffleUploadContent');

    shuffleFilesArea.addEventListener('click', () => shuffleFilesInput.click());
    shuffleFilesArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        shuffleFilesArea.style.borderColor = '#3b82f6';
    });
    shuffleFilesArea.addEventListener('dragleave', () => {
        shuffleFilesArea.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    });
    shuffleFilesArea.addEventListener('drop', (e) => {
        e.preventDefault();
        shuffleFilesArea.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        if (e.dataTransfer.files.length > 0) {
            shuffleFilesInput.files = e.dataTransfer.files;
            handleShuffleFiles();
        }
    });

    shuffleFilesInput.addEventListener('change', handleShuffleFiles);

    document.getElementById('removeShuffleFiles').addEventListener('click', (e) => {
        e.stopPropagation();
        shuffleFilesInput.value = '';
        shuffleFilesPreview.style.display = 'none';
        shuffleUploadContent.style.display = 'block';
        shuffleFilesArea.classList.remove('has-file');
        document.getElementById('shuffleConfigSection').style.display = 'none';
        document.getElementById('shuffleButtonGroup').style.display = 'none';
    });

    // Shuffle Form Submit
    document.getElementById('shuffleForm').addEventListener('submit', async (e) => {
        e.preventDefault();

        const btn = document.getElementById('shuffleBtn');
        const btnText = btn.querySelector('.btn-text');
        btn.disabled = true;
        btnText.textContent = 'Shuffling...';

        try {
            const formData = new FormData();
            Array.from(shuffleFilesInput.files).forEach(file => {
                formData.append('files', file);
            });
            formData.append('seed', document.getElementById('shuffleSeed').value);

            const response = await fetch('/api/sampling/shuffle', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Shuffling failed');
            }

            // Download the ZIP file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'shuffled_files.zip';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            btn.disabled = false;
            btnText.textContent = 'Shuffle Files';
        }
    });

    // Navigation toggle for mobile
    document.getElementById('navToggle')?.addEventListener('click', () => {
        document.querySelector('.nav-menu')?.classList.toggle('active');
    });
}

// ============== Progress Indicator ==============
const progressElements = {
    container: null,
    fill: null,
    percent: null,
    stages: null
};

function initProgressElements() {
    progressElements.container = document.getElementById('progressContainer');
    progressElements.fill = document.getElementById('progressFill');
    progressElements.percent = document.getElementById('progressPercent');
    progressElements.stages = {
        1: document.getElementById('stage1'),
        2: document.getElementById('stage2'),
        3: document.getElementById('stage3'),
        4: document.getElementById('stage4')
    };
}

function showProgress() {
    if (!progressElements.container) initProgressElements();
    if (progressElements.container) {
        progressElements.container.style.display = 'block';
        updateProgress(0, 1);
    }
    if (elements.cancelProcessBtn) {
        elements.cancelProcessBtn.style.display = 'inline-flex';
    }
}

function hideProgress() {
    if (progressElements.container) {
        progressElements.container.style.display = 'none';
    }
    // Reset all stages
    Object.values(progressElements.stages || {}).forEach(stage => {
        if (stage) {
            stage.classList.remove('active', 'completed');
        }
    });
    if (elements.cancelProcessBtn) {
        elements.cancelProcessBtn.style.display = 'none';
    }
}

function cancelProcessing() {
    if (!state.isProcessing) return;

    state.isProcessing = false;

    // Reset UI
    if (elements.processBtn) {
        elements.processBtn.classList.remove('loading');
        elements.processBtn.disabled = false;
    }
    hideProgress();
    clearFile();

    showToast('Processing cancelled', 'info');
}

function updateProgress(percent, activeStage) {
    if (progressElements.fill) {
        progressElements.fill.style.width = `${percent}%`;
    }
    if (progressElements.percent) {
        progressElements.percent.textContent = `${Math.round(percent)}%`;
    }

    // Update stages
    Object.entries(progressElements.stages || {}).forEach(([num, stage]) => {
        if (!stage) return;
        const stageNum = parseInt(num);
        stage.classList.remove('active', 'completed');

        if (stageNum < activeStage) {
            stage.classList.add('completed');
        } else if (stageNum === activeStage) {
            stage.classList.add('active');
        }
    });
}

// ============== Process File ==============
async function processFile() {
    console.log('[ProcessFile] Starting...', {
        isProcessing: state.isProcessing,
        uploadedFile: state.uploadedFile?.name,
        language: state.selectedLanguage,
        tool: state.selectedTool,
        warmupStatus: state.warmupStatus
    });

    // Check if warmup is complete for this tool
    if (state.selectedLanguage === 'fa' && state.selectedTool !== 'syllables') {
        if (!state.warmupStatus.completed) {
            console.log('[ProcessFile] Warmup not complete - models may still be loading');
        }
        if (!state.warmupStatus.loaded.includes('fa_g2p')) {
            console.log('[ProcessFile] WARNING: fa_g2p not in loaded models:', state.warmupStatus.loaded);
        }
    }

    // Check if already processing
    if (state.isProcessing) {
        console.log('[ProcessFile] Already processing, returning');
        return;
    }

    // Early validation with user feedback
    if (!state.uploadedFile) {
        showToast('Please upload a file first', 'error');
        return;
    }
    if (!state.selectedLanguage) {
        showToast('Please select a language', 'error');
        return;
    }
    if (!state.selectedTool) {
        showToast('Please select a tool', 'error');
        return;
    }

    if (!elements.processBtn) return;

    // Show progress and disable button
    state.isProcessing = true;
    elements.processBtn.classList.add('loading');
    elements.processBtn.disabled = true;
    showProgress();
    const processStartTime = Date.now();

    try {
        // Stage 1: Reading file
        updateProgress(10, 1);
        await new Promise(r => setTimeout(r, 300));

        const formData = new FormData();
        formData.append('file', state.uploadedFile);
        formData.append('language', state.selectedLanguage);
        formData.append('tool', state.selectedTool);
        formData.append('word_column', state.wordColumn || 'word');

        // Stage 2: Processing words
        updateProgress(25, 2);

        if (!state.isProcessing) {
            return; // User cancelled
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log('[ProcessFile] Timeout reached, aborting...');
            controller.abort();
        }, 300000); // 5 minute timeout

        console.log('[ProcessFile] Sending fetch request to /api/process-file...');
        console.log('[ProcessFile] FormData contents:', {
            file: state.uploadedFile?.name,
            language: state.selectedLanguage,
            tool: state.selectedTool,
            word_column: state.wordColumn
        });
        let response;
        try {
            response = await fetch('/api/process-file', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            console.log('[ProcessFile] Got response:', response.status, response.statusText);
            console.log('[ProcessFile] Response headers:', {
                contentType: response.headers.get('content-type'),
                contentDisposition: response.headers.get('content-disposition')
            });
        } catch (fetchError) {
            console.error('[ProcessFile] Fetch error:', fetchError);
            console.error('[ProcessFile] Error name:', fetchError.name);
            console.error('[ProcessFile] Error message:', fetchError.message);
            throw fetchError;
        } finally {
            clearTimeout(timeoutId);
        }

        // Stage 3: Generating output
        updateProgress(70, 3);

        if (!state.isProcessing) {
            return; // User cancelled
        }

        const contentType = response.headers.get('content-type');

        if (!response.ok) {
            let errorMessage = 'Processing failed';
            if (contentType && contentType.includes('application/json')) {
                const errorData = await response.json();
                errorMessage = errorData.error || errorMessage;
            }
            throw new Error(errorMessage);
        }

        // Get the processed file as blob
        const blob = await response.blob();

        if (!state.isProcessing) {
            return; // User cancelled
        }

        updateProgress(90, 3);

        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'lexprep_results.xlsx';
        if (contentDisposition) {
            const match = contentDisposition.match(/filename="?([^";\n]+)"?/);
            if (match) filename = match[1];
        }

        // Stage 4: Complete
        updateProgress(100, 4);
        await new Promise(r => setTimeout(r, 500));

        // Trigger download
        downloadBlob(blob, filename);

        // GA4: track successful analysis
        if (typeof gtag === 'function') {
            const wordCount = parseInt(response.headers.get('X-Word-Count')) || 0;
            const durationSec = Math.round((Date.now() - processStartTime) / 1000);
            gtag('event', 'analysis_run', {
                analysis_type: state.selectedTool,
                language: state.selectedLanguage,
                item_count: wordCount,
                duration_sec: durationSec,
                file_size: state.uploadedFile?.size || 0
            });
        }

        showToast('Processing complete! Download started.', 'success');

        // Hide progress after a moment
        setTimeout(() => {
            hideProgress();
        }, 1500);

    } catch (error) {
        console.error('Processing error:', error);
        showToast(error.message || 'Processing failed. Please try again.', 'error');
        hideProgress();

        // GA4: track error
        if (typeof gtag === 'function') {
            const errorType = error.name === 'AbortError' ? 'timeout' : (error.message?.includes('validation') ? 'validation' : 'server');
            gtag('event', 'error_occurred', {
                error_type: errorType,
                analysis_type: state.selectedTool,
                language: state.selectedLanguage
            });
        }
    } finally {
        state.isProcessing = false;
        elements.processBtn.classList.remove('loading');
        updateProcessButton();
    }
}

function downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// ============== Async Processing (for very slow operations) ==============
async function processFileAsync() {
    if (state.isProcessing) return;

    if (!state.uploadedFile || !state.selectedLanguage || !state.selectedTool) {
        showToast('Please select language, tool, and upload a file', 'error');
        return;
    }

    state.isProcessing = true;
    if (elements.processBtn) {
        elements.processBtn.classList.add('loading');
        elements.processBtn.disabled = true;
    }
    showProgress();
    updateProgress(5, 1);
    const asyncStartTime = Date.now();

    try {
        const formData = new FormData();
        formData.append('file', state.uploadedFile);
        formData.append('language', state.selectedLanguage);
        formData.append('tool', state.selectedTool);
        formData.append('word_column', state.wordColumn || 'word');

        // Start async processing
        const startResponse = await fetch('/api/process-file-async', {
            method: 'POST',
            body: formData
        });

        if (!startResponse.ok) {
            const errorData = await startResponse.json();
            throw new Error(errorData.error || 'Failed to start processing');
        }

        const { job_id } = await startResponse.json();
        state.currentJobId = job_id;

        // Poll for job completion
        await pollJobStatus(job_id, asyncStartTime);

    } catch (error) {
        console.error('Async processing error:', error);
        showToast(error.message || 'Processing failed', 'error');
        hideProgress();

        // GA4: track error
        if (typeof gtag === 'function') {
            const errorType = error.message?.includes('timed out') ? 'timeout' : 'server';
            gtag('event', 'error_occurred', {
                error_type: errorType,
                analysis_type: state.selectedTool,
                language: state.selectedLanguage
            });
        }
    } finally {
        state.isProcessing = false;
        state.currentJobId = null;
        if (elements.processBtn) {
            elements.processBtn.classList.remove('loading');
        }
        updateProcessButton();
    }
}

async function pollJobStatus(jobId, startTime) {
    const maxAttempts = 360; // 30 minutes max (5 sec intervals)
    let attempts = 0;

    while (attempts < maxAttempts && state.isProcessing) {
        try {
            const response = await fetch(`/api/job/${jobId}`);
            const job = await response.json();

            if (job.error && job.status === 'failed') {
                throw new Error(job.error);
            }

            // Update progress
            updateProgress(job.progress, Math.ceil(job.progress / 25) || 1);

            if (job.status === 'completed') {
                // Download the result
                updateProgress(100, 4);
                await new Promise(r => setTimeout(r, 300));

                const downloadResponse = await fetch(`/api/job/${jobId}/download`);
                const blob = await downloadResponse.blob();

                const contentDisposition = downloadResponse.headers.get('Content-Disposition');
                let filename = 'lexprep_results.xlsx';
                if (contentDisposition) {
                    const match = contentDisposition.match(/filename="?([^";\n]+)"?/);
                    if (match) filename = match[1];
                }

                downloadBlob(blob, filename);

                // GA4: track successful async analysis
                if (typeof gtag === 'function') {
                    const wordCount = parseInt(downloadResponse.headers.get('X-Word-Count')) || 0;
                    const durationSec = Math.round((Date.now() - startTime) / 1000);
                    gtag('event', 'analysis_run', {
                        analysis_type: state.selectedTool,
                        language: state.selectedLanguage,
                        item_count: wordCount,
                        duration_sec: durationSec,
                        file_size: state.uploadedFile?.size || 0
                    });
                }

                showToast('Processing complete! Download started.', 'success');

                setTimeout(() => hideProgress(), 1500);
                return;
            }

            // Wait before next poll
            await new Promise(r => setTimeout(r, 5000));
            attempts++;

        } catch (error) {
            throw error;
        }
    }

    if (attempts >= maxAttempts) {
        throw new Error('Processing timed out. Please try with a smaller file.');
    }
}

// ============== Toast Notifications ==============
function showToast(message, type = 'info') {
    if (!elements.toast) return;

    const icons = {
        success: '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg>',
        error: '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>',
        info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>'
    };

    elements.toast.querySelector('.toast-icon').innerHTML = icons[type] || icons.info;
    elements.toast.querySelector('.toast-message').textContent = message;
    elements.toast.className = `toast ${type}`;

    // Show toast
    setTimeout(() => elements.toast.classList.add('show'), 10);

    // Hide after 4 seconds
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 4000);
}

// ============== Utility Functions ==============
function formatColumnName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// ============== Admin Functions ==============
async function loginAdmin(secret) {
    try {
        const response = await fetch('/api/admin/stats', {
            headers: {
                'Authorization': `Bearer ${secret}`
            }
        });

        if (!response.ok) {
            throw new Error('Invalid credentials');
        }

        const data = await response.json();
        return data;

    } catch (error) {
        throw error;
    }
}

// Export for admin page
window.lexprep = {
    loginAdmin,
    showToast
};
