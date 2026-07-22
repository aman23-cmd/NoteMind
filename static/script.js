// NoteMind Frontend Logic
const API_BASE_URL = "https://notemind-g3rb.onrender.com";

document.addEventListener('DOMContentLoaded', () => {

    // Current Active State
    let currentVideoId = null;
    let currentVideoTitle = '';
    let currentVideoUrl = '';

    // Cache DOM Elements
    const processForm = document.getElementById('process-form');
    const videoUrlInput = document.getElementById('video-url');
    const btnProcess = document.getElementById('btn-process');
    const btnProcessLoader = btnProcess.querySelector('.btn-loader');
    
    const loadingContainer = document.getElementById('loading-container');
    const loadingText = document.getElementById('loading-text');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    const historyList = document.getElementById('history-list');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');

    const dashboard = document.getElementById('dashboard');
    const metaTitle = document.getElementById('meta-title');
    const metaUrl = document.getElementById('meta-url');

    const tabNotes = document.getElementById('tab-notes');
    const tabSearch = document.getElementById('tab-search');
    const panelNotes = document.getElementById('panel-notes');
    const panelSearch = document.getElementById('panel-search');

    const notesContent = document.getElementById('notes-content');
    const notesProvider = document.getElementById('notes-provider');
    const notesMethod = document.getElementById('notes-method');
    const btnRegenerateNotes = document.getElementById('btn-regenerate-notes');
    const btnDownloadNotes = document.getElementById('btn-download-notes');

    const searchForm = document.getElementById('search-form');
    const searchQuery = document.getElementById('search-query');
    const btnSearch = document.getElementById('btn-search');
    const searchLoading = document.getElementById('search-loading');
    const searchResultContainer = document.getElementById('search-result-container');
    const searchAnswer = document.getElementById('search-answer');
    const searchProvider = document.getElementById('search-provider');
    const searchTimestampLink = document.getElementById('search-timestamp-link');
    const sourceChunksList = document.getElementById('source-chunks-list');

    // Configure Marked Options
    if (typeof marked !== 'undefined') {
        marked.setOptions({
            gfm: true,
            breaks: true,
            headerIds: false,
            mangle: false
        });
    }

    // Initialize App
    loadHistory();

    // Event Listeners
    processForm.addEventListener('submit', handleProcessSubmit);
    btnRefreshHistory.addEventListener('click', loadHistory);
    searchForm.addEventListener('submit', handleSearchSubmit);
    btnRegenerateNotes.addEventListener('click', () => loadNotes(currentVideoId, true));
    btnDownloadNotes.addEventListener('click', downloadMarkdownNotes);

    // Tab Navigation
    tabNotes.addEventListener('click', () => switchTab('notes'));
    tabSearch.addEventListener('click', () => switchTab('search'));

    // Functions

    /**
     * Handle Process Form Submission
     */
    async function handleProcessSubmit(e) {
        e.preventDefault();
        const url = videoUrlInput.value.trim();
        if (!url) return;

        showLoading('Initializing video processing...');
        hideError();
        hideDashboard();

        try {
            const response = await fetch(`${API_BASE_URL}/process-video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to process the video transcript.');
            }

            currentVideoId = result.video_id;
            
            // Re-fetch history to get the new title
            await loadHistory();
            
            // Find current video details in the loaded history list
            const foundVideo = findVideoInHistory(currentVideoId);
            currentVideoTitle = foundVideo ? foundVideo.title : 'Processed Video';
            currentVideoUrl = foundVideo ? foundVideo.url : url;

            hideLoading();
            showDashboard(currentVideoId, currentVideoTitle, currentVideoUrl);
            
            // Automatically fetch notes
            loadNotes(currentVideoId);

        } catch (error) {
            hideLoading();
            showError(error.message);
        }
    }

    /**
     * Load processed history from SQLite
     */
    async function loadHistory() {
        try {
            const response = await fetch(`${API_BASE_URL}/history`);
            const result = await response.json();

            if (result.success && result.videos) {
                renderHistory(result.videos);
            }
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }

    /**
     * Helper to find a video record from local history memory
     */
    let loadedVideos = [];
    function findVideoInHistory(videoId) {
        return loadedVideos.find(v => v.video_id === videoId);
    }

    /**
     * Render the History list in the sidebar
     */
    function renderHistory(videos) {
        loadedVideos = videos;
        if (videos.length === 0) {
            historyList.innerHTML = '<div class="empty-state">No videos processed yet.</div>';
            return;
        }

        historyList.innerHTML = '';
        videos.forEach(v => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.dataset.videoId = v.video_id;
            
            const displayTitle = v.title || `Video (${v.video_id})`;
            const statusClass = `status-${v.transcript_status}`;
            const statusLabel = v.transcript_status.toUpperCase();

            // Format date slightly
            let dateStr = '';
            if (v.created_at) {
                const parts = v.created_at.split(' ');
                dateStr = parts[0] || '';
            }

            item.innerHTML = `
                <div class="history-title" title="${displayTitle}">${displayTitle}</div>
                <div class="history-meta">
                    <span class="status-pill ${statusClass}">${statusLabel}</span>
                    <span>${dateStr}</span>
                </div>
            `;

            // Clicking history loads the dashboard directly (cache-friendly)
            item.addEventListener('click', () => {
                if (v.transcript_status === 'done') {
                    currentVideoId = v.video_id;
                    currentVideoTitle = v.title;
                    currentVideoUrl = v.url;
                    hideError();
                    showDashboard(currentVideoId, currentVideoTitle, currentVideoUrl);
                    loadNotes(currentVideoId);
                } else {
                    videoUrlInput.value = v.url;
                    // If it's pending/failed, trigger reprocessing
                    handleProcessSubmit({ preventDefault: () => {} });
                }
            });

            historyList.appendChild(item);
        });
    }

    /**
     * Fetch Notes from Flask Cache/LLM
     */
    async function loadNotes(videoId, forceRegenerate = false) {
        notesContent.innerHTML = '<div class="spinner"></div><p style="text-align:center;color:var(--text-muted);">Generating study notes...</p>';
        notesProvider.textContent = 'Provider: Loading...';
        notesMethod.textContent = 'Method: Loading...';
        
        try {
            const response = await fetch(`${API_BASE_URL}/generate-notes`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_id: videoId, force: forceRegenerate })
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to generate notes.');
            }

            // Render Markdown
            if (typeof marked !== 'undefined') {
                notesContent.innerHTML = marked.parse(result.notes);
            } else {
                // Fallback text formatting
                notesContent.innerHTML = `<pre style="white-space: pre-wrap; font-family:inherit;">${result.notes}</pre>`;
            }

            // Update Metadata labels
            notesProvider.textContent = `Provider: ${result.provider.toUpperCase()}`;
            notesMethod.textContent = `Method: ${result.method.toUpperCase()} ${result.cached ? '(Cached)' : ''}`;

        } catch (error) {
            notesContent.innerHTML = `
                <div class="error-container">
                    <span class="error-icon">⚠️</span>
                    <p>Failed to load notes: ${error.message}</p>
                </div>
            `;
        }
    }

    /**
     * Download notes as a Markdown (.md) file
     */
    function downloadMarkdownNotes() {
        // Extract raw text content (using markdown header title as file name)
        const markdownContent = getRawMarkdownNotes();
        if (!markdownContent) return;

        const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        
        link.href = url;
        link.setAttribute('download', `${currentVideoId}_notes.md`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    /**
     * Recovers text content of notes. Real implementations might save notes_content to window state.
     * We'll fetch it from notesContent container innerText or do a quick hack.
     */
    function getRawMarkdownNotes() {
        // Find clean markdown format headers and points from notesContent
        // We can just construct a basic markdown representation, or we can use the text inside notesContent
        return notesContent.innerText || '';
    }

    /**
     * Handle Topic Search Submission
     */
    async function handleSearchSubmit(e) {
        e.preventDefault();
        const query = searchQuery.value.trim();
        if (!query || !currentVideoId) return;

        searchLoading.classList.remove('hidden');
        searchResultContainer.classList.add('hidden');
        btnSearch.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_id: currentVideoId, query: query })
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to search video contents.');
            }

            renderSearchResults(result);

        } catch (error) {
            alert(`Search Error: ${error.message}`);
        } finally {
            searchLoading.classList.add('hidden');
            btnSearch.disabled = false;
        }
    }

    /**
     * Render the search result components
     */
    function renderSearchResults(result) {
        searchAnswer.textContent = result.answer;
        searchProvider.textContent = `Provider: ${result.provider.toUpperCase()}`;
        
        // Setup Best Match Timestamp Link
        const bestTimeSeconds = parseTimestampToSeconds(result.timestamp);
        const bestYoutubeLink = `https://www.youtube.com/watch?v=${currentVideoId}&t=${bestTimeSeconds}s`;
        searchTimestampLink.href = bestYoutubeLink;
        searchTimestampLink.textContent = result.timestamp;

        // Render Source Chunks list
        sourceChunksList.innerHTML = '';
        if (result.source_chunks && result.source_chunks.length > 0) {
            result.source_chunks.forEach((chunk, index) => {
                const chunkDiv = document.createElement('div');
                chunkDiv.className = 'source-chunk';
                
                const seconds = Math.floor(chunk.start_time);
                const chunkYoutubeLink = `https://www.youtube.com/watch?v=${currentVideoId}&t=${seconds}s`;

                chunkDiv.innerHTML = `
                    <div class="chunk-header">
                        <span>Segment ${index + 1}</span>
                        <div class="chunk-footer">
                            <span class="pill pill-interactive">
                                ⏱️ <a href="${chunkYoutubeLink}" target="_blank">Jump to ${chunk.timestamp}</a>
                            </span>
                            <span class="pill">Score: ${(1 - chunk.distance).toFixed(2)}</span>
                        </div>
                    </div>
                    <p class="chunk-text">"... ${chunk.chunk_text} ..."</p>
                `;
                sourceChunksList.appendChild(chunkDiv);
            });
        } else {
            sourceChunksList.innerHTML = '<div class="empty-state">No source segments returned.</div>';
        }

        searchResultContainer.classList.remove('hidden');
    }

    /**
     * Parse timestamp format (HH:MM:SS) to total seconds
     */
    function parseTimestampToSeconds(tsStr) {
        if (!tsStr) return 0;
        const parts = tsStr.split(':').map(Number);
        if (parts.length === 3) {
            return parts[0] * 3600 + parts[1] * 60 + parts[2];
        } else if (parts.length === 2) {
            return parts[0] * 60 + parts[1];
        }
        return 0;
    }

    /**
     * Switch Tabs: Notes vs Search
     */
    function switchTab(tabType) {
        if (tabType === 'notes') {
            tabNotes.classList.add('active');
            tabSearch.classList.remove('active');
            tabNotes.setAttribute('aria-selected', 'true');
            tabSearch.setAttribute('aria-selected', 'false');
            panelNotes.classList.add('active');
            panelSearch.classList.remove('active');
        } else {
            tabNotes.classList.remove('active');
            tabSearch.classList.add('active');
            tabNotes.setAttribute('aria-selected', 'false');
            tabSearch.setAttribute('aria-selected', 'true');
            panelNotes.classList.remove('active');
            panelSearch.classList.add('active');
        }
    }

    // UI Helpers

    function showLoading(msg) {
        loadingText.textContent = msg;
        loadingContainer.classList.remove('hidden');
        btnProcess.disabled = true;
        btnProcessLoader.classList.remove('hidden');
    }

    function hideLoading() {
        loadingContainer.classList.add('hidden');
        btnProcess.disabled = false;
        btnProcessLoader.classList.add('hidden');
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorContainer.classList.remove('hidden');
    }

    function hideError() {
        errorContainer.classList.add('hidden');
    }

    function showDashboard(videoId, title, url) {
        metaTitle.textContent = title || `Video (${videoId})`;
        metaUrl.href = url;
        dashboard.classList.remove('hidden');
        
        // Reset Search Tab fields
        searchQuery.value = '';
        searchResultContainer.classList.add('hidden');
        switchTab('notes');
    }

    function hideDashboard() {
        dashboard.classList.add('hidden');
    }
});
