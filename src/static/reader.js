// Reader tab: library, add dialog, reading view and a chained-part audio player.
// Loaded after app.js and reuses its helpers (byId, esc, showToast, api, readStorage, writeStorage,
// handleTabKeydown, getTTSModels, formatModelName, fetchVoices, selectedSpeechVoice, state).
(() => {
  const API = '/v1/reader/documents';
  const SPEEDS = [0.75, 1, 1.25, 1.5, 1.75, 2];
  const SPEED_KEY = 'open-speech-reader-speed';
  const VOICE_KEY = 'open-speech-reader-voice';
  const POLL_MS = 500;
  const LOOKAHEAD_POLL_MS = 1000;
  const RERENDER_AFTER_MS = 3000;
  const PREPARING_DELAY_MS = 1000;
  const SAVE_EVERY_MS = 10000;
  const BACK_SECONDS = 15;
  const FORWARD_SECONDS = 30;
  const CHARS_PER_SECOND = 15;   // rough 1x speaking rate, only for length estimates
  const FILE_TYPES = ['.epub', '.txt', '.md', '.markdown', '.html', '.htm'];
  const SOURCE_BADGES = {
    epub: ['Book', 'book'],
    markdown: ['Markdown', 'file'],
    html: ['Web page', 'link'],
    text: ['Text', 'paste'],
  };
  const SMALL_WORDS = new Set(['a', 'an', 'and', 'of', 'or', 'the', 'to', 'in', 'on', 'for', 'at', 'by']);

  const reader = {
    docs: [],
    loaded: false,
    loading: false,
    loadError: null,
    doc: null,
    blockEls: [],
    chapterAt: [],      // block index -> chapter position
    cursor: 0,          // current block index
    sentence: -1,       // current sentence index inside the cursor block
    chapter: -1,
    playing: false,     // the listener wants audio (also true while waiting for a part)
    waiting: false,     // a start or skip is waiting for its first part
    token: 0,           // bumped on every start, skip and stop so stale async work can bail out
    lookahead: 0,
    parts: [],          // latest planned window from the server
    active: null,       // { el, part } now playing (or paused)
    preloaded: null,    // { el, part } loaded into the idle element
    resumeOffsetMs: 0,
    follow: true,       // auto-scroll with playback until the listener scrolls away
    voice: null,        // { model, voice }
    speed: 1,
    unlocked: false,
    recovering: null,
    lastSaved: '',
    artwork: { id: null, url: '' },
    raf: 0,
    saveTimer: null,
    preparingTimer: null,
    addMode: 'file',
    addFile: null,
    silentUrl: '',
  };

  /* ── Small helpers ── */
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const icon = (name) => `<svg class="ico" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const docUrl = (id, suffix = '') => `${API}/${encodeURIComponent(id)}${suffix}`;
  const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  // Like app.js api(), but keeps the HTTP status so 404/503 can get a friendly message.
  async function readerApi(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      let message = `${res.status} ${res.statusText}`;
      try {
        const payload = await res.json();
        message = payload?.error?.message
          || payload?.detail?.message
          || (typeof payload?.detail === 'string' ? payload.detail : null)
          || message;
      } catch { }
      const error = new Error(message);
      error.status = res.status;
      throw error;
    }
    if (res.status === 204) return null;
    return (res.headers.get('content-type') || '').includes('application/json') ? res.json() : null;
  }

  function timeValue(value) {
    if (typeof value === 'number') return value < 1e12 ? value * 1000 : value;
    return Date.parse(value || '') || 0;
  }
  function hueFor(text) {
    let hue = 0;
    for (const ch of String(text || '')) hue = (hue * 31 + ch.codePointAt(0)) % 360;
    return hue;
  }
  function monogram(title) {
    const words = String(title || '').match(/[\p{L}\p{N}]+/gu) || [];
    const picked = words.filter((word, i) => i === 0 || !SMALL_WORDS.has(word.toLowerCase()));
    if (!picked.length) return '?';
    const letters = picked.length > 1 ? picked[0][0] + picked[1][0] : picked[0].slice(0, 2);
    return letters.toUpperCase();
  }
  function tile(doc, extraClass = '') {
    return `<span class="rd-tile${extraClass}" style="--tile:${hueFor(doc.id || doc.title)}" aria-hidden="true">${esc(monogram(doc.title))}</span>`;
  }
  function setTile(el, doc) {
    el.style.setProperty('--tile', hueFor(doc.id || doc.title));
    el.textContent = monogram(doc.title);
  }
  function percentOf(doc) {
    if (!doc.progress || !doc.block_count) return null;
    return clamp(Math.round((doc.progress.block_index / doc.block_count) * 100), 0, 100);
  }
  function lengthLabel(chars) {
    const minutes = Math.round((chars || 0) / CHARS_PER_SECOND / 60);
    if (minutes < 1) return 'under a minute';
    if (minutes < 60) return `~${minutes} min`;
    return `~${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} m`;
  }
  function plural(count, word) { return `${count} ${word}${count === 1 ? '' : 's'}`; }

  /* ── Library ── */
  async function loadLibrary() {
    reader.loading = true;
    if (!reader.loaded) renderLibrary();
    try {
      const data = await readerApi(API);
      reader.docs = Array.isArray(data?.documents) ? data.documents : [];
      reader.loaded = true;
      reader.loadError = null;
    } catch (error) {
      reader.loadError = error;
    } finally {
      reader.loading = false;
    }
    renderLibrary();
  }

  function setEmptyState({ title, text, action, error = false }) {
    const box = byId('rd-empty');
    box.hidden = false;
    box.classList.toggle('is-error', error);
    byId('rd-empty-title').textContent = title;
    byId('rd-empty-text').textContent = text;
    const button = byId('rd-empty-action');
    button.textContent = action;
    button.dataset.action = error ? 'retry' : 'add';
  }

  function renderLibrary() {
    const list = byId('rd-list');
    const hero = byId('rd-hero');
    const head = byId('rd-list-head');
    byId('rd-empty').hidden = true;
    hero.innerHTML = '';
    head.hidden = true;
    list.innerHTML = '';
    if (!reader.loaded && reader.loading) {
      list.innerHTML = '<li class="rd-loading"><span class="spin-dot" aria-hidden="true"></span>Loading library…</li>';
      return;
    }
    if (reader.loadError) {
      const status = reader.loadError.status;
      setEmptyState({
        title: 'Reader unavailable',
        text: status === 404
          ? "This server doesn't have the Reader yet. Update Open Speech to use it."
          : status === 503
            ? `The Reader is turned off or still starting. ${reader.loadError.message}`
            : `Couldn't load the library: ${reader.loadError.message}`,
        action: 'Try again',
        error: true,
      });
      return;
    }
    if (!reader.docs.length) {
      setEmptyState({
        title: 'Your library is empty',
        text: 'Add an EPUB, Markdown, text or HTML file, or paste some text, and Open Speech will read it aloud.',
        action: 'Add your first document',
      });
      return;
    }
    const query = byId('rd-search').value.trim().toLowerCase();
    const docs = query
      ? reader.docs.filter((doc) => `${doc.title || ''} ${doc.author || ''}`.toLowerCase().includes(query))
      : reader.docs;
    if (!query) {
      const recent = reader.docs
        .filter((doc) => doc.progress)
        .sort((a, b) => timeValue(b.progress.updated_at) - timeValue(a.progress.updated_at))[0];
      if (recent) hero.innerHTML = heroHtml(recent);
    }
    head.hidden = false;
    byId('rd-count').textContent = query
      ? `${plural(docs.length, 'match')} of ${reader.docs.length}`
      : plural(reader.docs.length, 'document');
    list.innerHTML = docs.length
      ? docs.map(rowHtml).join('')
      : `<li class="rd-none">No titles match “${esc(query)}”.</li>`;
  }

  function heroHtml(doc) {
    const pct = percentOf(doc) ?? 0;
    const left = lengthLabel((doc.char_count || 0) * (1 - pct / 100));
    const meta = [doc.author, `${pct}% listened`, `${left} left`].filter(Boolean).map(esc).join(' · ');
    return `<article class="card rd-hero" data-open="${esc(doc.id)}">
      ${tile(doc, ' rd-tile-lg')}
      <div class="rd-hero-body">
        <p class="rd-kicker">Continue listening</p>
        <h3 class="rd-hero-title">${esc(doc.title)}</h3>
        <p class="rd-meta">${meta}</p>
        <div class="rd-progress" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${pct}% listened"><span style="width:${pct}%"></span></div>
      </div>
      <button class="btn rd-resume" type="button" data-resume="${esc(doc.id)}" aria-label="Resume ${esc(doc.title)}">${icon('play')}Resume</button>
    </article>`;
  }

  function rowHtml(doc) {
    const [badge, badgeIcon] = SOURCE_BADGES[doc.source_type] || ['Text', 'file'];
    const pct = percentOf(doc);
    const meta = [
      doc.author,
      doc.chapter_count > 1 ? plural(doc.chapter_count, 'chapter') : plural(doc.block_count || 0, 'paragraph'),
      lengthLabel(doc.char_count),
    ].filter(Boolean).map(esc).join(' · ');
    return `<li class="rd-row">
      <button class="rd-row-main" type="button" data-open="${esc(doc.id)}">
        ${tile(doc)}
        <span class="rd-row-body">
          <span class="rd-row-title">${esc(doc.title)}</span>
          <span class="rd-meta"><span class="rd-badge">${icon(badgeIcon)}${esc(badge)}</span>${meta}</span>
          <span class="rd-progress rd-progress-sm" aria-hidden="true"><span style="width:${pct ?? 0}%"></span></span>
        </span>
        ${pct == null ? '<span class="rd-pct rd-pct-new">New</span>' : `<span class="rd-pct">${pct}%</span>`}
      </button>
      <button class="rd-iconbtn rd-delete" type="button" data-delete="${esc(doc.id)}" aria-label="Delete ${esc(doc.title)}">${icon('trash')}</button>
    </li>`;
  }

  async function deleteDocument(id) {
    const doc = reader.docs.find((item) => item.id === id);
    if (!window.confirm(`Delete “${doc?.title || 'this document'}” from the library? Its saved position is removed too.`)) return false;
    try {
      await readerApi(docUrl(id), { method: 'DELETE' });
    } catch (error) {
      showToast(`Delete failed: ${error.message}`, 'error');
      return false;
    }
    if (reader.doc?.id === id) {
      stopPlayback();
      reader.doc = null;
    }
    reader.docs = reader.docs.filter((item) => item.id !== id);
    renderLibrary();
    showToast(`Deleted “${doc?.title || 'document'}”`, 'success');
    return true;
  }

  function handleLibraryClick(event) {
    const resume = event.target.closest('[data-resume]');
    if (resume) {
      unlockAudio();
      openDocument(resume.dataset.resume, { autoplay: true });
      return;
    }
    const remove = event.target.closest('[data-delete]');
    if (remove) {
      deleteDocument(remove.dataset.delete);
      return;
    }
    const open = event.target.closest('[data-open]');
    if (open) openDocument(open.dataset.open);
  }

  /* ── Add dialog ── */
  function setAddMode(mode) {
    reader.addMode = mode;
    document.querySelectorAll('#rd-add-dialog .rd-seg-btn').forEach((button) => {
      const active = button.dataset.seg === mode;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
      button.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll('#rd-add-dialog .rd-seg-panel').forEach((panel) => {
      panel.hidden = panel.dataset.panel !== mode;
    });
  }

  function acceptedFile(file) {
    const name = (file?.name || '').toLowerCase();
    return FILE_TYPES.some((ext) => name.endsWith(ext));
  }

  function setAddFile(file) {
    if (file && !acceptedFile(file)) {
      showToast('Choose an EPUB, TXT, Markdown or HTML file', 'error');
      file = null;
    }
    reader.addFile = file || null;
    byId('rd-dropzone').classList.toggle('has-file', !!file);
    byId('rd-file-title').textContent = file ? file.name : 'Drop a book or text file';
    byId('rd-file-hint').textContent = file
      ? `${(file.size / 1024 / 1024).toFixed(file.size < 1024 * 1024 ? 2 : 1)} MB · click to choose another`
      : 'EPUB, TXT, Markdown or HTML · or click to choose';
  }

  function setAddBusy(busy) {
    const submit = byId('rd-add-submit');
    submit.disabled = busy;
    submit.classList.toggle('loading', busy);
    submit.textContent = busy ? (reader.addMode === 'file' ? 'Uploading…' : 'Adding…') : 'Add to library';
    byId('rd-add-form').setAttribute('aria-busy', busy ? 'true' : 'false');
  }

  function openAddDialog() {
    const dialog = byId('rd-add-dialog');
    if (!dialog.open) dialog.showModal();
  }

  async function submitAdd(event) {
    event.preventDefault();
    if (byId('rd-add-submit').disabled) return;
    let request;
    if (reader.addMode === 'file') {
      if (!reader.addFile) {
        showToast('Choose a file first', 'error');
        return;
      }
      const form = new FormData();
      form.append('file', reader.addFile);
      request = { method: 'POST', body: form };
    } else {
      const text = byId('rd-paste-text').value.trim();
      if (!text) {
        showToast('Paste some text first', 'error');
        byId('rd-paste-text').focus();
        return;
      }
      const title = byId('rd-paste-title').value.trim();
      request = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'text', text, ...(title ? { title } : {}) }),
      };
    }
    setAddBusy(true);
    try {
      const doc = await readerApi(API, request);
      const existed = reader.docs.some((item) => item.id === doc?.id);
      byId('rd-add-dialog').close();
      setAddFile(null);
      byId('rd-file').value = '';
      byId('rd-paste-title').value = '';
      byId('rd-paste-text').value = '';
      showToast(existed ? `Already in your library: “${doc.title}”` : `Added “${doc?.title || 'document'}”`, 'success');
      await loadLibrary();
    } catch (error) {
      showToast(`Couldn't add: ${error.message}`, 'error');
    } finally {
      setAddBusy(false);
    }
  }

  /* ── Reading view ── */
  function readerTabActive() { return !!byId('panel-reader')?.classList.contains('active'); }
  function readingVisible() { return readerTabActive() && !byId('rd-reading').hidden; }
  function syncBodyClass() { document.body.classList.toggle('is-reading', readingVisible()); }
  function setHash(hash) {
    const url = `${location.pathname}${location.search}${hash}`;
    if (`${location.pathname}${location.search}${location.hash}` !== url) history.replaceState(null, '', url);
  }

  function showReading() {
    byId('rd-library').hidden = true;
    byId('rd-reading').hidden = false;
    syncBodyClass();
    if (reader.doc) setHash(`#reader/${encodeURIComponent(reader.doc.id)}`);
  }

  async function showLibrary() {
    const saving = saveProgress();
    stopPlayback();
    byId('rd-reading').hidden = true;
    byId('rd-library').hidden = false;
    syncBodyClass();
    setHash('#reader');
    window.scrollTo(0, 0);
    await saving;
    loadLibrary();
  }

  async function openDocument(id, { autoplay = false } = {}) {
    if (reader.doc?.id === id && byId('rd-text').childElementCount) {
      showReading();
      if (autoplay && !reader.playing) play();
      return;
    }
    if (reader.doc) {
      saveProgress();
      stopPlayback();
    }
    let doc;
    try {
      doc = await readerApi(docUrl(id));
    } catch (error) {
      showToast(error.status === 404 ? 'That document is no longer in the library' : `Couldn't open the document: ${error.message}`, 'error');
      if (!byId('rd-reading').hidden) showLibrary();
      return;
    }
    doc.blocks = Array.isArray(doc.blocks) ? doc.blocks : [];
    doc.chapters = Array.isArray(doc.chapters) ? doc.chapters : [];
    reader.doc = doc;
    reader.lastSaved = '';
    reader.follow = true;
    reader.voice = pickVoice(doc.progress);
    const saved = doc.progress;
    const sameVoice = saved && reader.voice && saved.model === reader.voice.model && saved.voice === reader.voice.voice;
    reader.resumeOffsetMs = sameVoice ? Math.max(0, Number(saved.offset_ms) || 0) : 0;
    renderReading();
    showReading();
    window.scrollTo(0, 0);
    if (doc.blocks.length) setCursor(saved?.block_index ?? 0, -1, { scroll: true });
    reader.lastSaved = doc.blocks.length ? JSON.stringify(progressPayload()) : '';
    if (autoplay) play();
  }

  function renderReading() {
    const doc = reader.doc;
    const count = doc.blocks.length;
    reader.chapterAt = new Array(count).fill(0);
    doc.chapters.forEach((chapter, position) => {
      const end = Math.min(count, chapter.first_block + Math.max(1, chapter.block_count || 0));
      for (let i = Math.max(0, chapter.first_block); i < end; i += 1) reader.chapterAt[i] = position;
    });
    const text = byId('rd-text');
    text.innerHTML = count
      ? doc.blocks.map((block, i) => {
        const tag = block.kind === 'heading' ? 'h2' : 'p';
        const cls = block.kind === 'heading' ? 'rd-p rd-chapter-title' : 'rd-p';
        return `<${tag} class="${cls}" data-block="${i}" tabindex="-1">${esc(block.text)}</${tag}>`;
      }).join('')
      : '<p class="rd-text-empty">This document has no readable text.</p>';
    reader.blockEls = [...text.querySelectorAll('[data-block]')];
    reader.cursor = 0;
    reader.sentence = -1;
    reader.chapter = -1;

    const chapters = doc.chapters.map((chapter, position) => `<li data-chapter-item="${position}"><button type="button" data-chapter="${position}"><span class="rd-ch-n">${position + 1}</span><span class="rd-ch-t">${esc(chapter.title || `Chapter ${position + 1}`)}</span></button></li>`).join('');
    byId('rd-chapter-list').innerHTML = chapters;
    byId('rd-chapter-list-sheet').innerHTML = chapters;
    byId('rd-top-title').textContent = doc.title;
    byId('rd-np-title').textContent = doc.title;
    byId('rd-side-title').textContent = doc.title;
    setTile(byId('rd-side-tile'), doc);
    setTile(byId('rd-np-tile'), doc);
    updateSideMeta();
    updateVoiceChip();
    updateChapter();
  }

  function updateSideMeta() {
    const doc = reader.doc;
    if (!doc) return;
    const pct = doc.blocks.length ? Math.round((reader.cursor / doc.blocks.length) * 100) : 0;
    byId('rd-side-meta').textContent = [doc.author, `${pct}%`].filter(Boolean).join(' · ');
  }

  function currentChapter() {
    const doc = reader.doc;
    return doc ? doc.chapters[reader.chapterAt[reader.cursor] ?? 0] || null : null;
  }

  function updateChapter() {
    const position = reader.chapterAt[reader.cursor] ?? 0;
    if (position === reader.chapter) return;
    reader.chapter = position;
    const title = currentChapter()?.title || '';
    byId('rd-top-chapter').textContent = title;
    byId('rd-np-chapter').textContent = [title, reader.doc?.author].filter(Boolean).join(' · ');
    document.querySelectorAll('[data-chapter-item]').forEach((item) => {
      const index = Number(item.dataset.chapterItem);
      item.classList.toggle('is-current', index === position);
      item.classList.toggle('is-done', index < position);
      const button = item.querySelector('button');
      if (index === position) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    });
    if (reader.playing) updateMediaMetadata();
  }

  // Sentence offsets come from the server as code-point indices; slice by code point when the text has astral characters.
  function codePointSlicer(text) {
    if (!/[\uD800-\uDFFF]/.test(text)) return (start, end) => text.slice(start, end);
    const points = Array.from(text);
    return (start, end) => points.slice(start, end).join('');
  }

  function ensureSentenceSpans(index) {
    const el = reader.blockEls[index];
    const block = reader.doc?.blocks[index];
    if (!el || !block || el.dataset.split) return;
    el.dataset.split = '1';
    const sentences = Array.isArray(block.sentences) ? block.sentences : [];
    if (!sentences.length) return;
    const slice = codePointSlicer(block.text || '');
    let html = '';
    let pos = 0;
    sentences.forEach(([start, end], i) => {
      if (start > pos) html += esc(slice(pos, start));
      html += `<span class="rd-s" data-s="${i}">${esc(slice(start, end))}</span>`;
      pos = Math.max(pos, end);
    });
    html += esc(slice(pos));
    el.innerHTML = html;
  }

  function setCursor(index, sentence = -1, { scroll = false } = {}) {
    const doc = reader.doc;
    if (!doc?.blocks.length) return;
    index = clamp(index, 0, doc.blocks.length - 1);
    const previous = reader.blockEls[reader.cursor];
    const el = reader.blockEls[index];
    if (index !== reader.cursor || !el.classList.contains('is-current')) {
      if (previous) {
        previous.classList.remove('is-current');
        previous.tabIndex = -1;
        previous.querySelector('.rd-s.is-now')?.classList.remove('is-now');
      }
      reader.cursor = index;
      reader.sentence = -1;
      el.classList.add('is-current');
      el.tabIndex = 0;
      updateChapter();
      updateSideMeta();
    }
    if (sentence !== reader.sentence) {
      el.querySelector('.rd-s.is-now')?.classList.remove('is-now');
      if (sentence >= 0) {
        ensureSentenceSpans(index);
        el.querySelector(`.rd-s[data-s="${sentence}"]`)?.classList.add('is-now');
      }
      reader.sentence = sentence;
    }
    if (scroll || reader.follow) keepInView();
    updateReturnPill();
  }

  // Scroll the current sentence (or paragraph) back into the band between the top bar and the transport.
  function keepInView(force = false) {
    const block = reader.blockEls[reader.cursor];
    if (!block || !readingVisible()) return;
    const target = block.querySelector('.rd-s.is-now') || block;
    const rect = target.getBoundingClientRect();
    const topbar = document.querySelector('.rd-topbar');
    const top = (topbar && getComputedStyle(topbar).display !== 'none' ? topbar.getBoundingClientRect().bottom : 0) + 16;
    const transport = document.querySelector('.rd-transport');
    const bottom = (transport ? transport.getBoundingClientRect().top : window.innerHeight) - 16;
    if (!force && rect.top >= top && rect.bottom <= bottom) return;
    const room = bottom - top;
    const delta = rect.height > room ? rect.top - top : rect.top - (top + (room - rect.height) / 2);
    window.scrollBy({ top: delta, behavior: reducedMotion() ? 'auto' : 'smooth' });
  }

  function updateReturnPill() {
    byId('rd-return').hidden = reader.follow || !reader.playing;
  }

  function noteManualScroll(event) {
    if (!readingVisible() || !reader.playing) return;
    if (event?.target?.closest?.('dialog, .rd-side, .rd-transport, .rd-topbar')) return;
    if (!reader.follow) return;
    reader.follow = false;
    updateReturnPill();
  }

  function returnToPlaying() {
    reader.follow = true;
    updateReturnPill();
    keepInView(true);
  }

  /* ── Voice and speed ── */
  function availableModels() {
    try { return getTTSModels(); } catch { return []; }
  }
  function modelAvailable(id) { return !!id && availableModels().some((model) => model.id === id); }
  function speakVoice() {
    const model = byId('tts-model')?.value || '';
    if (!modelAvailable(model)) return null;
    let voice = '';
    try { voice = selectedSpeechVoice(); } catch { }
    return { model, voice };
  }
  function storedVoice() {
    try {
      const value = JSON.parse(readStorage(VOICE_KEY, 'null'));
      return value?.model && value?.voice ? { model: value.model, voice: value.voice } : null;
    } catch {
      return null;
    }
  }
  // The document's saved voice first, then the last Reader choice, then whatever Speak has selected.
  function pickVoice(progress) {
    if (progress?.model && progress?.voice && modelAvailable(progress.model)) return { model: progress.model, voice: progress.voice };
    const stored = storedVoice();
    if (stored && modelAvailable(stored.model)) return stored;
    return speakVoice();
  }
  function currentVoice() {
    if (!reader.voice || !modelAvailable(reader.voice.model)) reader.voice = pickVoice(reader.doc?.progress) || reader.voice;
    return reader.voice;
  }
  function voiceName(choice) {
    const voice = choice?.voice || '';
    if (voice.startsWith('voice:')) {
      const id = voice.slice(6);
      return (state.namedVoices || []).find((item) => item.id === id)?.name || id;
    }
    return voice || 'Voice';
  }
  function updateVoiceChip() {
    const choice = reader.voice || speakVoice();
    byId('rd-voice-label').textContent = voiceName(choice);
    byId('rd-voice').setAttribute('aria-label', `Reading voice: ${voiceName(choice)}. Change voice`);
  }

  async function fillVoiceOptions(model, selected) {
    const select = byId('rd-voice-select');
    select.innerHTML = '<option value="">Loading voices…</option>';
    select.disabled = true;
    let voices = [];
    let named = [];
    if (model === byId('tts-model')?.value) {
      voices = state.ttsVoices || [];
      named = state.namedVoices || [];
    } else {
      voices = await fetchVoices(model);
      try { named = (await api(`/api/voices/identities?model=${encodeURIComponent(model)}`)).voices || []; } catch { }
    }
    if (byId('rd-voice-model').value !== model) return;
    const namedOptions = named.filter((voice) => voice.available !== false)
      .map((voice) => `<option value="voice:${esc(voice.id)}">${esc(voice.name)}</option>`).join('');
    const providerOptions = voices.map((voice) => {
      const id = voice.id || voice.name || voice;
      return `<option value="${esc(id)}">${esc(id)}</option>`;
    }).join('');
    select.innerHTML = (namedOptions ? `<optgroup label="Named voices">${namedOptions}</optgroup>` : '')
      + (providerOptions ? `<optgroup label="Provider voices">${providerOptions}</optgroup>` : '')
      || '<option value="">No voices for this model</option>';
    select.disabled = false;
    if (selected && [...select.options].some((option) => option.value === selected)) select.value = selected;
  }

  function openVoiceDialog() {
    const models = availableModels();
    const choice = currentVoice() || {};
    const modelSelect = byId('rd-voice-model');
    modelSelect.innerHTML = models.length
      ? models.map((model) => `<option value="${esc(model.id)}">${esc(formatModelName(model))}</option>`).join('')
      : '<option value="">No speech models available</option>';
    if (choice.model && models.some((model) => model.id === choice.model)) modelSelect.value = choice.model;
    fillVoiceOptions(modelSelect.value, choice.voice).catch(() => {});
    byId('rd-voice-dialog').showModal();
  }

  function applyVoice(event) {
    event.preventDefault();
    const model = byId('rd-voice-model').value;
    const voice = byId('rd-voice-select').value;
    if (!model || !voice) {
      showToast('Choose a model and a voice', 'error');
      return;
    }
    const changed = model !== reader.voice?.model || voice !== reader.voice?.voice;
    reader.voice = { model, voice };
    writeStorage(VOICE_KEY, JSON.stringify(reader.voice));
    updateVoiceChip();
    byId('rd-voice-dialog').close();
    if (!changed) return;
    // Rendered audio belongs to the old voice: restart the current paragraph (or play from it later).
    reader.resumeOffsetMs = 0;
    if (reader.playing) startAt(reader.cursor);
    else {
      reader.token += 1;
      haltAudio();
    }
  }

  function speedLabel(speed) { return `${Number.isInteger(speed) ? speed.toFixed(1) : speed}×`; }
  function applySpeed() {
    audioEls().forEach((el) => {
      el.defaultPlaybackRate = reader.speed;
      el.playbackRate = reader.speed;
    });
    const button = byId('rd-speed');
    button.textContent = speedLabel(reader.speed);
    button.setAttribute('aria-label', `Playback speed ${speedLabel(reader.speed)}. Change speed`);
    updatePositionState();
  }
  function cycleSpeed() {
    const index = SPEEDS.indexOf(reader.speed);
    reader.speed = SPEEDS[(index + 1) % SPEEDS.length];
    writeStorage(SPEED_KEY, String(reader.speed));
    applySpeed();
  }

  /* ── Playback engine: two <audio> elements, one playing and one preloading the next part ── */
  function audioEls() { return [byId('rd-audio-a'), byId('rd-audio-b')]; }
  function otherEl(el) {
    const [a, b] = audioEls();
    return el === a ? b : a;
  }
  function partsUrl(fromBlock, voice) {
    const qs = new URLSearchParams({ from_block: String(fromBlock), model: voice.model, voice: voice.voice });
    return docUrl(reader.doc.id, `/parts?${qs}`);
  }

  function silentWav() {
    if (reader.silentUrl) return reader.silentUrl;
    const samples = 800;   // 0.1 s of 8 kHz 8-bit silence
    const bytes = new Uint8Array(44 + samples);
    const view = new DataView(bytes.buffer);
    const ascii = (offset, text) => [...text].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
    ascii(0, 'RIFF'); view.setUint32(4, 36 + samples, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 8000, true); view.setUint32(28, 8000, true); view.setUint16(32, 1, true); view.setUint16(34, 8, true);
    ascii(36, 'data'); view.setUint32(40, samples, true);
    bytes.fill(128, 44);
    let binary = '';
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    reader.silentUrl = `data:audio/wav;base64,${btoa(binary)}`;
    return reader.silentUrl;
  }

  // iOS only lets an element play later (after network waits) if it was first played inside a tap.
  function unlockAudio() {
    if (reader.unlocked) return;
    reader.unlocked = true;
    const url = silentWav();
    audioEls().forEach((el) => {
      if (el.getAttribute('src')) return;
      el.src = url;
      el.play()?.then(() => { if (el.src === url) el.pause(); }).catch(() => {});
    });
  }

  function seekEl(el, seconds) {
    const to = Math.max(0, seconds || 0);
    try { el.currentTime = to; } catch { }
    if (el.readyState === 0 && to > 0) el.addEventListener('loadedmetadata', () => { try { el.currentTime = to; } catch { } }, { once: true });
  }

  // Where block `index` starts inside the given parts: the part beginning with it, or a sentence mark.
  function locate(index, parts = reader.parts) {
    const exact = parts.find((part) => part.first_block === index);
    if (exact) return { part: exact, offsetMs: 0 };
    const part = parts.find((item) => item.first_block <= index && index <= item.last_block);
    const mark = (part?.sentences || []).find((item) => item.block === index);
    return mark ? { part, offsetMs: mark.start_ms || 0 } : null;
  }

  async function requestRender(fromBlock, voice) {
    const data = await readerApi(docUrl(reader.doc.id, '/render'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ from_block: fromBlock, model: voice.model, voice: voice.voice }),
    });
    reader.parts = Array.isArray(data?.parts) ? data.parts : [];
  }

  async function refreshParts(fromBlock, voice) {
    const data = await readerApi(partsUrl(fromBlock, voice));
    reader.parts = Array.isArray(data?.parts) ? data.parts : [];
  }

  function haltAudio() {
    audioEls().forEach((el) => el.pause());
    reader.active = null;
    reader.preloaded = null;
    reader.lookahead += 1;
    stopTicker();
  }

  async function startAt(index, offsetMs = 0, { keepFollow = false } = {}) {
    const doc = reader.doc;
    if (!doc?.blocks.length) return;
    const voice = currentVoice();
    updateVoiceChip();
    if (!voice?.model || !voice.voice) {
      showToast(availableModels().length ? 'Choose a voice for the Reader first' : 'No speech model is ready yet. Check the Models tab.', 'error');
      setPlaying(false);
      if (availableModels().length) openVoiceDialog();
      return;
    }
    index = clamp(index, 0, doc.blocks.length - 1);
    const token = ++reader.token;
    haltAudio();
    if (!keepFollow) reader.follow = true;
    setCursor(index, -1, { scroll: !keepFollow });
    setPlaying(true);
    setWaiting(true);
    setStatus('');
    try {
      await requestRender(index, voice);
      let lastRender = Date.now();
      for (;;) {
        if (token !== reader.token) return;
        const found = locate(index);
        if (found?.part.status === 'error') throw new Error(found.part.error || 'The audio for this paragraph could not be rendered');
        if (found?.part.status === 'ready' && found.part.url) {
          await beginPart(found.part, found.offsetMs + offsetMs, token);
          return;
        }
        await sleep(POLL_MS);
        if (token !== reader.token) return;
        await refreshParts(index, voice);
        // The single server job can be replaced (another tab, another voice); ask again if our window vanished.
        if (!locate(index) && Date.now() - lastRender > RERENDER_AFTER_MS) {
          await requestRender(index, voice);
          lastRender = Date.now();
        }
      }
    } catch (error) {
      if (token === reader.token) fail(error.message);
    }
  }

  async function beginPart(part, offsetMs, token, el = null) {
    const target = el || (reader.active ? otherEl(reader.active.el) : audioEls()[0]);
    const previous = reader.active?.el;
    if (target.dataset.part !== part.id || !target.getAttribute('src') || target.src.startsWith('data:')) {
      target.src = part.url;
      target.dataset.part = part.id;
      if (offsetMs > 0) seekEl(target, offsetMs / 1000);
    } else if (target !== reader.preloaded?.el || offsetMs > 0) {
      seekEl(target, offsetMs / 1000);
    }
    target.defaultPlaybackRate = reader.speed;
    target.playbackRate = reader.speed;
    reader.active = { el: target, part };
    reader.preloaded = null;
    if (previous && previous !== target) previous.pause();
    try {
      await target.play();
    } catch (error) {
      if (token !== reader.token || reader.active?.el !== target) return;
      if (error.name === 'NotAllowedError') {
        setWaiting(false);
        setPlaying(false);
        setStatus('Tap play to start listening');
        return;
      }
      if (error.name === 'AbortError') return;
      throw error;
    }
    if (token !== reader.token || reader.active?.el !== target) return;
    onPartStarted(part, token);
  }

  function onPartStarted(part, token) {
    setWaiting(false);
    setPlaying(true);
    if (reader.recovering && reader.recovering.id !== part.id) reader.recovering = null;
    installMediaSession();
    updateMediaMetadata();
    updatePositionState();
    syncFromAudio();
    saveProgress();
    // Re-point the server job at the part now playing so it keeps rendering a few parts ahead.
    const voice = currentVoice();
    requestRender(part.first_block, voice).catch(() => {}).finally(() => {
      if (token === reader.token) watchNext();
    });
  }

  // Poll (gently) until the part after the active one is ready, then load it into the idle element.
  async function watchNext() {
    const loop = ++reader.lookahead;
    const active = reader.active;
    const voice = currentVoice();
    if (!active || !voice || !reader.doc) return;
    const nextBlock = active.part.last_block + 1;
    if (nextBlock >= reader.doc.blocks.length) return;
    while (loop === reader.lookahead && reader.active === active) {
      const next = reader.parts.find((part) => part.first_block === nextBlock);
      if (next?.status === 'error') return;
      if (next?.status === 'ready' && next.url) {
        preload(next);
        return;
      }
      await sleep(LOOKAHEAD_POLL_MS);
      if (loop !== reader.lookahead || reader.active !== active) return;
      if (!reader.playing) continue;
      try { await refreshParts(active.part.first_block, voice); } catch { }
    }
  }

  function preload(part) {
    const active = reader.active;
    if (!active) return;
    const el = otherEl(active.el);
    if (el.dataset.part !== part.id || !el.getAttribute('src')) {
      el.src = part.url;
      el.dataset.part = part.id;
    } else {
      seekEl(el, 0);
    }
    el.preload = 'auto';
    el.defaultPlaybackRate = reader.speed;
    reader.preloaded = { el, part };
  }

  function handleEnded(event) {
    const active = reader.active;
    if (!active || event.target !== active.el || !reader.playing) return;
    const nextBlock = active.part.last_block + 1;
    if (nextBlock >= reader.doc.blocks.length) {
      finishDocument();
      return;
    }
    const preloaded = reader.preloaded;
    if (preloaded && preloaded.part.first_block === nextBlock) {
      beginPart(preloaded.part, 0, reader.token, preloaded.el).catch((error) => fail(error.message));
      return;
    }
    const next = reader.parts.find((part) => part.first_block === nextBlock);
    if (next?.status === 'error') {
      fail(next.error || 'The next part could not be rendered');
      return;
    }
    startAt(nextBlock, 0, { keepFollow: true });
  }

  // A failed load on the active element is most often an expired signed URL: refetch parts and retry once.
  async function handleAudioError(event) {
    const el = event.target;
    if (!el.getAttribute('src') || el.src.startsWith('data:')) return;
    if (reader.preloaded?.el === el) {
      reader.preloaded = null;
      delete el.dataset.part;
      return;
    }
    const active = reader.active;
    if (!active || active.el !== el) return;
    if (reader.recovering?.id === active.part.id && Date.now() - reader.recovering.at < 30000) {
      fail('The audio could not be loaded');
      return;
    }
    reader.recovering = { id: active.part.id, at: Date.now() };
    const token = reader.token;
    const resumeAt = (el.currentTime || 0) * 1000;
    const wasPlaying = reader.playing;
    try {
      const voice = currentVoice();
      await refreshParts(active.part.first_block, voice);
      if (token !== reader.token) return;
      const fresh = reader.parts.find((part) => part.first_block === active.part.first_block && part.status === 'ready' && part.url);
      if (!fresh) {
        if (wasPlaying) startAt(reader.cursor, 0, { keepFollow: true });
        else haltAudio();
        return;
      }
      delete el.dataset.part;
      if (wasPlaying) {
        await beginPart(fresh, resumeAt, token, el);
      } else {
        el.src = fresh.url;
        el.dataset.part = fresh.id;
        seekEl(el, resumeAt / 1000);
        reader.active = { el, part: fresh };
      }
    } catch (error) {
      if (token === reader.token) fail(error.message);
    }
  }

  function handlePause(event) {
    const el = event.target;
    if (el !== reader.active?.el || el.ended || reader.waiting || !reader.playing) return;
    // Paused by the system (headphones removed, another app took audio).
    setPlaying(false);
    saveProgress();
  }

  function handlePlay(event) {
    if (event.target === reader.active?.el && !reader.playing) setPlaying(true);
  }

  function finishDocument() {
    const doc = reader.doc;
    reader.token += 1;
    haltAudio();
    setCursor(doc.blocks.length - 1);
    setPlaying(false);
    setWaiting(false);
    setStatus('Finished');
    saveProgress();
    if ('mediaSession' in navigator) navigator.mediaSession.playbackState = 'none';
  }

  function fail(message) {
    reader.token += 1;
    haltAudio();
    setWaiting(false);
    setPlaying(false);
    setStatus(`Couldn't play: ${message}`, true);
    showToast(message, 'error');
  }

  function stopPlayback() {
    reader.token += 1;
    haltAudio();
    setWaiting(false);
    setPlaying(false);
    setStatus('');
    clearMediaSession();
  }

  /* ── Transport actions ── */
  function play() {
    if (!reader.doc?.blocks.length || reader.playing) return;
    unlockAudio();
    setStatus('');
    const active = reader.active;
    if (active?.el.getAttribute('src')) {
      setPlaying(true);
      active.el.play().then(() => {
        installMediaSession();
        updateMediaMetadata();
        if (!reader.preloaded) watchNext();
      }).catch((error) => {
        if (error.name === 'NotAllowedError') setPlaying(false);
        else if (error.name !== 'AbortError') startAt(reader.cursor, 0, { keepFollow: true });
      });
      return;
    }
    const offset = reader.resumeOffsetMs;
    reader.resumeOffsetMs = 0;
    startAt(reader.cursor, offset);
  }

  function pause() {
    if (!reader.playing) return;
    if (reader.waiting || !reader.active) {
      reader.token += 1;
      haltAudio();
    } else {
      reader.active.el.pause();
    }
    setWaiting(false);
    setPlaying(false);
    saveProgress();
  }

  function togglePlay() {
    if (reader.playing) pause();
    else play();
  }

  // Move playback (or the paused position) to the start of a block.
  function jumpTo(index) {
    const doc = reader.doc;
    if (!doc?.blocks.length) return;
    index = clamp(index, 0, doc.blocks.length - 1);
    reader.follow = true;
    reader.resumeOffsetMs = 0;
    const active = reader.active;
    if (active && !reader.waiting) {
      const here = locate(index, [active.part]);
      if (here) {
        seekEl(active.el, here.offsetMs / 1000);
        setCursor(index, -1, { scroll: true });
        updatePositionState();
        if (!reader.playing) saveProgress();
        return;
      }
      if (reader.playing && reader.preloaded?.part.first_block === index) {
        beginPart(reader.preloaded.part, 0, reader.token, reader.preloaded.el).catch((error) => fail(error.message));
        setCursor(index, -1, { scroll: true });
        return;
      }
    }
    if (!reader.playing) {
      reader.token += 1;
      haltAudio();
      setCursor(index, -1, { scroll: true });
      saveProgress();
      return;
    }
    startAt(index);
  }

  function skipBlock(delta) {
    if (!reader.doc?.blocks.length) return;
    const target = clamp(reader.cursor + delta, 0, reader.doc.blocks.length - 1);
    if (target === reader.cursor && delta > 0) return;
    jumpTo(target);
  }

  function seekBy(seconds) {
    const active = reader.active;
    if (!active || reader.waiting) return;
    const el = active.el;
    const to = el.currentTime + seconds;
    if (to < 0) {
      jumpTo(active.part.first_block - 1);
      return;
    }
    const duration = Number.isFinite(el.duration) ? el.duration : (active.part.duration_ms || 0) / 1000;
    if (duration && to >= duration) {
      if (active.part.last_block + 1 < reader.doc.blocks.length) jumpTo(active.part.last_block + 1);
      return;
    }
    seekEl(el, to);
    syncFromAudio();
    updatePositionState();
  }

  function playFrom(index) {
    unlockAudio();
    jumpTo(index);
    if (!reader.playing) play();
  }

  /* ── Transport UI ── */
  function setPlaying(on) {
    reader.playing = on;
    const button = byId('rd-play');
    button.setAttribute('aria-label', on ? 'Pause' : 'Play');
    button.setAttribute('aria-pressed', on ? 'true' : 'false');
    button.querySelector('use').setAttribute('href', on ? '#i-pause' : '#i-play');
    if ('mediaSession' in navigator && reader.active) navigator.mediaSession.playbackState = on ? 'playing' : 'paused';
    clearInterval(reader.saveTimer);
    reader.saveTimer = on ? setInterval(() => saveProgress(), SAVE_EVERY_MS) : null;
    if (on) startTicker();
    else stopTicker();
    updateReturnPill();
  }

  function setWaiting(on) {
    reader.waiting = on;
    clearTimeout(reader.preparingTimer);
    if (on) reader.preparingTimer = setTimeout(() => showPreparing(true), PREPARING_DELAY_MS);
    else showPreparing(false);
  }

  function showPreparing(on) {
    const button = byId('rd-play');
    button.classList.toggle('is-preparing', on);
    button.setAttribute('aria-busy', on ? 'true' : 'false');
    if (on) setStatus('Preparing audio…');
    else if (byId('rd-status').textContent === 'Preparing audio…') setStatus('');
  }

  function setStatus(text, error = false) {
    const status = byId('rd-status');
    status.textContent = text;
    status.classList.toggle('is-error', !!text && error);
  }

  // Track the sentence being spoken; rAF because Safari only fires timeupdate about 4 times a second.
  function startTicker() {
    if (!reader.raf) reader.raf = requestAnimationFrame(tick);
  }
  function stopTicker() {
    if (reader.raf) cancelAnimationFrame(reader.raf);
    reader.raf = 0;
  }
  function tick() {
    reader.raf = 0;
    if (!reader.playing) return;
    if (reader.active && !reader.active.el.paused) syncFromAudio();
    reader.raf = requestAnimationFrame(tick);
  }

  function syncFromAudio() {
    const active = reader.active;
    if (!active || !reader.doc) return;
    const ms = active.el.currentTime * 1000;
    const { part } = active;
    const marks = Array.isArray(part.sentences) ? part.sentences : [];
    let block = part.first_block;
    let sentence = -1;
    if (marks.length) {
      for (const mark of marks) {
        if (mark.start_ms > ms) break;
        block = mark.block;
        sentence = mark.index;
      }
    } else if (part.last_block > part.first_block && active.el.duration > 0) {
      // No timings: estimate the block from its share of the part's characters.
      const blocks = reader.doc.blocks.slice(part.first_block, part.last_block + 1);
      const total = blocks.reduce((sum, item) => sum + (item.text || '').length, 0) || 1;
      let reached = (active.el.currentTime / active.el.duration) * total;
      for (let i = 0; i < blocks.length; i += 1) {
        reached -= (blocks[i].text || '').length;
        if (reached <= 0) {
          block = part.first_block + i;
          break;
        }
      }
    }
    if (block !== reader.cursor || sentence !== reader.sentence) setCursor(block, sentence);
  }

  /* ── Saved position ── */
  function progressPayload() {
    syncFromAudio();
    let offset = 0;
    const active = reader.active;
    if (active && locate(reader.cursor, [active.part])) {
      const marks = active.part.sentences || [];
      const blockStart = locate(reader.cursor, [active.part]).offsetMs;
      const sentenceStart = marks.find((mark) => mark.block === reader.cursor && mark.index === reader.sentence)?.start_ms;
      offset = Math.max(0, Math.round((sentenceStart ?? active.el.currentTime * 1000) - blockStart));
    } else if (!active) {
      offset = reader.resumeOffsetMs;
    }
    const voice = reader.voice;
    return { block_index: reader.cursor, offset_ms: offset, model: voice?.model || null, voice: voice?.voice || null, speed: reader.speed };
  }

  function saveProgress({ keepalive = false } = {}) {
    const doc = reader.doc;
    if (!doc?.blocks.length) return Promise.resolve();
    const body = JSON.stringify(progressPayload());
    if (body === reader.lastSaved) return Promise.resolve();
    reader.lastSaved = body;
    return fetch(docUrl(doc.id, '/progress'), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body,
      keepalive,
    }).then(async (res) => {
      if (!res.ok) return;
      const data = await res.json().catch(() => null);
      const summary = reader.docs.find((item) => item.id === doc.id);
      if (data?.progress && summary) summary.progress = data.progress;
    }).catch(() => { reader.lastSaved = ''; });
  }

  /* ── Media Session (lock screen, headphones) ── */
  function artworkFor(doc) {
    if (reader.artwork.id === doc.id) return reader.artwork.url;
    let url = '';
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      const hue = hueFor(doc.id || doc.title);
      const gradient = ctx.createLinearGradient(0, 0, 512, 512);
      gradient.addColorStop(0, `hsl(${hue}, 52%, 46%)`);
      gradient.addColorStop(1, `hsl(${(hue + 40) % 360}, 58%, 34%)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 512, 512);
      ctx.fillStyle = '#fff';
      ctx.font = '700 200px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(monogram(doc.title), 256, 270);
      url = canvas.toDataURL('image/png');
    } catch { }
    reader.artwork = { id: doc.id, url };
    return url;
  }

  function updateMediaMetadata() {
    if (!('mediaSession' in navigator) || typeof MediaMetadata === 'undefined' || !reader.doc) return;
    const art = artworkFor(reader.doc);
    try {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: reader.doc.title,
        artist: currentChapter()?.title || reader.doc.author || '',
        album: 'Open Speech',
        artwork: art ? [{ src: art, sizes: '512x512', type: 'image/png' }] : [],
      });
    } catch { }
  }

  function updatePositionState() {
    const session = navigator.mediaSession;
    const el = reader.active?.el;
    if (!session?.setPositionState || !el || !(el.duration > 0) || !Number.isFinite(el.duration)) return;
    try {
      session.setPositionState({ duration: el.duration, playbackRate: el.playbackRate || 1, position: Math.min(el.currentTime, el.duration) });
    } catch { }
  }

  const MEDIA_ACTIONS = {
    play: () => play(),
    pause: () => pause(),
    previoustrack: () => skipBlock(-1),
    nexttrack: () => skipBlock(1),
    seekbackward: (details) => seekBy(-(details?.seekOffset || BACK_SECONDS)),
    seekforward: (details) => seekBy(details?.seekOffset || FORWARD_SECONDS),
    seekto: (details) => {
      if (reader.active && Number.isFinite(details?.seekTime)) {
        seekEl(reader.active.el, details.seekTime);
        updatePositionState();
      }
    },
  };

  // Handlers are installed while the Reader owns playback so they don't hijack the Speak tab's audio.
  function installMediaSession() {
    if (!('mediaSession' in navigator)) return;
    Object.entries(MEDIA_ACTIONS).forEach(([action, handler]) => {
      try { navigator.mediaSession.setActionHandler(action, handler); } catch { }
    });
  }

  function clearMediaSession() {
    if (!('mediaSession' in navigator)) return;
    Object.keys(MEDIA_ACTIONS).forEach((action) => {
      try { navigator.mediaSession.setActionHandler(action, null); } catch { }
    });
    try {
      navigator.mediaSession.metadata = null;
      navigator.mediaSession.playbackState = 'none';
    } catch { }
  }

  /* ── Keyboard ── */
  function handleKeydown(event) {
    if (!readingVisible() || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
    if (document.querySelector('#panel-reader dialog[open]')) return;
    const target = event.target;
    if (target.closest?.('input, textarea, select, [contenteditable="true"], [role="tab"]')) return;
    const paragraph = target.closest?.('[data-block]');
    if (paragraph && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      playFrom(Number(paragraph.dataset.block));
      return;
    }
    switch (event.key) {
      case ' ':
        if (target.closest?.('button, a, summary')) return;   // let buttons activate natively
        togglePlay();
        break;
      case 'ArrowLeft': skipBlock(-1); break;
      case 'ArrowRight': skipBlock(1); break;
      case 'j': case 'J': seekBy(-BACK_SECONDS); break;
      case 'l': case 'L': seekBy(FORWARD_SECONDS); break;
      case 'ArrowUp': case 'ArrowDown': case 'PageUp': case 'PageDown': case 'Home': case 'End':
        noteManualScroll(event);
        return;
      default:
        return;
    }
    event.preventDefault();
  }

  /* ── Wiring ── */
  function openFromHash() {
    const match = /^#reader(?:\/(.+))?$/.exec(location.hash);
    if (!match) return;
    if (!readerTabActive()) byId('tab-reader').click();
    else if (!reader.loaded && !reader.loading) loadLibrary();   // app.js opened the tab from the URL before this file was wired
    if (match[1]) {
      let id = match[1];
      try { id = decodeURIComponent(id); } catch { }
      openDocument(id);
    }
  }

  function bindDialogs() {
    document.querySelectorAll('#panel-reader dialog').forEach((dialog) => {
      dialog.addEventListener('click', (event) => {
        if (event.target === dialog) dialog.close();   // tap on the backdrop
        if (event.target.closest('[data-rd-close]')) dialog.close();
      });
    });
    const segs = [...document.querySelectorAll('#rd-add-dialog .rd-seg-btn')];
    segs.forEach((button) => {
      button.addEventListener('click', () => setAddMode(button.dataset.seg));
      button.addEventListener('keydown', (event) => handleTabKeydown(event, segs));
    });
    byId('rd-add-form').addEventListener('submit', submitAdd);
    byId('rd-file').addEventListener('change', (event) => setAddFile(event.target.files?.[0] || null));
    const zone = byId('rd-dropzone');
    zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      zone.classList.remove('dragover');
      const file = event.dataTransfer?.files?.[0];
      if (file) setAddFile(file);
    });
    byId('rd-voice-model').addEventListener('change', (event) => fillVoiceOptions(event.target.value, '').catch(() => {}));
    byId('rd-voice-form').addEventListener('submit', applyVoice);
  }

  function bindChapters() {
    const handler = (event) => {
      const button = event.target.closest('[data-chapter]');
      const chapter = button && reader.doc?.chapters[Number(button.dataset.chapter)];
      if (!chapter) return;
      byId('rd-chapters-dialog').close();
      playFrom(chapter.first_block);
    };
    byId('rd-chapter-list').addEventListener('click', handler);
    byId('rd-chapter-list-sheet').addEventListener('click', handler);
    document.querySelectorAll('[data-rd-chapters]').forEach((button) => button.addEventListener('click', () => {
      byId('rd-chapters-dialog').showModal();
      byId('rd-chapter-list-sheet').querySelector('.is-current')?.scrollIntoView({ block: 'center' });
    }));
  }

  function bindTransport() {
    byId('rd-play').addEventListener('click', () => {
      unlockAudio();
      togglePlay();
    });
    byId('rd-prev').addEventListener('click', () => skipBlock(-1));
    byId('rd-next').addEventListener('click', () => skipBlock(1));
    byId('rd-back15').addEventListener('click', () => seekBy(-BACK_SECONDS));
    byId('rd-fwd30').addEventListener('click', () => seekBy(FORWARD_SECONDS));
    byId('rd-speed').addEventListener('click', cycleSpeed);
    byId('rd-voice').addEventListener('click', openVoiceDialog);
    byId('rd-return').addEventListener('click', returnToPlaying);
    audioEls().forEach((el) => {
      el.addEventListener('ended', handleEnded);
      el.addEventListener('error', handleAudioError);
      el.addEventListener('pause', handlePause);
      el.addEventListener('play', handlePlay);
      ['loadedmetadata', 'durationchange', 'ratechange', 'seeked'].forEach((type) => el.addEventListener(type, () => {
        if (el === reader.active?.el) updatePositionState();
      }));
    });
  }

  function initReader() {
    if (!byId('panel-reader')) return;
    const savedSpeed = Number(readStorage(SPEED_KEY, '1'));
    reader.speed = SPEEDS.includes(savedSpeed) ? savedSpeed : 1;
    applySpeed();

    byId('rd-library').addEventListener('click', handleLibraryClick);
    byId('rd-search').addEventListener('input', renderLibrary);
    byId('rd-add').addEventListener('click', openAddDialog);
    byId('rd-empty-action').addEventListener('click', (event) => {
      if (event.currentTarget.dataset.action === 'retry') loadLibrary();
      else openAddDialog();
    });
    document.querySelectorAll('[data-rd-back]').forEach((button) => button.addEventListener('click', showLibrary));
    byId('rd-text').addEventListener('click', (event) => {
      const paragraph = event.target.closest('[data-block]');
      if (!paragraph || String(window.getSelection?.() || '').length) return;
      playFrom(Number(paragraph.dataset.block));
    });
    bindTransport();
    bindChapters();
    bindDialogs();

    // Runs after app.js's tab handler, so the panels are already switched.
    document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
      syncBodyClass();
      if (tab.dataset.tab === 'reader') {
        if (!reader.loaded && !reader.loading) loadLibrary();
        if (readingVisible() && reader.doc) setHash(`#reader/${encodeURIComponent(reader.doc.id)}`);
      } else if (location.hash.startsWith('#reader')) {
        setHash('');
      }
    }));
    document.addEventListener('keydown', handleKeydown);
    window.addEventListener('wheel', noteManualScroll, { passive: true });
    window.addEventListener('touchmove', noteManualScroll, { passive: true });
    window.addEventListener('hashchange', openFromHash);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) saveProgress({ keepalive: true });
    });
    window.addEventListener('pagehide', () => saveProgress({ keepalive: true }));
    openFromHash();
  }

  document.addEventListener('DOMContentLoaded', initReader);
})();
