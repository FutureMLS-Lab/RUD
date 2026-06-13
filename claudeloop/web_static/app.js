/**
 * claudeloop web client.
 *
 * One task surface: an agent tmux pane plus a read-only Markdown viewer
 * that can switch between scanned task/worktree Markdown files.
 *
 * Project-scoped NOTES.md is reached via the sidebar's Notes button.
 *
 * The client talks to /api/projects, /api/tasks, /api/tmux/*,
 * /api/tasks/<slug>/(interview|claude)/*, and template GET/PUT for
 * PLAN.md.
 */

const FILES = {
  plan: 'PLAN.md',
};

// "interview" is the embedded read-only Markdown viewer on the agent pane.
const MARKDOWN_PANELS = ['interview'];

// Tab labels are computed per task so the agent pane name matches the
// task's agent setting (Claude / Codex).
const TABS = [
  { id: 'claude', label: 'Claude', getLabel: (meta) => agentLabel(meta?.agent) },
  { id: 'changes', label: 'Changes' },
];
const DEFAULT_TAB = TABS[0].id;

const AGENT_LABELS = { claude: 'Claude', codex: 'Codex' };
function agentLabel(name) { return AGENT_LABELS[(name || '').toLowerCase()] || 'Claude'; }
function normalizeAgent(name) { return AGENT_LABELS[(name || '').toLowerCase()] ? name.toLowerCase() : 'claude'; }

const STATE = {
  slug: null,
  projectId: null,
  projects: [],
  skillsPath: '',
  skillsOptions: [],
  tasks: [],
  currentMeta: null,
  worktreeStatuses: [],
  taskRoot: '',
  planPath: '',
  launchRoot: '',
  launchRootChildren: [],
  paneTimer: null,
  activePanel: TABS[0].id,
  previewCache: {},
  previewDebounce: {},
  sidebarOpen: false,
  notesDirty: false,
  notesSaving: false,
  taskFilter: '',
  pollInFlight: {
    capture: false,
    templates: false,
    sessions: false,
  },
  // Per-task unsent text in the terminal input box. Keep this client-side
  // only: drafts can contain arbitrary user text and shouldn't be written
  // into task metadata or markdown files.
  paneDrafts: {},
  // Embedded read-only markdown viewer on the Claude tab. The picker
  // lets the user flip between any top-level *.md file in the task root;
  // PLAN.md is the default.
  interviewMdFile: FILES.plan,
  interviewMdFiles: [],
  interviewMdContents: {},
  // Changes tab: cached diff payload + which file is selected.
  changesData: null,
  changesSelected: '',
  changesLoading: false,
};

let PROJECT_DRAG_ID = '';
let PROJECT_JUST_DRAGGED = false;
let TASK_DRAG_SLUG = '';
let TASK_JUST_DRAGGED = false;

function withProjectQuery(path) {
  if (!STATE.projectId) return path;
  if (path.startsWith('/api/projects')) return path;
  if (!path.startsWith('/api/project') && !path.startsWith('/api/tasks') && !path.startsWith('/api/kernel')) return path;
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}project=${encodeURIComponent(STATE.projectId)}`;
}

async function apiNoProject(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body !== undefined && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(path, { ...opts, headers });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { error: text }; }
  if (!res.ok) throw makeApiError(res, data);
  return data;
}

async function api(path, opts = {}) {
  const url = withProjectQuery(path);
  const headers = { ...(opts.headers || {}) };
  if (opts.body !== undefined && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(url, { ...opts, headers });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { error: text }; }
  if (!res.ok) throw makeApiError(res, data);
  return data;
}

function makeApiError(res, data) {
  const err = new Error((data && data.error) || res.statusText || `HTTP ${res.status}`);
  err.status = res.status;
  err.body = data;
  return err;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTransientApiError(err) {
  // These usually mean the local web server is restarting, the proxy timed
  // out, or a long-running tmux/git request temporarily blocked the route.
  // Retrying is safe for GETs and avoids surfacing noisy "Bad Gateway" text
  // in the task terminal.
  return [502, 503, 504].includes(Number(err && err.status));
}

async function apiWithRetry(path, opts = {}, retryOpts = {}) {
  const attempts = retryOpts.attempts ?? 3;
  const delayMs = retryOpts.delayMs ?? 250;
  let lastErr;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await api(path, opts);
    } catch (err) {
      lastErr = err;
      if (!isTransientApiError(err) || i === attempts - 1) break;
      await sleep(delayMs * (i + 1));
    }
  }
  throw lastErr;
}

function $(sel) { return document.querySelector(sel); }

// ===== Tabs =====

function showPanel(id) {
  const hasTabs = !!document.getElementById('main-tabs');
  if (hasTabs) {
    document.querySelectorAll('.tab').forEach((t) => {
      t.classList.toggle('active', t.dataset.tab === id);
    });
  }
  document.querySelectorAll('.tab-panel').forEach((p) => {
    const on = p.dataset.panel === id;
    p.classList.toggle('active', on);
    p.hidden = !on;
  });
  STATE.activePanel = id;
  if (id === 'claude') {
    // The Claude tab embeds a read-only viewer for any scanned *.md
    // file in the task root - defaults to PLAN.md.
    updateMarkdownPreview('interview');
    // Show the most-recent tmux output (terminal-style) the moment the
    // user lands on this tab. The element was previously hidden so its
    // scrollTop is stale (0 = top of buffer) - we want the bottom.
    scrollTmuxOutputToBottom();
    deferIdle(refreshInterviewPreview);
    deferIdle(refreshTaskTemplates);
    deferIdle(refreshClaudeSessions);
  }
  if (id === 'changes') {
    refreshChangesView();
  }
}

function deferIdle(fn) {
  if (typeof requestIdleCallback === 'function') {
    requestIdleCallback(() => { try { fn(); } catch (_) {} }, { timeout: 200 });
  } else {
    setTimeout(() => { try { fn(); } catch (_) {} }, 0);
  }
}

function buildTabs(meta) {
  const nav = $('#main-tabs');
  if (!nav) return;
  const isKernel = !!(meta && meta.kind === 'kernel');
  nav.hidden = isKernel;
  nav.innerHTML = '';
  if (isKernel) return;
  for (const t of TABS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tab' + (t.id === DEFAULT_TAB ? ' active' : '');
    b.dataset.tab = t.id;
    b.textContent = typeof t.getLabel === 'function' ? t.getLabel(meta) : t.label;
    b.addEventListener('click', () => showPanel(t.id));
    nav.appendChild(b);
  }
}

// ===== Projects =====

async function loadProjectsList() {
  const d = await apiNoProject('/api/projects');
  STATE.projects = d.projects || [];
  STATE.launchRoot = String(d.launchRoot || '').trim();
  STATE.launchRootChildren = Array.isArray(d.launchRootChildren) ? d.launchRootChildren : [];
  const cur = String(d.currentProjectId || d.defaultProjectId || '').trim();
  if (cur && STATE.projects.some((p) => p.id === cur)) {
    STATE.projectId = cur;
  } else {
    STATE.projectId = null;
  }
  renderProjectToggleBar();
}

function renderProjectToggleBar() {
  const scroll = document.getElementById('project-toggle-scroll');
  if (!scroll) return;
  scroll.innerHTML = '';
  const list = STATE.projects || [];
  if (!list.length) {
    const em = document.createElement('span');
    em.className = 'project-bar__empty-msg';
    em.textContent = 'No repos yet — use + Add repo to register a project root.';
    scroll.appendChild(em);
    return;
  }
  list.forEach((p) => {
    const item = document.createElement('div');
    item.className = 'project-toggle' + (p.id === STATE.projectId ? ' is-active' : '');
    item.dataset.projectId = p.id;
    item.title = p.path || p.name || p.id;
    item.draggable = true;
    item.addEventListener('dragstart', (ev) => {
      PROJECT_DRAG_ID = p.id;
      PROJECT_JUST_DRAGGED = true;
      item.classList.add('is-dragging');
      ev.dataTransfer.effectAllowed = 'move';
      ev.dataTransfer.setData('text/plain', p.id);
    });
    item.addEventListener('dragover', (ev) => {
      if (!PROJECT_DRAG_ID || PROJECT_DRAG_ID === p.id) return;
      ev.preventDefault();
      ev.dataTransfer.dropEffect = 'move';
      const rect = item.getBoundingClientRect();
      const after = ev.clientX > rect.left + (rect.width / 2);
      clearProjectDropMarkers(scroll);
      item.classList.toggle('is-drop-before', !after);
      item.classList.toggle('is-drop-after', after);
    });
    item.addEventListener('drop', async (ev) => {
      if (!PROJECT_DRAG_ID || PROJECT_DRAG_ID === p.id) return;
      ev.preventDefault();
      const dragId = ev.dataTransfer.getData('text/plain') || PROJECT_DRAG_ID;
      const after = item.classList.contains('is-drop-after');
      clearProjectDropMarkers(scroll);
      await reorderProjectsByDrag(dragId, p.id, after);
    });
    item.addEventListener('dragend', () => {
      PROJECT_DRAG_ID = '';
      item.classList.remove('is-dragging');
      clearProjectDropMarkers(scroll);
      setTimeout(() => { PROJECT_JUST_DRAGGED = false; }, 0);
    });
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'project-toggle__main';
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', p.id === STATE.projectId ? 'true' : 'false');
    const label = document.createElement('span');
    label.className = 'project-toggle__label';
    label.textContent = p.name || p.id;
    btn.appendChild(label);
    btn.addEventListener('click', () => {
      if (PROJECT_JUST_DRAGGED) return;
      if (p.id !== STATE.projectId) switchProject(p.id);
    });
    item.appendChild(btn);
    const controls = document.createElement('span');
    controls.className = 'project-toggle__controls';
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'project-toggle__rm';
    rm.setAttribute('aria-label', `Remove ${p.name || p.id} from list`);
    rm.textContent = 'x';
    rm.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      removeProject(p.id);
    });
    controls.appendChild(rm);
    item.appendChild(controls);
    scroll.appendChild(item);
  });
  requestAnimationFrame(() => {
    const active = scroll.querySelector('.project-toggle.is-active');
    if (active) active.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
  });
}

function clearProjectDropMarkers(root = document) {
  root.querySelectorAll('.project-toggle.is-drop-before, .project-toggle.is-drop-after').forEach((el) => {
    el.classList.remove('is-drop-before', 'is-drop-after');
  });
}

async function reorderProjectsByDrag(dragId, targetId, afterTarget) {
  const activeId = STATE.projectId;
  const ids = (STATE.projects || []).map((p) => p.id);
  const from = ids.indexOf(dragId);
  const target = ids.indexOf(targetId);
  if (from < 0 || target < 0 || dragId === targetId) return;
  ids.splice(from, 1);
  const targetAfterRemoval = ids.indexOf(targetId);
  ids.splice(targetAfterRemoval + (afterTarget ? 1 : 0), 0, dragId);
  if (ids.every((id, idx) => id === (STATE.projects[idx] && STATE.projects[idx].id))) return;
  const byId = new Map((STATE.projects || []).map((p) => [p.id, p]));
  STATE.projects = ids.map((id) => byId.get(id)).filter(Boolean);
  renderProjectToggleBar();
  try {
    const d = await apiNoProject('/api/projects/reorder', {
      method: 'POST',
      body: JSON.stringify({ ids }),
    });
    STATE.projects = d.projects || STATE.projects || [];
    if (activeId && STATE.projects.some((p) => p.id === activeId)) {
      STATE.projectId = activeId;
    }
    renderProjectToggleBar();
  } catch (e) {
    alert(e.message);
    await loadProjectsList();
  }
}

async function switchProject(id) {
  if (!id || id === STATE.projectId) return;
  await apiNoProject(`/api/projects/${encodeURIComponent(id)}/activate`, { method: 'POST', body: '{}' });
  STATE.projectId = id;
  clearTaskSelection();
  await loadProjectsList();
  await loadProject();
  await loadTasks();
  await loadTmuxSessions();
}

async function removeProject(id) {
  if (!confirm('Remove this project from the web UI list? Task files on disk are not deleted.')) return;
  try {
    await apiNoProject(`/api/projects/${encodeURIComponent(id)}`, { method: 'DELETE' });
    clearTaskSelection();
    await loadProjectsList();
    await loadProject();
    await loadTasks();
    await loadTmuxSessions();
  } catch (e) {
    alert(e.message);
  }
}

async function openAddProjectModal() {
  const modal = $('#add-project-modal');
  if (!modal) return;
  modal.hidden = false;
  $('#add-project-status').textContent = '';
  $('#new-project-path').value = '';
  try {
    await loadProjectsList();
  } catch (e) {
    $('#add-project-status').textContent = e.message;
  }
  renderAddProjectChips();
  requestAnimationFrame(() => $('#new-project-path').focus());
}

function renderAddProjectChips() {
  const wrap = document.getElementById('add-project-launch-wrap');
  const host = document.getElementById('add-project-chips');
  if (!wrap || !host) return;
  host.innerHTML = '';
  const kids = STATE.launchRootChildren || [];
  const root = (STATE.launchRoot || '').trim();
  if (!kids.length || !root) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const elRoot = document.getElementById('add-project-launch-root');
  if (elRoot) elRoot.textContent = root;
  for (const k of kids) {
    const name = k && k.name != null ? String(k.name) : '';
    const path = k && k.path != null ? String(k.path) : '';
    if (!name || !path) continue;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'add-project-chip';
    b.textContent = name;
    b.title = path;
    b.addEventListener('click', () => {
      $('#new-project-path').value = path;
      $('#add-project-status').textContent = '';
      const inp = $('#new-project-path');
      inp.focus();
      inp.select();
    });
    host.appendChild(b);
  }
}

function closeAddProjectModal() {
  const m = $('#add-project-modal');
  if (m) m.hidden = true;
}

async function submitAddProject() {
  const path = $('#new-project-path').value.trim();
  const status = $('#add-project-status');
  const btn = $('#btn-add-project-save');
  if (!path) {
    status.textContent = 'Enter a directory path.';
    return;
  }
  btn.disabled = true;
  status.textContent = 'Adding…';
  try {
    const created = await apiNoProject('/api/projects', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    if (created.id) STATE.projectId = created.id;
    else if (created.defaultProjectId) STATE.projectId = created.defaultProjectId;
    closeAddProjectModal();
    await loadProjectsList();
    await loadProject();
    await loadTasks();
    await loadTmuxSessions();
  } catch (e) {
    status.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

async function loadProject() {
  if (!STATE.projectId) {
    $('#hdr-project').textContent = '(select a project above)';
    STATE.skillsPath = '';
    STATE.skillsOptions = [];
    renderSkillsPicker();
    renderTaskSkillsPicker();
    return;
  }
  const d = await api('/api/project');
  const meta = (STATE.projects || []).find((x) => x.id === STATE.projectId);
  const pathLine = d.projectRoot || '';
  $('#hdr-project').textContent = meta ? `${meta.name} — ${pathLine}` : pathLine;
  STATE.skillsPath = d.skillsPath || '';
  STATE.skillsOptions = Array.isArray(d.skillsOptions) ? d.skillsOptions : [];
  renderSkillsPicker();
  renderTaskSkillsPicker(STATE.currentMeta || {});
}

function renderSkillsPicker() {
  const sel = document.getElementById('new-skills');
  if (!sel) return;
  const current = sel.value || STATE.skillsPath || '';
  sel.innerHTML = '';
  const options = STATE.skillsOptions.length
    ? STATE.skillsOptions
    : (STATE.skillsPath ? [{ label: STATE.skillsPath, path: STATE.skillsPath }] : []);
  for (const opt of options) {
    const path = String(opt.path || '').trim();
    if (!path) continue;
    const option = document.createElement('option');
    option.value = path;
    option.textContent = opt.label ? `${opt.label}` : path;
    option.title = path;
    sel.appendChild(option);
  }
  if ([...sel.options].some((opt) => opt.value === current)) {
    sel.value = current;
  } else if ([...sel.options].some((opt) => opt.value === STATE.skillsPath)) {
    sel.value = STATE.skillsPath;
  }
}

async function loadTmuxSessions() {
  const ul = $('#tmux-sessions');
  ul.innerHTML = '';
  try {
    if (!STATE.projectId) {
      ul.innerHTML = '<li class="task-list__empty">Select a project to list claudeloop tmux sessions for that root.</li>';
      return;
    }
    const q = `?project=${encodeURIComponent(STATE.projectId)}`;
    const d = await apiNoProject(`/api/tmux/sessions${q}`);
    const list = d.sessions || [];
    if (!list.length) {
      ul.innerHTML = '<li class="task-list__empty">No claudeloop tmux sessions for this project (or tmux not installed).</li>';
      return;
    }
    for (const s of list) {
      const li = document.createElement('li');
      if (s.attached === '1') li.classList.add('attached');
      li.innerHTML = `<strong>${escapeHtml(s.name)}</strong>${s.attached === '1' ? ' <span class="status-ok">attached</span>' : ''}`;
      ul.appendChild(li);
    }
  } catch (e) {
    ul.innerHTML = `<li class="status-bad">${escapeHtml(e.message)}</li>`;
  }
}

// ===== Markdown rendering =====

const HTML_ESCAPE_MAP = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, (ch) => HTML_ESCAPE_MAP[ch]);
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}

function renderMarkdown(md) {
  const lines = (md || '').replace(/\r\n/g, '\n').split('\n');
  const out = [];
  let paragraph = [];
  let listType = null;
  let codeLines = null;

  function flushParagraph() {
    if (!paragraph.length) return;
    out.push(`<p>${renderInlineMarkdown(paragraph.join(' '))}</p>`);
    paragraph = [];
  }
  function flushList() {
    if (!listType) return;
    out.push(`</${listType}>`);
    listType = null;
  }
  function isTableSeparator(line) {
    return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
  }
  function parseTableRow(line) {
    return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
  }
  function renderTable(headers, rows) {
    const head = headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join('');
    const body = rows
      .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join('')}</tr>`)
      .join('');
    return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (codeLines) {
      if (/^```/.test(line.trim())) {
        out.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
        codeLines = null;
      } else {
        codeLines.push(line);
      }
      continue;
    }
    if (/^```/.test(line.trim())) {
      flushParagraph();
      flushList();
      codeLines = [];
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      flushParagraph();
      flushList();
      const headers = parseTableRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(parseTableRow(lines[i]));
        i += 1;
      }
      i -= 1;
      out.push(renderTable(headers, rows));
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      out.push(`<h${heading[1].length}>${renderInlineMarkdown(heading[2])}</h${heading[1].length}>`);
      continue;
    }
    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    if (unordered) {
      flushParagraph();
      if (listType !== 'ul') { flushList(); listType = 'ul'; out.push('<ul>'); }
      out.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      if (listType !== 'ol') { flushList(); listType = 'ol'; out.push('<ol>'); }
      out.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
      continue;
    }
    const quote = line.match(/^>\s?(.+)$/);
    if (quote) {
      flushParagraph();
      flushList();
      out.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }
    paragraph.push(line.trim());
  }
  if (codeLines) out.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  flushParagraph();
  flushList();
  return out.join('\n') || '<p class="empty-preview">Nothing to preview yet.</p>';
}

function updateMarkdownPreview(which, force = false) {
  const editor = $(`#editor-${which}`);
  const preview = $(`#preview-${which}`);
  if (!editor || !preview) return;
  const text = editor.value || '';
  if (!force && STATE.previewCache[which] === text) return;
  STATE.previewCache[which] = text;
  preview.innerHTML = renderMarkdown(text);
}

function updateActiveMarkdownPreview() {
  const which = STATE.activePanel;
  if (MARKDOWN_PANELS.includes(which)) updateMarkdownPreview(which);
}

function invalidatePreviewCache() {
  STATE.previewCache = {};
}

function initMarkdownPreviews() {
  MARKDOWN_PANELS.forEach((which) => {
    const editor = $(`#editor-${which}`);
    if (!editor) return;
    editor.addEventListener('input', () => {
      if (STATE.previewDebounce[which]) cancelAnimationFrame(STATE.previewDebounce[which]);
      STATE.previewDebounce[which] = requestAnimationFrame(() => {
        STATE.previewDebounce[which] = 0;
        updateMarkdownPreview(which, true);
      });
    });
  });
  updateActiveMarkdownPreview();
  injectMarkdownViewSwitchers();
}

function injectMarkdownViewSwitchers() {
  document.querySelectorAll('.markdown-workbench').forEach((wb) => {
    if (wb.querySelector('.md-view-switch')) return;
    wb.classList.add('markdown-workbench--view-edit');
    const bar = document.createElement('div');
    bar.className = 'md-view-switch';
    bar.setAttribute('role', 'tablist');
    bar.setAttribute('aria-label', 'Editor or preview');
    for (const view of ['edit', 'preview']) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'md-view-tab' + (view === 'edit' ? ' is-active' : '');
      btn.dataset.view = view;
      btn.setAttribute('role', 'tab');
      btn.setAttribute('aria-selected', view === 'edit' ? 'true' : 'false');
      btn.textContent = view === 'edit' ? 'Edit' : 'Preview';
      btn.addEventListener('click', () => setMarkdownView(wb, view));
      bar.appendChild(btn);
    }
    wb.insertBefore(bar, wb.firstChild);
  });
}

function setMarkdownView(wb, view) {
  wb.classList.toggle('markdown-workbench--view-edit', view === 'edit');
  wb.classList.toggle('markdown-workbench--view-preview', view === 'preview');
  wb.querySelectorAll('.md-view-tab').forEach((b) => {
    const on = b.dataset.view === view;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  if (view === 'preview') {
    const which = STATE.activePanel;
    if (MARKDOWN_PANELS.includes(which)) updateMarkdownPreview(which, true);
  }
}

function previewTitle(which) {
  // "interview" reflects whichever Markdown file the user picked in the
  // embedded read-only viewer on the Claude tab.
  const names = {
    notes: 'NOTES.md',
    interview: STATE.interviewMdFile || 'PLAN.md',
  };
  const taskTitle = $('#task-title')?.textContent?.trim() || 'Task';
  return `${names[which] || 'Preview'} · ${taskTitle}`;
}

// ===== Embedded markdown picker (Claude tab) =====

// Apply a new task payload's file list + contents to STATE, refresh the
// <select> options, and load the currently selected file into the embed.
// Falls back to PLAN.md when the previous selection has disappeared.
function applyInterviewMdPayload(d) {
  const planText = d.templates && d.templates[FILES.plan] != null
    ? d.templates[FILES.plan]
    : '';
  let files = Array.isArray(d.task_markdown_files) && d.task_markdown_files.length
    ? d.task_markdown_files.slice()
    : [FILES.plan];
  if (!files.includes(FILES.plan)) files = [FILES.plan, ...files];
  const contents = {};
  for (const name of files) {
    if (d.templates && Object.prototype.hasOwnProperty.call(d.templates, name)) {
      contents[name] = d.templates[name];
    } else if (name === FILES.plan) {
      contents[name] = planText;
    } else {
      contents[name] = '';
    }
  }
  STATE.interviewMdFiles = files;
  STATE.interviewMdContents = contents;
  if (!files.includes(STATE.interviewMdFile)) {
    STATE.interviewMdFile = FILES.plan;
  }
  populateInterviewMdSelect();
  loadInterviewMdIntoEditor();
}

function populateInterviewMdSelect() {
  const sel = document.getElementById('interview-md-select');
  if (!sel) return;
  const current = STATE.interviewMdFile;
  const wanted = STATE.interviewMdFiles.map((name) => `${name}`).join('\u0000');
  if (sel.dataset.options !== wanted) {
    sel.innerHTML = '';
    for (const name of STATE.interviewMdFiles) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    }
    sel.dataset.options = wanted;
  }
  if (sel.value !== current) sel.value = current;
  updateInterviewMdHint();
  if (!sel.dataset.bound) {
    sel.dataset.bound = '1';
    sel.addEventListener('change', onInterviewMdSelectChange);
  }
}

function onInterviewMdSelectChange(ev) {
  const name = ev.target.value;
  if (!name || !STATE.interviewMdFiles.includes(name)) return;
  STATE.interviewMdFile = name;
  loadInterviewMdIntoEditor();
  updateInterviewMdHint();
}

function loadInterviewMdIntoEditor() {
  const editor = document.getElementById('editor-interview');
  if (!editor) return;
  const name = STATE.interviewMdFile;
  const text = STATE.interviewMdContents[name] || '';
  const changed = editor.value !== text;
  if (changed) {
    editor.value = text;
    STATE.previewCache.interview = null;
  }
  // Only re-render the preview when the text actually changed and the
  // user is looking at the panel - polling otherwise wastefully runs
  // the markdown renderer every 4s on unchanged content.
  if (changed && (STATE.activePanel === 'claude' || STATE.activePanel === 'interview')) {
    updateMarkdownPreview('interview', true);
  }
}

function updateInterviewMdHint() {
  const hint = document.getElementById('interview-md-hint');
  if (!hint) return;
  if (STATE.interviewMdFile === FILES.plan) {
    hint.textContent = 'Read-only preview of PLAN.md.';
  } else {
    hint.textContent = `Read-only preview of ${STATE.interviewMdFile}.`;
  }
}

async function openFullscreenPreview(which) {
  updateMarkdownPreview(which);
  const source = $(`#preview-${which}`);
  const modal = $('#preview-modal');
  const card = modal.querySelector('.preview-modal__card');
  const title = $('#preview-modal-title');
  const content = $('#preview-modal-content');
  if (!source || !modal || !card || !title || !content) return;
  title.textContent = previewTitle(which);
  content.innerHTML = source.innerHTML;
  modal.dataset.preview = which;
  modal.hidden = false;
  document.body.classList.add('preview-open');
  requestAnimationFrame(() => {
    content.scrollTop = 0;
    card.scrollTop = 0;
  });
  try {
    if (card.requestFullscreen && !document.fullscreenElement) await card.requestFullscreen();
  } catch { /* fullscreen may be blocked */ }
}

async function closeFullscreenPreview() {
  const modal = $('#preview-modal');
  if (!modal) return;
  modal.hidden = true;
  document.body.classList.remove('preview-open');
  if (document.fullscreenElement) {
    try { await document.exitFullscreen(); } catch { /* ignore */ }
  }
}

function printFullscreenPreview() {
  const modal = $('#preview-modal');
  if (!modal || modal.hidden) return;
  // Some browsers print the wrong viewport when triggered from within
  // requestFullscreen(); drop fullscreen first, let layout settle, then
  // print.  Two RAFs are enough for Chrome/Firefox to lay out the @media
  // print rules before window.print() snapshots them.
  const fire = () => {
    requestAnimationFrame(() => requestAnimationFrame(() => window.print()));
  };
  if (document.fullscreenElement) {
    document.exitFullscreen().then(fire, fire);
  } else {
    fire();
  }
}

function initFullscreenPreviews() {
  MARKDOWN_PANELS.forEach((which) => {
    const preview = $(`#preview-${which}`);
    if (!preview) return;
    preview.title = 'Double-click to open fullscreen preview';
    preview.addEventListener('dblclick', () => openFullscreenPreview(which));
  });
}

// ===== Tasks =====

async function loadTasks() {
  STATE.tasks = [];
  if (!STATE.projectId) {
    renderTasksFromState();
    return;
  }
  const { tasks } = await api('/api/tasks');
  STATE.tasks = tasks || [];
  renderTasksFromState();
}

function clearTaskDropMarkers(root = document) {
  root.querySelectorAll('.task-list li.is-drop-before, .task-list li.is-drop-after').forEach((el) => {
    el.classList.remove('is-drop-before', 'is-drop-after');
  });
}

async function reorderTasksByDrag(dragSlug, targetSlug, afterTarget) {
  const slugs = (STATE.tasks || []).map((t) => t.slug);
  const from = slugs.indexOf(dragSlug);
  const target = slugs.indexOf(targetSlug);
  if (from < 0 || target < 0 || dragSlug === targetSlug) return;
  slugs.splice(from, 1);
  const targetAfterRemoval = slugs.indexOf(targetSlug);
  slugs.splice(targetAfterRemoval + (afterTarget ? 1 : 0), 0, dragSlug);
  if (slugs.every((slug, idx) => slug === (STATE.tasks[idx] && STATE.tasks[idx].slug))) return;
  const bySlug = new Map((STATE.tasks || []).map((t) => [t.slug, t]));
  STATE.tasks = slugs.map((slug) => bySlug.get(slug)).filter(Boolean);
  renderTasksFromState();
  try {
    const d = await api('/api/tasks/reorder', {
      method: 'POST',
      body: JSON.stringify({ slugs }),
    });
    STATE.tasks = d.tasks || STATE.tasks || [];
    renderTasksFromState();
  } catch (e) {
    alert(e.message);
    await loadTasks();
  }
}

function renderTasksFromState() {
  const ul = $('#task-list');
  if (!ul) return;
  const selected = STATE.slug;
  ul.innerHTML = '';
  const all = STATE.tasks || [];
  const filter = (STATE.taskFilter || '').trim().toLowerCase();
  const tasks = filter
    ? all.filter((t) => `${t.title || ''} ${t.slug || ''}`.toLowerCase().includes(filter))
    : all;
  const countEl = document.getElementById('task-count');
  if (countEl) {
    countEl.textContent = filter && all.length !== tasks.length
      ? `${tasks.length}/${all.length}`
      : (all.length ? String(all.length) : '');
  }
  if (!tasks.length) {
    const li = document.createElement('li');
    li.className = 'task-list__empty';
    if (!STATE.projectId) li.textContent = 'Select or add a project';
    else if (filter) li.textContent = `No tasks match "${filter}"`;
    else li.textContent = 'No tasks yet';
    ul.appendChild(li);
    return;
  }
  for (const t of tasks) {
    const li = document.createElement('li');
    li.dataset.slug = t.slug;
    li.draggable = true;
    if (t.slug === selected) li.classList.add('active');
    li.innerHTML = `<div class="task-title">${escapeHtml(t.title)}</div><div class="task-slug">${escapeHtml(t.slug)}</div>`;
    li.addEventListener('dragstart', (ev) => {
      TASK_DRAG_SLUG = t.slug;
      TASK_JUST_DRAGGED = true;
      li.classList.add('is-dragging');
      ev.dataTransfer.effectAllowed = 'move';
      ev.dataTransfer.setData('text/plain', t.slug);
    });
    li.addEventListener('dragover', (ev) => {
      if (!TASK_DRAG_SLUG || TASK_DRAG_SLUG === t.slug) return;
      ev.preventDefault();
      ev.dataTransfer.dropEffect = 'move';
      const rect = li.getBoundingClientRect();
      const after = ev.clientY > rect.top + (rect.height / 2);
      clearTaskDropMarkers(ul);
      li.classList.toggle('is-drop-before', !after);
      li.classList.toggle('is-drop-after', after);
    });
    li.addEventListener('drop', async (ev) => {
      if (!TASK_DRAG_SLUG || TASK_DRAG_SLUG === t.slug) return;
      ev.preventDefault();
      const dragSlug = ev.dataTransfer.getData('text/plain') || TASK_DRAG_SLUG;
      const after = li.classList.contains('is-drop-after');
      clearTaskDropMarkers(ul);
      await reorderTasksByDrag(dragSlug, t.slug, after);
    });
    li.addEventListener('dragend', () => {
      TASK_DRAG_SLUG = '';
      li.classList.remove('is-dragging');
      clearTaskDropMarkers(ul);
      setTimeout(() => { TASK_JUST_DRAGGED = false; }, 0);
    });
    li.addEventListener('click', () => {
      if (TASK_JUST_DRAGGED) return;
      if (STATE.slug === t.slug) clearTaskSelection();
      else {
        selectTask(t.slug);
        if (isMobileViewport()) setSidebarOpen(false);
      }
    });
    ul.appendChild(li);
  }
}

function clearTaskSelection() {
  savePaneDraftForTask(STATE.slug);
  STATE.slug = null;
  STATE.currentMeta = null;
  STATE.worktreeStatuses = [];
  STATE.taskRoot = '';
  STATE.planPath = '';
  if (STATE.paneTimer) {
    clearInterval(STATE.paneTimer);
    STATE.paneTimer = null;
  }
  document.querySelectorAll('#task-list li').forEach((li) => li.classList.remove('active'));
  restorePaneDraftForTask(null);
  $('#task-view').hidden = true;
  $('#task-empty').hidden = false;
  const hdr = document.getElementById('hdr-skills');
  if (hdr) hdr.textContent = STATE.skillsPath || '—';
  renderTaskSkillsPicker({});
}

async function selectTask(slug) {
  if (STATE.slug && STATE.slug !== slug) {
    savePaneDraftForTask(STATE.slug);
  }
  STATE.slug = slug;
  STATE.changesData = null;
  STATE.changesSelected = '';
  document.querySelectorAll('#task-list li').forEach((li) => {
    li.classList.toggle('active', li.dataset.slug === slug);
  });

  // ---------- Optimistic render ----------
  // The sidebar's loadTasks() already cached the full TaskMeta for every
  // task. Render the header / tab bar / agent labels from that cache
  // BEFORE awaiting the API so the click feels instant; the heavier
  // /api/tasks/<slug> response (worktree git status + claude session
  // enrichment + markdown contents) then enriches the view in-place.
  const cached = (STATE.tasks || []).find((t) => t.slug === slug) || null;
  if (cached) {
    STATE.currentMeta = cached;
    STATE.worktreeStatuses = [];
    $('#task-empty').hidden = true;
    $('#task-view').hidden = false;
    $('#task-title').textContent = cached.title || slug;
    $('#task-slug').textContent = cached.slug;
    $('#task-backend').textContent = `${agentLabel(cached.agent)}${cached.interview_model ? ' · ' + cached.interview_model : ''}`;
    $('#task-goal').textContent = cached.general_goal || '';
    $('#inp-interview-target').value = cached.tmux_interview_target || '';
    setTmuxOutputText(cached.tmux_interview_target
      ? 'Loading Claude pane…'
      : 'Click Start Claude to launch a tmux pane in the worktree.');
    // Empty out the markdown viewer so the previous task's content
    // doesn't briefly flash through.
    $('#editor-interview').value = '';
    restorePaneDraftForTask(slug);
    STATE.interviewMdContents = {};
    STATE.previewCache = {};
    applyAgentLabels(cached);
    buildTabs(cached);
    if (cached.kind === 'kernel') { showPanel('kernel'); initKernelLab(cached); }
    else showPanel(DEFAULT_TAB);
  }

  let d;
  try {
    d = await apiWithRetry('/api/tasks/' + encodeURIComponent(slug), {}, { attempts: 4, delayMs: 300 });
  } catch (err) {
    if (STATE.slug !== slug) return;
    console.debug('selectTask detail load failed', err);
    const msg = isTransientApiError(err)
      ? 'Temporary gateway error while refreshing task details; kept cached task view.'
      : `Failed to refresh task details: ${err.message || err}`;
    const backend = document.getElementById('task-backend');
    if (backend && cached) backend.textContent = `${backend.textContent.replace(/ · refresh failed.*$/, '')} · refresh failed`;
    if (!cached) setTmuxOutputText(msg);
    return;
  }
  // The user may have clicked a different task while we were awaiting -
  // abort cleanly so we don't trample the newer selection.
  if (STATE.slug !== slug) return;

  // ---------- Full render with fresh server data ----------
  $('#task-empty').hidden = true;
  $('#task-view').hidden = false;
  $('#task-title').textContent = d.meta.title || slug;
  $('#task-slug').textContent = d.meta.slug;
  $('#task-backend').textContent = `${agentLabel(d.meta.agent)}${d.meta.interview_model ? ' · ' + d.meta.interview_model : ''}`;
  $('#task-goal').textContent = d.meta.general_goal || '';
  STATE.currentMeta = d.meta || null;
  STATE.worktreeStatuses = d.worktree_statuses || [];
  STATE.taskRoot = d.task_root || '';
  STATE.planPath = d.plan_path || '';
  applyInterviewMdPayload(d);
  invalidatePreviewCache();
  updateActiveMarkdownPreview();
  $('#inp-interview-target').value = d.meta.tmux_interview_target || '';
  if (!d.meta.tmux_interview_target) {
    setTmuxOutputText('Click Start Claude to launch a tmux pane in the worktree.');
  }
  restorePaneDraftForTask(slug);
  renderClaudeInfo(d.meta, d.claude || null, STATE.worktreeStatuses);
  if (d.meta.kind === 'kernel') renderKernelWorktrees(d.meta, STATE.worktreeStatuses);
  applyAgentLabels(d.meta || {});
  buildTabs(d.meta);
  // Keep the user on whatever panel optimistic-render showed (DEFAULT_TAB
  // by default); calling showPanel again would re-trigger the deferred
  // refresh callbacks unnecessarily.
  if (!cached) {
    if (d.meta.kind === 'kernel') { showPanel('kernel'); initKernelLab(d.meta); }
    else showPanel(DEFAULT_TAB);
  }
  refreshInterviewPreview(true);
  refreshClaudeSessions();
  startPanePolling();
}

function applyAgentLabels(meta) {
  const label = agentLabel(meta.agent);
  const startBtn = document.getElementById('btn-interview-start');
  const pasteBtn = document.getElementById('btn-interview-paste');
  const stopBtn = document.getElementById('btn-interview-stop');
  if (startBtn) startBtn.textContent = `Start ${label}`;
  if (pasteBtn) pasteBtn.textContent = 'Start Deep Interview';
  if (stopBtn) stopBtn.textContent = `Stop ${label}`;
  const heading = document.querySelector('.tab-panel[data-panel="claude"] .terminal-card__bar h4');
  if (heading) heading.textContent = `${label} Terminal`;
  const out = document.getElementById('interview-out');
  if (out && /^Click Start (Claude|Codex)/.test(out.textContent || '')) {
    setTmuxOutputText(`Click Start ${label} to launch a tmux pane in the worktree.`);
  }
  // Show agent in info card + bind change.
  const sel = document.getElementById('claude-info-agent');
  if (sel) {
    sel.value = normalizeAgent(meta.agent);
    // bind once
    if (!sel.dataset.bound) {
      sel.dataset.bound = '1';
      sel.addEventListener('change', onAgentChange);
    }
  }
  renderTaskSkillsPicker(meta);
}

function renderTaskSkillsPicker(meta = STATE.currentMeta || {}) {
  const sel = document.getElementById('claude-info-skills');
  if (!sel) return;
  const current = String(meta.skills_path || STATE.skillsPath || '').trim();
  const options = STATE.skillsOptions.length
    ? STATE.skillsOptions.slice()
    : (current ? [{ label: current, path: current }] : []);
  if (current && !options.some((opt) => String(opt.path || '') === current)) {
    options.unshift({ label: current, path: current });
  }
  const wanted = options.map((opt) => String(opt.path || '')).join('\u0000');
  if (sel.dataset.options !== wanted) {
    sel.innerHTML = '';
    for (const opt of options) {
      const path = String(opt.path || '').trim();
      if (!path) continue;
      const option = document.createElement('option');
      option.value = path;
      option.textContent = opt.label || path;
      option.title = path;
      sel.appendChild(option);
    }
    sel.dataset.options = wanted;
  }
  sel.disabled = !STATE.slug || sel.options.length === 0;
  if ([...sel.options].some((opt) => opt.value === current)) {
    sel.value = current;
  }
  const hdr = document.getElementById('hdr-skills');
  if (hdr && current) hdr.textContent = current;
  if (!sel.dataset.bound) {
    sel.dataset.bound = '1';
    sel.addEventListener('change', onTaskSkillsChange);
  }
}

async function onTaskSkillsChange(ev) {
  const sel = ev.target;
  const skillsPath = String(sel.value || '').trim();
  if (!STATE.slug || !skillsPath) return;
  const previous = STATE.currentMeta?.skills_path || STATE.skillsPath || '';
  sel.disabled = true;
  try {
    const r = await saveTaskMeta({ skills_path: skillsPath });
    if (r?.meta) {
      renderTaskSkillsPicker(r.meta);
      const hint = document.getElementById('claude-info-skills-hint');
      if (hint) {
        hint.textContent = 'saved for next deep interview';
        setTimeout(() => {
          hint.textContent = 'used when starting deep interview';
        }, 1800);
      }
    }
  } catch (err) {
    alert(err.message || 'failed to update skills');
    if (previous) sel.value = previous;
  } finally {
    sel.disabled = false;
  }
}

async function onAgentChange(ev) {
  const sel = ev.target;
  const next = sel.value;
  if (!STATE.slug) return;
  // Look at the current meta to confirm change is meaningful.
  if (!confirm(`Switch agent to ${agentLabel(next)}? Stop any running pane first; the new pane will use the ${agentLabel(next)} CLI.`)) {
    sel.value = sel.dataset.previous || 'claude';
    return;
  }
  sel.dataset.previous = next;
  try {
    const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/meta', {
      method: 'PUT',
      body: JSON.stringify({ agent: next }),
    });
    if (r.meta) await selectTask(STATE.slug);
  } catch (err) {
    alert(err.message || 'agent switch failed');
  }
}

// ===== Per-task terminal drafts =====

function paneDraftKey(slug = STATE.slug) {
  if (!slug) return '';
  // Slugs can repeat across projects, so include the active project id.
  return `${STATE.projectId || 'default'}::${slug}`;
}

function savePaneDraftForTask(slug = STATE.slug) {
  const input = document.getElementById('interview-in');
  const key = paneDraftKey(slug);
  if (!input || !key) return;
  STATE.paneDrafts[key] = input.value;
}

function restorePaneDraftForTask(slug = STATE.slug) {
  const input = document.getElementById('interview-in');
  if (!input) return;
  const key = paneDraftKey(slug);
  input.value = key ? (STATE.paneDrafts[key] || '') : '';
}

function clearPaneDraftForTask(slug = STATE.slug) {
  const key = paneDraftKey(slug);
  if (key) STATE.paneDrafts[key] = '';
}

async function deleteSelectedTask() {
  if (!STATE.slug) return;
  const slug = STATE.slug;
  const title = $('#task-title')?.textContent || slug;
  const ok = confirm(
    `Delete task "${title}" (${slug})?\n\n` +
    `This permanently removes .RUD/${slug}/, including PLAN.md, the worktree, and task metadata. ` +
    `Running tmux sessions are not stopped automatically.`
  );
  if (!ok) return;
  const btn = document.getElementById('btn-delete-task');
  if (btn) btn.disabled = true;
  try {
    await api('/api/tasks/' + encodeURIComponent(slug), { method: 'DELETE' });
    clearTaskSelection();
    await loadTasks();
    await loadTmuxSessions();
  } catch (e) {
    alert(e.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function refreshTaskTemplates() {
  if (!STATE.slug) return;
  if (STATE.pollInFlight.templates) return;
  const slug = STATE.slug;
  STATE.pollInFlight.templates = true;
  try {
    const d = await apiWithRetry('/api/tasks/' + encodeURIComponent(slug), {}, { attempts: 2, delayMs: 300 });
    if (STATE.slug !== slug) return;
    STATE.currentMeta = d.meta || STATE.currentMeta;
    STATE.worktreeStatuses = d.worktree_statuses || STATE.worktreeStatuses || [];
    STATE.taskRoot = d.task_root || STATE.taskRoot || '';
    STATE.planPath = d.plan_path || STATE.planPath || '';
    // Sync the picker payload (file list + content) and reload the embed
    // from the freshly fetched contents.
    applyInterviewMdPayload(d);
    if (d.meta) renderClaudeInfo(d.meta, d.claude || null, STATE.worktreeStatuses);
  } catch (err) {
    console.debug('refreshTaskTemplates failed', err);
  } finally {
    STATE.pollInFlight.templates = false;
  }
}

async function saveTemplate(name, textareaId, statusId) {
  if (!STATE.slug) return;
  const content = $(textareaId).value;
  await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/template', {
    method: 'PUT',
    body: JSON.stringify({ name, content }),
  });
  if (statusId) {
    $(statusId).textContent = 'Saved';
    setTimeout(() => { $(statusId).textContent = ''; }, 2000);
  }
}

// ===== Interview pane (tmux) =====

// Smart updater for the captured-tmux <pre>. By default we scroll to the
// bottom (most-recent output, the way a real terminal feels). If the
// user has scrolled up to read earlier output, we leave their position
// alone so polling doesn't yank them away.
function setTmuxOutputText(text, scrollMode = 'auto') {
  const out = document.getElementById('interview-out');
  if (!out) return;
  // When the element isn't laid out yet (e.g. tab hidden, clientHeight=0)
  // treat that as "near bottom" so the next time the user actually sees
  // the pane it lands at the latest output.
  const nearBottom =
    out.clientHeight === 0
    || (out.scrollHeight - out.clientHeight - out.scrollTop) < 80;
  out.textContent = text;
  if (scrollMode === 'top') {
    requestAnimationFrame(() => {
      out.scrollTop = 0;
    });
  } else if (nearBottom) {
    scrollTmuxOutputToBottom();
  }
}

function scrollTmuxOutputToBottom() {
  const out = document.getElementById('interview-out');
  if (!out) return;
  // Defer to next frame so scrollHeight reflects the freshly-set text
  // (and so any "display: block" tab activation has actually laid out).
  requestAnimationFrame(() => {
    out.scrollTop = out.scrollHeight;
  });
}

function revealInterviewTerminal(block = 'center') {
  const card = document.querySelector('.terminal-card--interview') || document.getElementById('interview-out');
  if (!card) return;
  requestAnimationFrame(() => {
    card.scrollIntoView({ block, inline: 'nearest', behavior: 'smooth' });
  });
}

async function refreshInterviewPreview(force = false, scrollMode = 'auto') {
  const target = $('#inp-interview-target').value.trim();
  if (!target) return;
  if (!force && STATE.pollInFlight.capture) return;
  STATE.pollInFlight.capture = true;
  try {
    const d = await api('/api/tmux/capture?target=' + encodeURIComponent(target) + '&lines=200');
    if ($('#inp-interview-target').value.trim() !== target) return;
    setTmuxOutputText(d.ok ? d.text : (d.error || '(error)'), scrollMode);
  } catch (err) {
    if (isTransientApiError(err)) {
      console.debug('tmux capture transient failure', err);
    } else {
      setTmuxOutputText(err.message);
    }
  } finally {
    STATE.pollInFlight.capture = false;
  }
}

// ===== Changes (read-only git diff) tab =====

const CHANGES_STATUS_GLYPH = { added: 'A', deleted: 'D', renamed: 'R', modified: 'M' };

function changesBaseName(p) {
  const s = String(p || '').replace(/\/+$/, '');
  const i = s.lastIndexOf('/');
  return i >= 0 ? s.slice(i + 1) : s;
}

function changesFileByKey(d, key) {
  if (!d || !key) return null;
  const [wi, fi] = key.split(':').map(Number);
  const wt = d.worktrees && d.worktrees[wi];
  if (!wt) return null;
  return (wt.files && wt.files[fi]) || null;
}

async function refreshChangesView(force = false) {
  if (!STATE.slug) return;
  const body = document.getElementById('changes-body');
  const statusEl = document.getElementById('changes-status');
  if (!body) return;
  if (STATE.changesData && !force) {
    renderChanges(STATE.changesData);
  } else if (!STATE.changesData) {
    body.innerHTML = '<div class="changes-empty">Loading changes…</div>';
  }
  if (STATE.changesLoading) return;
  STATE.changesLoading = true;
  if (statusEl) statusEl.textContent = 'Loading…';
  const slug = STATE.slug;
  try {
    const d = await apiWithRetry(
      '/api/tasks/' + encodeURIComponent(slug) + '/diff', {}, { attempts: 2 }
    );
    if (STATE.slug !== slug) return;
    STATE.changesData = d;
    renderChanges(d);
    if (statusEl) statusEl.textContent = '';
  } catch (err) {
    if (statusEl) statusEl.textContent = '';
    if (!STATE.changesData) {
      body.innerHTML = '<div class="changes-empty">Failed to load changes: ' + escapeHtml(err.message) + '</div>';
    }
  } finally {
    STATE.changesLoading = false;
  }
}

function changesFileRowHtml(f, key, sel) {
  const st = f.status || 'modified';
  const glyph = CHANGES_STATUS_GLYPH[st] || 'M';
  const stat = (f.additions || f.deletions)
    ? '<span class="changes-file__stat"><span class="add">+' + (f.additions || 0) + '</span><span class="del">-' + (f.deletions || 0) + '</span></span>'
    : '';
  const name = f.old_path
    ? (escapeHtml(f.old_path) + ' → ' + escapeHtml(f.path))
    : escapeHtml(f.path);
  return '<button type="button" class="changes-file' + (key === sel ? ' is-active' : '') + '" data-key="' + key + '" title="' + escapeHtml(f.path) + '">' +
    '<span class="changes-file__status changes-file__status--' + st + '">' + glyph + '</span>' +
    '<span class="changes-file__path">' + name + '</span>' + stat + '</button>';
}

function renderChanges(d) {
  const body = document.getElementById('changes-body');
  if (!body) return;
  const worktrees = (d && d.worktrees) || [];
  const totalFiles = worktrees.reduce((n, wt) => n + ((wt.files || []).length), 0);
  if (!worktrees.length) {
    const agent = agentLabel(STATE.currentMeta && STATE.currentMeta.agent);
    body.innerHTML = '<div class="changes-empty">No worktree registered for this task yet. Create one from the ' + escapeHtml(agent) + ' tab to see changes here.</div>';
    return;
  }
  if (!totalFiles) {
    body.innerHTML = '<div class="changes-empty">No changes — every worktree matches its base branch with a clean working tree.</div>';
    return;
  }
  let sel = STATE.changesSelected;
  if (!changesFileByKey(d, sel)) {
    sel = '';
    for (let wi = 0; wi < worktrees.length && !sel; wi++) {
      if ((worktrees[wi].files || []).length) sel = wi + ':0';
    }
    STATE.changesSelected = sel;
  }
  const listParts = [];
  worktrees.forEach((wt, wi) => {
    const files = wt.files || [];
    const wtName = changesBaseName(wt.path);
    const branch = wt.branch || '(detached)';
    const base = wt.base ? (' · base ' + escapeHtml(wt.base)) : '';
    listParts.push('<div class="changes-wt">');
    listParts.push('<div class="changes-wt__head" title="' + escapeHtml(wt.path) + '"><span class="changes-wt__name">' + escapeHtml(wtName) + '</span><span class="changes-wt__branch">' + escapeHtml(branch) + base + '</span></div>');
    if (!files.length) {
      listParts.push('<div class="changes-wt__empty">clean</div>');
    } else {
      [['uncommitted', 'Uncommitted'], ['committed', 'Committed (vs base)']].forEach(([scope, label]) => {
        const rows = files
          .map((f, i) => [f, i])
          .filter(([f]) => f.scope === scope);
        if (!rows.length) return;
        listParts.push('<div class="changes-scope">' + label + '</div>');
        rows.forEach(([f, i]) => listParts.push(changesFileRowHtml(f, wi + ':' + i, sel)));
      });
    }
    listParts.push('</div>');
  });
  body.innerHTML =
    '<div class="changes-summary">' + totalFiles + ' file' + (totalFiles === 1 ? '' : 's') + ' changed' + (d && d.worktrees.length > 1 ? ' across ' + d.worktrees.length + ' worktrees' : '') + '</div>' +
    '<div class="changes-layout">' +
      '<div class="changes-filelist">' + listParts.join('') + '</div>' +
      '<div class="changes-diff" id="changes-diff"></div>' +
    '</div>';
  body.querySelectorAll('.changes-file').forEach((btn) => {
    btn.addEventListener('click', () => {
      STATE.changesSelected = btn.dataset.key;
      body.querySelectorAll('.changes-file').forEach((b) => b.classList.toggle('is-active', b === btn));
      renderChangesDiffPanel();
    });
  });
  renderChangesDiffPanel();
}

function renderDiffBody(f) {
  if (!f.patch || !f.patch.trim()) {
    return '<div class="changes-empty">' + (f.binary ? 'Binary file — no text preview.' : 'No diff preview available.') + '</div>';
  }
  const out = [];
  for (const line of f.patch.split('\n')) {
    let cls = 'ctx';
    if (line.startsWith('@@')) cls = 'hunk';
    else if (
      line.startsWith('+++') || line.startsWith('---') ||
      line.startsWith('diff --git') || line.startsWith('index ') ||
      line.startsWith('new file') || line.startsWith('deleted file') ||
      line.startsWith('rename ') || line.startsWith('similarity ') ||
      line.startsWith('old mode') || line.startsWith('new mode') ||
      line.startsWith('Binary ')
    ) cls = 'meta';
    else if (line.startsWith('+')) cls = 'add';
    else if (line.startsWith('-')) cls = 'del';
    out.push('<div class="diffline diffline--' + cls + '">' + (escapeHtml(line) || '&#8203;') + '</div>');
  }
  return '<div class="diffview">' + out.join('') + '</div>';
}

function renderChangesDiffPanel() {
  const host = document.getElementById('changes-diff');
  if (!host) return;
  const f = changesFileByKey(STATE.changesData, STATE.changesSelected);
  if (!f) {
    host.innerHTML = '<div class="changes-empty">Select a file to view its diff.</div>';
    return;
  }
  const glyph = CHANGES_STATUS_GLYPH[f.status] || 'M';
  const name = f.old_path ? (escapeHtml(f.old_path) + ' → ' + escapeHtml(f.path)) : escapeHtml(f.path);
  host.innerHTML =
    '<div class="changes-diff__head">' +
      '<span class="changes-file__status changes-file__status--' + (f.status || 'modified') + '">' + glyph + '</span>' +
      '<span class="changes-diff__path">' + name + '</span>' +
      '<span class="changes-diff__scope">' + (f.scope === 'committed' ? 'committed' : 'uncommitted') + '</span>' +
    '</div>' +
    renderDiffBody(f);
}

// ===== Run monitor (tmux watcher -> OpenClaw) =====

function formatMonitorTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch (_) { return iso; }
}

function applyMonitorState(d) {
  const input = document.getElementById('monitor-pattern');
  const toggle = document.getElementById('monitor-toggle');
  const status = document.getElementById('monitor-status');
  if (!input || !toggle || !status) return;
  const running = !!(d && d.running);
  toggle.checked = running;
  if (document.activeElement !== input) {
    input.value = (d && d.pattern) || '';
  }
  if (d && d.default_pattern) input.title = 'Default: ' + d.default_pattern;
  let txt = running ? 'watching the pane' : 'off';
  if (d && d.last_fired) {
    txt += ' · last fired ' + formatMonitorTime(d.last_fired);
    if (d.last_match) txt += ' (' + d.last_match + ')';
  }
  status.textContent = txt;
  status.classList.toggle('is-on', running);
}

async function loadMonitor() {
  if (!STATE.slug) return;
  const slug = STATE.slug;
  try {
    const d = await api('/api/tasks/' + encodeURIComponent(slug) + '/monitor');
    if (STATE.slug !== slug) return;
    applyMonitorState(d);
  } catch (err) {
    console.debug('loadMonitor failed', err);
  }
}

async function setMonitor(enabled) {
  if (!STATE.slug) return;
  const slug = STATE.slug;
  const input = document.getElementById('monitor-pattern');
  const status = document.getElementById('monitor-status');
  try {
    let d;
    if (enabled) {
      d = await api('/api/tasks/' + encodeURIComponent(slug) + '/monitor', {
        method: 'POST',
        body: JSON.stringify({ pattern: (input && input.value || '').trim() }),
      });
    } else {
      d = await api('/api/tasks/' + encodeURIComponent(slug) + '/monitor', { method: 'DELETE' });
    }
    if (STATE.slug !== slug) return;
    applyMonitorState(d);
  } catch (err) {
    if (status) status.textContent = 'error: ' + err.message;
    loadMonitor();
  }
}

async function sendPaneText(submit = false) {
  const target = $('#inp-interview-target').value.trim();
  if (!target) {
    alert('Start the interview pane first.');
    return;
  }
  const input = $('#interview-in');
  const text = input.value;
  if (!text && !submit) return;
  await api('/api/tmux/send-text', {
    method: 'POST',
    body: JSON.stringify({ target, text, submit }),
  });
  input.value = '';
  clearPaneDraftForTask();
  await refreshInterviewPreview(true);
}

async function sendPaneKey(key) {
  const target = $('#inp-interview-target').value.trim();
  if (!target) {
    alert('Start the interview pane first.');
    return;
  }
  await api('/api/tmux/send-key', {
    method: 'POST',
    body: JSON.stringify({ target, key }),
  });
  await refreshInterviewPreview(true);
}

function startPanePolling() {
  if (STATE.paneTimer) clearInterval(STATE.paneTimer);
  STATE.paneTimer = setInterval(() => {
    if (!STATE.slug) return;
    const claudeTab = document.querySelector('.tab-panel[data-panel="claude"]');
    if (claudeTab && !claudeTab.hidden) {
      refreshInterviewPreview(true);
      refreshTaskTemplates();
      refreshClaudeSessions();
    }
  }, 4000);
}

async function startInterviewPane() {
  if (!STATE.slug) return;
  showPanel('claude');
  setTmuxOutputText('Starting Claude Code pane…\nWhen Claude is ready, click Start Deep Interview to paste the prompt.');
  revealInterviewTerminal();
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/interview/start', {
    method: 'POST',
    body: '{}',
  });
  $('#inp-interview-target').value = r.target || '';
  $('#interview-target-label').textContent = r.target || 'Not started';
  await refreshInterviewPreview(true, 'top');
  revealInterviewTerminal();
  setTimeout(() => {
    refreshInterviewPreview(true, 'top');
    revealInterviewTerminal('nearest');
  }, 6500);
  setTimeout(() => {
    refreshInterviewPreview(true, 'top');
    revealInterviewTerminal('nearest');
  }, 10000);
}

async function pasteInterviewPrompt() {
  if (!STATE.slug) return;
  const target = $('#inp-interview-target').value.trim();
  if (!target) {
    alert('Start the Claude pane first.');
    return;
  }
  const btn = document.getElementById('btn-interview-paste');
  if (btn) btn.disabled = true;
  try {
    setTmuxOutputText('Pasting deep-interview prompt with general goal + skills…');
    const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/claude/paste-prompt', {
      method: 'POST',
      body: '{}',
    });
    $('#inp-interview-target').value = r.target || target;
    setTmuxOutputText(
      `Pasted deep-interview prompt (${r.prompt_chars || 0} chars, skills: ${r.has_skills ? 'yes' : 'no'}).\n` +
      'Refreshing terminal capture…'
    );
    setTimeout(refreshInterviewPreview, 700);
    setTimeout(refreshInterviewPreview, 2000);
  } catch (err) {
    setTmuxOutputText(`Failed to paste prompt: ${err.message || err}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function currentPlanPathForPrompt() {
  return STATE.planPath || `.RUD/${STATE.slug || '<task>'}/PLAN.md`;
}

async function sendWorkflowPrompt(kind, text) {
  const target = $('#inp-interview-target').value.trim();
  if (!STATE.slug || !target) {
    alert('Start the Claude pane first.');
    return;
  }
  try {
    setTmuxOutputText(`Sending workflow prompt: ${kind}…`);
    await api('/api/tmux/send-text', {
      method: 'POST',
      body: JSON.stringify({ target, text, submit: true }),
    });
    setTimeout(() => refreshInterviewPreview(true), 500);
    setTimeout(() => refreshTaskTemplates(), 1800);
  } catch (err) {
    setTmuxOutputText(`Failed to send ${kind}: ${err.message || err}`);
  }
}

async function writeInterviewToPlan() {
  const planPath = currentPlanPathForPrompt();
  await sendWorkflowPrompt(
    'write PLAN.md',
    `Please finish the interview phase now and write the result directly into ${planPath}.

Use this structure:
- Goal
- Context / Decisions from the interview
- Constraints / non-goals
- Acceptance criteria
- Next steps as checkbox items
- Progress Log / Result

Keep it concise and executable. Do not create INTERVIEW.md, TODO.md, PROGRESS.md, or any other task-state file.`
  );
}

async function runGoalFromPlan() {
  const planPath = currentPlanPathForPrompt();
  await sendWorkflowPrompt(
    'run /goal',
    `/goal Execute the task plan in ${planPath}. Keep ${planPath} updated with useful progress, blockers, decisions, and final results. Do not create separate status files.`
  );
}

async function writeResultToPlan() {
  const planPath = currentPlanPathForPrompt();
  await sendWorkflowPrompt(
    'write result',
    `Please summarize the current execution result back into ${planPath}.

Update only useful information:
- what was done
- important decisions
- test/eval results
- blockers or follow-up work
- final status

Remove obsolete noisy details, but preserve unrelated prior sections. Do not create separate status files.`
  );
}

// ===== Modals & sidebar =====

function openCreateModal() {
  if (!STATE.projectId) {
    alert('Select or add a project first.');
    return;
  }
  const modal = $('#create-modal');
  modal.hidden = false;
  $('#new-task-status').textContent = '';
  updateCreateAgentHint();
  requestAnimationFrame(() => $('#new-title').focus());
}

function closeCreateModal() {
  $('#create-modal').hidden = true;
}

const AGENT_HINTS = {
  claude: 'Claude Code pane. Resume a past session by UUID.',
  codex: 'Codex CLI pane. Resume with codex resume <id>.',
  kernel: 'TKCC kernel optimization. The task view becomes the Kernel Lab.',
};

function updateCreateAgentHint() {
  const sel = document.getElementById('new-agent-select');
  const hint = document.getElementById('new-agent-hint');
  if (!sel || !hint) return;
  hint.textContent = AGENT_HINTS[sel.value] || '';
}

function resetCreateForm() {
  $('#new-title').value = '';
  $('#new-goal').value = '';
  $('#new-task-status').textContent = '';
  renderSkillsPicker();
  const sel = document.getElementById('new-agent-select');
  if (sel) sel.value = 'claude';
  updateCreateAgentHint();
}

function isMobileViewport() {
  return window.matchMedia('(max-width: 820px)').matches;
}

function setSidebarOpen(open) {
  STATE.sidebarOpen = !!open;
  document.body.classList.toggle('sidebar-open', STATE.sidebarOpen);
  const toggle = document.getElementById('btn-sidebar-toggle');
  if (toggle) toggle.setAttribute('aria-expanded', STATE.sidebarOpen ? 'true' : 'false');
  const backdrop = document.getElementById('sidebar-backdrop');
  if (backdrop) backdrop.hidden = !STATE.sidebarOpen;
}

function toggleSidebar() {
  setSidebarOpen(!STATE.sidebarOpen);
}

// ===== Wire-up =====

(function initSidebarToggle() {
  const toggle = document.getElementById('btn-sidebar-toggle');
  if (toggle) toggle.addEventListener('click', toggleSidebar);
  const backdrop = document.getElementById('sidebar-backdrop');
  if (backdrop) backdrop.addEventListener('click', () => setSidebarOpen(false));
  window.addEventListener('resize', () => {
    if (!isMobileViewport() && STATE.sidebarOpen) setSidebarOpen(false);
  });
})();

document.getElementById('btn-add-project').addEventListener('click', openAddProjectModal);
document.getElementById('btn-add-project-close').addEventListener('click', closeAddProjectModal);
document.getElementById('btn-add-project-cancel').addEventListener('click', closeAddProjectModal);
document.getElementById('btn-add-project-save').addEventListener('click', submitAddProject);
$('#add-project-modal').addEventListener('click', (event) => {
  if (event.target.id === 'add-project-modal') closeAddProjectModal();
});

document.getElementById('btn-tmux-refresh').addEventListener('click', loadTmuxSessions);
document.getElementById('btn-tasks-refresh').addEventListener('click', loadTasks);

(function initTaskFilter() {
  const inp = document.getElementById('task-filter');
  if (!inp) return;
  let timer = 0;
  inp.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      STATE.taskFilter = inp.value;
      renderTasksFromState();
    }, 80);
  });
})();

// ===== Inline edit: task title + goal =====

function makeEditable(el, { multiline = false, placeholder = '', onSave }) {
  if (!el) return;
  el.classList.add('editable');
  el.title = 'Click to edit';
  el.addEventListener('click', (ev) => {
    if (el.dataset.editing === '1') return;
    ev.stopPropagation();
    el.dataset.editing = '1';
    const current = el.textContent || '';
    const input = document.createElement(multiline ? 'textarea' : 'input');
    if (!multiline) input.type = 'text';
    input.value = current;
    input.placeholder = placeholder;
    input.className = 'editable__input';
    if (multiline) input.rows = 3;
    el.innerHTML = '';
    el.appendChild(input);
    input.focus();
    if (!multiline) input.select();
    let done = false;
    const finish = async (commit) => {
      if (done) return;
      done = true;
      el.dataset.editing = '';
      const next = input.value.trim();
      if (!commit || next === current.trim()) {
        el.textContent = current;
        return;
      }
      el.textContent = next;
      try { await onSave(next); } catch (err) {
        el.textContent = current;
        alert(err.message || 'save failed');
      }
    };
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('keydown', (kev) => {
      if (kev.key === 'Escape') { kev.preventDefault(); finish(false); }
      if (kev.key === 'Enter' && !multiline) { kev.preventDefault(); finish(true); }
      if (kev.key === 'Enter' && (kev.ctrlKey || kev.metaKey) && multiline) {
        kev.preventDefault(); finish(true);
      }
    });
  });
}

async function saveTaskMeta(patch) {
  if (!STATE.slug) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/meta', {
    method: 'PUT',
    body: JSON.stringify(patch),
  });
  if (r.meta) {
    STATE.currentMeta = r.meta;
    // Update the local task list cache so the sidebar reflects the change.
    STATE.tasks = (STATE.tasks || []).map((t) => (t.slug === r.meta.slug ? r.meta : t));
    renderTasksFromState();
  }
  return r;
}

(function initTaskHeaderEditing() {
  makeEditable($('#task-title'), {
    placeholder: 'Task title',
    onSave: (title) => saveTaskMeta({ title }),
  });
  makeEditable($('#task-goal'), {
    multiline: true,
    placeholder: 'General goal',
    onSave: (general_goal) => saveTaskMeta({ general_goal }),
  });
})();
document.getElementById('btn-create-open').addEventListener('click', openCreateModal);
document.getElementById('btn-empty-create').addEventListener('click', openCreateModal);
document.getElementById('new-agent-select').addEventListener('change', updateCreateAgentHint);
document.getElementById('btn-create-close').addEventListener('click', closeCreateModal);
document.getElementById('btn-create-cancel').addEventListener('click', closeCreateModal);
document.getElementById('create-modal').addEventListener('click', (event) => {
  if (event.target.id === 'create-modal') closeCreateModal();
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!$('#preview-modal').hidden) closeFullscreenPreview();
  else if (!$('#notes-modal').hidden) closeNotesModal();
  else if (!$('#worktree-modal').hidden) closeWorktreeModal();
  else if (!$('#create-modal').hidden) closeCreateModal();
  else if (!$('#add-project-modal').hidden) closeAddProjectModal();
});

document.getElementById('btn-preview-close').addEventListener('click', closeFullscreenPreview);
document.getElementById('btn-preview-exit-fullscreen').addEventListener('click', closeFullscreenPreview);
document.getElementById('btn-preview-print').addEventListener('click', printFullscreenPreview);
document.getElementById('preview-modal').addEventListener('click', (event) => {
  if (event.target.id === 'preview-modal') closeFullscreenPreview();
});

document.getElementById('btn-worktree-push-all').addEventListener('click', pushAllWorktrees);

// ===== Project NOTES.md modal =====

async function openNotesModal() {
  if (!STATE.projectId) {
    alert('Select or add a project first.');
    return;
  }
  const modal = $('#notes-modal');
  if (!modal) return;
  modal.hidden = false;
  document.body.classList.add('preview-open');
  const editor = $('#editor-notes');
  const preview = $('#preview-notes');
  const status = $('#notes-modal-status');
  const pathEl = $('#notes-modal-path');
  status.textContent = 'Loading…';
  editor.disabled = true;
  try {
    const project = await api('/api/project');
    const projectRoot = project.projectRoot || '';
    if (pathEl) pathEl.textContent = projectRoot ? `${projectRoot}/.RUD/NOTES.md` : '.RUD/NOTES.md';
    const d = await api('/api/notes');
    editor.value = d.content || '';
    preview.innerHTML = renderMarkdown(editor.value);
    STATE.notesDirty = false;
    status.textContent = '';
  } catch (err) {
    status.textContent = err.message || 'Failed to load notes';
  } finally {
    editor.disabled = false;
    requestAnimationFrame(() => editor.focus());
  }
}

function closeNotesModal() {
  if (STATE.notesDirty && !confirm('Discard unsaved Notes changes?')) return;
  const modal = $('#notes-modal');
  if (modal) modal.hidden = true;
  document.body.classList.remove('preview-open');
}

async function saveNotes() {
  const editor = $('#editor-notes');
  const status = $('#notes-modal-status');
  if (!editor) return;
  STATE.notesSaving = true;
  status.textContent = 'Saving…';
  try {
    await api('/api/notes', {
      method: 'PUT',
      body: JSON.stringify({ content: editor.value }),
    });
    STATE.notesDirty = false;
    status.textContent = 'Saved';
    setTimeout(() => { if (status.textContent === 'Saved') status.textContent = ''; }, 1800);
  } catch (err) {
    status.textContent = err.message || 'Save failed';
  } finally {
    STATE.notesSaving = false;
  }
}

(function initNotesModalEditor() {
  const editor = $('#editor-notes');
  const preview = $('#preview-notes');
  if (!editor || !preview) return;
  editor.addEventListener('input', () => {
    STATE.notesDirty = true;
    requestAnimationFrame(() => { preview.innerHTML = renderMarkdown(editor.value); });
  });
  editor.addEventListener('keydown', (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && (ev.key === 's' || ev.key === 'S')) {
      ev.preventDefault();
      saveNotes();
    }
  });
})();

document.getElementById('btn-notes-open').addEventListener('click', openNotesModal);
document.getElementById('btn-notes-close').addEventListener('click', closeNotesModal);
document.getElementById('btn-notes-save').addEventListener('click', saveNotes);
document.getElementById('notes-modal').addEventListener('click', (event) => {
  if (event.target.id === 'notes-modal') closeNotesModal();
});

// ===== Kernel Lab (TKCC integration) =====

function kernelSpeedupText(best) {
  if (best && typeof best.speedup === 'number') return best.speedup.toFixed(2) + '×';
  return '—';
}

function setKernelBadge(up) {
  const b = $('#kernel-service-badge');
  if (!b) return;
  if (up === true) { b.dataset.state = 'up'; b.textContent = 'service: up'; }
  else if (up === false) { b.dataset.state = 'down'; b.textContent = 'service: down — Launch will start it'; }
  else { b.dataset.state = 'unknown'; b.textContent = 'service: …'; }
}

async function refreshKernelService() {
  try {
    const d = await api('/api/kernel/service');
    setKernelBadge(!!d.up);
  } catch { setKernelBadge(null); }
}

function applyKernelShapeTemplate() {
  const plugin = $('#kernel-plugin').value;
  const tpl = (STATE.kernelShapeTemplates || {})[plugin];
  if (tpl) $('#kernel-shape').value = JSON.stringify(tpl, null, 2);
}

async function loadKernelPlugins() {
  const d = await api('/api/kernel/plugins');
  STATE.kernelShapeTemplates = d.shape_templates || {};
  STATE.kernelUnverified = d.unverified || [];
  const pluginOpt = (v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}${(STATE.kernelUnverified || []).includes(v) ? ' ⚠ unverified' : ''}</option>`;
  const opt = (v) => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`;
  const sel = $('#kernel-plugin').value;
  $('#kernel-plugin').innerHTML = (d.plugins || []).map(pluginOpt).join('');
  if (sel) $('#kernel-plugin').value = sel;
  $('#kernel-target').innerHTML = (d.targets || []).map(opt).join('');
  $('#kernel-starter').innerHTML = (d.starter_modes || []).map(opt).join('');
  $('#kernel-model-list').innerHTML = (d.suggested_models || []).map((m) => `<option value="${escapeHtml(m)}"></option>`).join('');
  if (!$('#kernel-model').value && (d.suggested_models || []).length) {
    $('#kernel-model').value = d.suggested_models[0];
  }
  if (!STATE.kernelPluginListenerBound) {
    $('#kernel-plugin').addEventListener('change', applyKernelShapeTemplate);
    STATE.kernelPluginListenerBound = true;
  }
  applyKernelShapeTemplate();
}

// Runs that change over time and so are worth re-fetching/re-rendering live.
const KERNEL_LIVE_STATES = ['running', 'launching', 'resolving'];

// Compact signature of a run's *visible* state, so polling can skip DOM work
// when nothing changed (the main source of the "switching runs is laggy" jank).
function kernelRunSig(r) {
  const st = r.status || {};
  const best = (st.best && st.best.speedup) || '';
  const plugin = r.plugin || (r.config || {}).plugin || '';
  return [r.id, r.state || '', plugin, best, st.agents_running || 0, r.verified].join('|');
}

function highlightSelectedKernelRun() {
  document.querySelectorAll('#kernel-run-list .kernel-run').forEach((li) => {
    li.classList.toggle('is-active', li.dataset.runId === STATE.kernelSelected);
  });
}

function renderKernelRunsList(runs) {
  const ul = $('#kernel-run-list');
  if (!ul) return;
  if (!runs.length) {
    ul.innerHTML = '<li class="kernel-run--empty">No runs yet. Configure one and click Launch.</li>';
    return;
  }
  ul.innerHTML = '';
  runs.forEach((r) => {
    const cfg = r.config || {};
    const best = (r.status && r.status.best) || null;
    const li = document.createElement('li');
    li.dataset.runId = r.id;
    li.className = 'kernel-run' + (r.id === STATE.kernelSelected ? ' is-active' : '');
    const agents = r.status ? ` · ${r.status.agents_running || 0}/${(r.status.agents || []).length} agents` : '';
    const plugin = r.plugin || cfg.plugin || '';
    const unv = !!plugin && ((STATE.kernelUnverified || []).includes(plugin) || r.verified === false);
    const unvBadge = unv ? ' <span class="kernel-unverified">⚠ unverified</span>' : '';
    li.innerHTML =
      `<div class="kernel-run__head"><span class="kernel-run__plugin">${escapeHtml(cfg.plugin || plugin || '?')} · ${escapeHtml(cfg.target || (r.spec && r.spec.target) || '?')}</span>` +
      `<span class="kernel-run__state" data-state="${escapeHtml(r.state || '')}">${escapeHtml(r.state || '')}</span></div>` +
      `<div class="kernel-run__meta">best ${kernelSpeedupText(best)}${agents}${unvBadge}</div>`;
    li.addEventListener('click', () => selectKernelRun(r.id));
    ul.appendChild(li);
  });
}

async function loadKernelRuns() {
  const ul = $('#kernel-run-list');
  if (!ul) return;
  try {
    const d = await api('/api/kernel/runs');
    const runs = d.runs || [];
    STATE.kernelRuns = runs;
    // Only rebuild the list DOM when the runs actually changed.
    const sig = runs.map(kernelRunSig).join(';');
    if (sig !== STATE.kernelRunsSig) {
      STATE.kernelRunsSig = sig;
      renderKernelRunsList(runs);
    }
    highlightSelectedKernelRun();
  } catch (err) {
    ul.innerHTML = `<li class="status-bad">${escapeHtml(err.message)}</li>`;
  }
}

// Pull the useful text out of a run's error_detail (string, or a dict that may
// carry service/build stderr) so the UI can show *why* a launch failed.
function formatKernelErrorDetail(ed) {
  if (!ed) return '';
  if (typeof ed === 'string') return ed.slice(-4000);
  const parts = [];
  const svc = ed.service;
  if (svc && typeof svc === 'object') {
    if (svc.error) parts.push('service: ' + svc.error);
    if (svc.stderr) parts.push(String(svc.stderr));
  }
  for (const k of ['stderr', 'stdout', 'stdout_tail']) {
    if (ed[k]) parts.push(String(ed[k]));
  }
  if (!parts.length) {
    try { return JSON.stringify(ed, null, 2).slice(-4000); } catch { return String(ed).slice(-4000); }
  }
  return parts.join('\n\n').slice(-4000);
}

function renderKernelRunDetail(r) {
  const host = $('#kernel-run-detail');
  if (!host) return;
  const cfg = r.config || {};
  const st = r.status || {};
  const best = st.best || null;
  const bests = (st.agent_bests || []).slice().sort((a, b) => (b.speedup || 0) - (a.speedup || 0));
  const shape = typeof cfg.shape === 'string' ? cfg.shape : JSON.stringify(cfg.shape);
  const plugin = r.plugin || cfg.plugin || '';
  const unverified = !!plugin && ((STATE.kernelUnverified || []).includes(plugin) || r.verified === false);
  const vbadge = plugin ? (unverified ? ' <span class="kernel-unverified">⚠ unverified reference</span>' : ' <span class="kernel-verified">✓ verified</span>') : '';
  let html = `<div class="kernel-detail__head"><h4>${escapeHtml(cfg.plugin || plugin || '?')} · ${escapeHtml(cfg.target || (r.spec && r.spec.target) || '?')}${vbadge}</h4>`;
  html += `<button type="button" class="btn btn--danger btn--sm" id="btn-kernel-stop">Stop</button></div>`;
  if (cfg.build_mode) {
    html += `<div class="kernel-buildmode">Build mode (correctness-first) — ${best ? '✓ working kernel found' : '… no correct kernel yet'}</div>`;
  }
  html += '<div class="kernel-detail__grid">';
  html += `<span>State</span><span class="kernel-run__state" data-state="${escapeHtml(r.state || '')}">${escapeHtml(r.state || '')}</span>`;
  html += `<span>Run ID</span><code>${escapeHtml(r.run_id || '—')}</code>`;
  html += `<span>Shape</span><code>${escapeHtml(shape)}</code>`;
  html += `<span>Model</span><span>${escapeHtml(cfg.model || '')}</span>`;
  html += `<span>Best speedup</span><span class="kernel-speedup">${kernelSpeedupText(best)}</span>`;
  if (st.target_speedup) html += `<span>Target</span><span>${Number(st.target_speedup).toFixed(2)}×</span>`;
  html += `<span>Agents</span><span>${st.agents_running || 0} running / ${(st.agents || []).length}</span>`;
  html += `<span>Improvements</span><span>${st.improvements || 0}</span>`;
  html += '</div>';
  if (r.state === 'error' && r.error) html += `<p class="status-bad">${escapeHtml(r.error)}</p>`;
  // Live build / run log (docker build, service bring-up, agent launch).
  if (['launching', 'running', 'error'].includes(r.state)) {
    const shown = STATE.kernelLogRunId === r.id ? (STATE.kernelLogText || '') : '';
    html += '<details class="kernel-buildlog" open><summary>build / run log</summary>' +
      `<pre id="kernel-build-log">${escapeHtml(shown || '(loading log…)')}</pre></details>`;
  } else {
    const detail = formatKernelErrorDetail(r.error_detail);
    if (detail) html += `<details class="kernel-errdetail"><summary>details</summary><pre>${escapeHtml(detail)}</pre></details>`;
  }
  if (bests.length) {
    html += '<table class="kernel-leaderboard"><thead><tr><th>Agent</th><th>Speedup</th><th>kernel µs</th></tr></thead><tbody>';
    bests.forEach((e) => {
      html += `<tr><td>${escapeHtml(String(e.agent_index))}</td><td>${(e.speedup || 0).toFixed(2)}×</td><td>${(e.kernel_us || 0).toFixed(1)}</td></tr>`;
    });
    html += '</tbody></table>';
  }
  if (unverified && plugin) {
    html += '<div class="kernel-detail__verify"><button type="button" class="btn btn--sm" id="btn-kernel-verify">Mark reference verified</button>' +
      '<span class="tab-panel__hint"> review reference() (search TODO(review)) before marking</span></div>';
  }
  host.innerHTML = html;
  const stopBtn = $('#btn-kernel-stop');
  if (stopBtn) stopBtn.addEventListener('click', () => stopKernelRun(r.id));
  const verBtn = $('#btn-kernel-verify');
  if (verBtn) verBtn.addEventListener('click', () => markPluginVerified(plugin));
}

// Fetch + render the per-run build/run log tail (docker build, etc.).
async function refreshKernelBuildLog(id) {
  try {
    const d = await api('/api/kernel/runs/' + encodeURIComponent(id) + '/log');
    if (STATE.kernelSelected !== id) return;
    STATE.kernelLogRunId = id;
    STATE.kernelLogText = d.log || '';
    const pre = document.getElementById('kernel-build-log');
    if (!pre) return;
    const newText = STATE.kernelLogText || '(no log yet)';
    // Unchanged → don't touch the DOM/scroll at all (this is what made it jump).
    if (pre.textContent === newText) return;
    const atBottom = pre.clientHeight === 0
      || (pre.scrollHeight - pre.clientHeight - pre.scrollTop) < 60;
    const prevTop = pre.scrollTop;
    pre.textContent = newText;
    // Stick to the bottom if the user was already there; otherwise keep their
    // scroll position instead of snapping to the top.
    pre.scrollTop = atBottom ? pre.scrollHeight : prevTop;
  } catch { /* keep last */ }
}

async function selectKernelRun(id) {
  STATE.kernelSelected = id;
  highlightSelectedKernelRun();
  // Reset the cached log when switching to a different run.
  if (STATE.kernelLogRunId !== id) { STATE.kernelLogRunId = id; STATE.kernelLogText = ''; }
  const host = $('#kernel-run-detail');
  if (!host) return;
  host.hidden = false;
  // Render instantly from the cached run record so switching feels snappy.
  const cached = (STATE.kernelRuns || []).find((r) => r.id === id) || null;
  if (cached) {
    renderKernelRunDetail(cached);
    STATE.kernelDetailSig = kernelRunSig(cached);
    if (['launching', 'running', 'error'].includes(cached.state)) refreshKernelBuildLog(id);
  }
  // Only hit the network when the run is live (fresh leaderboard/agents) or we
  // had nothing cached to show.
  if (!cached || KERNEL_LIVE_STATES.includes(cached.state)) {
    try {
      const r = await api('/api/kernel/runs/' + encodeURIComponent(id));
      if (STATE.kernelSelected !== id) return;
      renderKernelRunDetail(r);
      STATE.kernelDetailSig = kernelRunSig(r);
      if (['launching', 'running', 'error'].includes(r.state)) refreshKernelBuildLog(id);
    } catch (err) {
      if (!cached) host.innerHTML = `<p class="status-bad">${escapeHtml(err.message)}</p>`;
    }
  }
}

// Poll-side refresh of the open run detail: re-fetch only live runs; otherwise
// re-render from cache only if that run's visible state changed.
async function refreshSelectedKernelDetail() {
  const id = STATE.kernelSelected;
  if (!id) return;
  const cached = (STATE.kernelRuns || []).find((r) => r.id === id) || null;
  if (!cached) return;
  if (KERNEL_LIVE_STATES.includes(cached.state)) {
    try {
      const r = await api('/api/kernel/runs/' + encodeURIComponent(id));
      if (STATE.kernelSelected !== id) return;
      // Only rebuild the detail DOM when the run's visible state changed, so a
      // growing log doesn't recreate the <pre> (and reset its scroll) every poll.
      const sig = kernelRunSig(r);
      if (sig !== STATE.kernelDetailSig) {
        renderKernelRunDetail(r);
        STATE.kernelDetailSig = sig;
      }
      refreshKernelBuildLog(id);
    } catch { /* keep last render */ }
  } else {
    const sig = kernelRunSig(cached);
    if (sig !== STATE.kernelDetailSig) {
      renderKernelRunDetail(cached);
      STATE.kernelDetailSig = sig;
    }
  }
}

async function launchKernelRun() {
  const status = $('#kernel-launch-status');
  let shape;
  try { shape = JSON.parse($('#kernel-shape').value); }
  catch { status.textContent = 'Shape must be valid JSON'; return; }
  const model = $('#kernel-model').value.trim();
  if (!model) { status.textContent = 'Model is required'; return; }
  const buildMode = $('#kernel-build-mode') ? $('#kernel-build-mode').checked : false;
  const body = {
    plugin: $('#kernel-plugin').value,
    target: $('#kernel-target').value,
    shape,
    model,
    n_agents: Number($('#kernel-nagents').value) || 1,
    starter_mode: $('#kernel-starter').value,
    auto_terminate: buildMode ? true : $('#kernel-auto-terminate').checked,
    build: $('#kernel-build').checked,
    build_mode: buildMode,
  };
  const ts = $('#kernel-target-speedup').value;
  if (ts) body.target_speedup = Number(ts);
  if (buildMode && body.target_speedup === undefined) body.target_speedup = 0;
  status.textContent = 'Launching… (first run builds the image — this can take a few minutes)';
  try {
    const r = await api('/api/kernel/runs', { method: 'POST', body: JSON.stringify(body) });
    status.textContent = 'Launched ' + r.id + ' — starting agents…';
    STATE.kernelSelected = r.id;
    await loadKernelRuns();
    await selectKernelRun(r.id);
  } catch (err) {
    status.textContent = err.message || 'Launch failed';
  }
}

async function stopKernelRun(id) {
  if (!confirm('Stop this run? Agents are terminated and the best kernel is postprocessed.')) return;
  try {
    await api('/api/kernel/runs/' + encodeURIComponent(id) + '/stop', { method: 'POST' });
    await loadKernelRuns();
    await selectKernelRun(id);
  } catch (err) { alert(err.message || 'Stop failed'); }
}

function startKernelPolling() {
  if (STATE.kernelTimer) clearInterval(STATE.kernelTimer);
  STATE.kernelPollTick = 0;
  STATE.kernelTimer = setInterval(async () => {
    const panel = document.querySelector('.tab-panel[data-panel="kernel"]');
    if (!panel || panel.hidden || STATE.kernelPolling) return;
    STATE.kernelPolling = true;
    try {
      STATE.kernelPollTick = (STATE.kernelPollTick || 0) + 1;
      // Service status rarely changes and costs a subprocess on the server;
      // poll it ~every 12s instead of every cycle.
      if (STATE.kernelPollTick % 3 === 1) await refreshKernelService();
      await loadKernelRuns();
      await refreshSelectedKernelDetail();
    } finally { STATE.kernelPolling = false; }
  }, 4000);
}

// Load the task's interview (messages + last spec) from disk so it survives
// task switches AND full page reloads. The interview turns themselves stay
// stateless (`claude -p`); we just persist the transcript next to the task.
async function loadKernelInterview(slug) {
  try {
    const d = await api('/api/tasks/' + encodeURIComponent(slug) + '/kernel-interview');
    if (STATE.slug !== slug) return; // user switched away mid-flight
    STATE.kernelChat = Array.isArray(d.messages) ? d.messages : [];
    renderKernelChat();
    STATE.kernelSpec = null;
    const sp = $('#kernel-spec'); if (sp) { sp.hidden = true; sp.innerHTML = ''; }
    if (d.spec) showKernelSpec(d.spec);
  } catch (err) {
    console.debug('loadKernelInterview failed', err);
  }
}

async function saveKernelInterview() {
  const slug = STATE.slug;
  if (!slug) return;
  try {
    await api('/api/tasks/' + encodeURIComponent(slug) + '/kernel-interview', {
      method: 'PUT',
      body: JSON.stringify({ messages: STATE.kernelChat || [], spec: STATE.kernelSpec || null }),
    });
  } catch (err) {
    console.debug('saveKernelInterview failed', err);
  }
}

// Called by selectTask when a kernel-kind task is opened: the task view's
// kernel panel IS the Kernel Lab, scoped to this task's project/worktree.
async function initKernelLab(meta) {
  const slug = STATE.slug;
  const projEl = $('#kernel-project');
  if (projEl) {
    const proj = (STATE.projects || []).find((x) => x.id === STATE.projectId);
    projEl.textContent = (meta && (meta.worktree_path || (meta.worktrees || [])[0])) || (proj ? proj.path : '');
  }
  renderKernelWorktrees(meta || {}, STATE.worktreeStatuses || []);
  const status = $('#kernel-launch-status');
  try { await loadKernelPlugins(); }
  catch (err) { if (status) status.textContent = err.message || 'Failed to load plugins'; }
  // Clear the previous task's interview from view, then load this task's
  // persisted transcript from .RUD/<slug>/kernel_interview.json.
  STATE.kernelChat = [];
  STATE.kernelSpec = null;
  renderKernelChat();
  const sp = $('#kernel-spec'); if (sp) { sp.hidden = true; sp.innerHTML = ''; }
  if (slug) await loadKernelInterview(slug);
  STATE.kernelSelected = null;
  // Reset the runs/detail caches so this task's list renders fresh.
  STATE.kernelRuns = [];
  STATE.kernelRunsSig = '';
  STATE.kernelDetailSig = '';
  const det = $('#kernel-run-detail'); if (det) { det.hidden = true; det.innerHTML = ''; }
  refreshKernelService();
  loadKernelRuns();
  startKernelPolling();
}

document.getElementById('btn-kernel-refresh').addEventListener('click', () => { refreshKernelService(); loadKernelRuns(); });
document.getElementById('btn-kernel-launch').addEventListener('click', launchKernelRun);
document.getElementById('btn-kernel-create-worktree').addEventListener('click', openWorktreeModal);
document.getElementById('btn-kernel-worktree-push-all').addEventListener('click', pushAllWorktrees);

// --- Kernel Lab interview (chat) ---

function renderKernelChat() {
  const host = $('#kernel-chat');
  if (!host) return;
  const msgs = STATE.kernelChat || [];
  if (!msgs.length) {
    host.innerHTML = '<div class="kernel-chat__msg kernel-chat__msg--assistant">Describe the kernel/operation to optimize — a GitHub raw URL works best. I\'ll ask what I can\'t infer, then produce a spec.</div>';
    return;
  }
  host.innerHTML = msgs.map((m) =>
    `<div class="kernel-chat__msg kernel-chat__msg--${m.role === 'user' ? 'user' : 'assistant'}">${escapeHtml(m.content)}</div>`
  ).join('');
  host.scrollTop = host.scrollHeight;
}

function fillFormFromSpec(spec) {
  if (!spec) return;
  if (spec.target && $('#kernel-target')) $('#kernel-target').value = spec.target;
  if (spec.shape && $('#kernel-shape')) $('#kernel-shape').value = JSON.stringify(spec.shape, null, 2);
  if (spec.model && $('#kernel-model')) $('#kernel-model').value = spec.model;
  if (spec.n_agents && $('#kernel-nagents')) $('#kernel-nagents').value = spec.n_agents;
  if (spec.starter_mode && $('#kernel-starter')) $('#kernel-starter').value = spec.starter_mode;
  if (spec.target_speedup && $('#kernel-target-speedup')) $('#kernel-target-speedup').value = spec.target_speedup;
  const st = $('#kernel-launch-status');
  if (st) st.textContent = 'Form pre-filled from the interview.';
}

async function prepareKernel(spec) {
  const status = $('#kernel-prepare-status');
  if (status) status.textContent = 'Resolving plugin (Claude is reading the source + registry — this can take a few minutes)…';
  try {
    const r = await api('/api/kernel/prepare', { method: 'POST', body: JSON.stringify({ spec }) });
    const id = r.id;
    for (let i = 0; i < 240; i += 1) {
      await sleep(2500);
      const rec = await api('/api/kernel/runs/' + encodeURIComponent(id));
      if (rec.state === 'prepared') {
        await loadKernelPlugins();
        fillFormFromSpec(spec);
        if (rec.plugin && $('#kernel-plugin')) {
          $('#kernel-plugin').value = rec.plugin;
          // If the interview spec didn't carry a usable shape, fall back to
          // the resolved plugin's shape template so the form isn't blank.
          if (!($('#kernel-shape').value || '').trim()) applyKernelShapeTemplate();
        }
        const v = rec.verified ? '' : ' ⚠ unverified reference — review before trusting results.';
        const nb = rec.needs_build ? ' New plugin: rebuild the eval image before launching.' : '';
        if (status) status.textContent = `Plugin ready: ${rec.plugin}.${v}${nb} Form filled — click Launch below.`;
        loadKernelRuns();
        return;
      }
      if (rec.state === 'error') {
        if (status) status.textContent = 'Prepare failed: ' + (rec.error || 'unknown');
        return;
      }
    }
    if (status) status.textContent = 'Prepare timed out (still resolving — check the runs list).';
  } catch (err) {
    if (status) status.textContent = 'Prepare error: ' + (err.message || 'failed');
  }
}

async function markPluginVerified(name) {
  if (!name) return;
  if (!confirm(`Mark the reference for "${name}" as verified? Only after reviewing reference() for correctness.`)) return;
  try {
    await api('/api/kernel/plugins/verify', { method: 'POST', body: JSON.stringify({ name }) });
    await loadKernelPlugins();
    await loadKernelRuns();
    if (STATE.kernelSelected) await selectKernelRun(STATE.kernelSelected);
  } catch (err) { alert(err.message || 'verify failed'); }
}

function showKernelSpec(spec) {
  STATE.kernelSpec = spec;
  const host = $('#kernel-spec');
  if (!host) return;
  host.hidden = false;
  host.innerHTML =
    '<div class="kernel-spec__title">Proposed spec</div>' +
    `<pre>${escapeHtml(JSON.stringify(spec, null, 2))}</pre>` +
    '<div class="kernel-spec__actions">' +
    '<button type="button" class="btn btn--primary" id="btn-kernel-prepare">🛠 Prepare (resolve plugin)</button>' +
    '<button type="button" class="btn" id="btn-kernel-spec-fill">Re-fill form</button>' +
    '</div><span class="tab-panel__hint" id="kernel-prepare-status"></span>';
  // Auto-fill the form fields (target / shape / model / agents / starter /
  // target-speedup) the moment the interview yields a spec - no extra click.
  // The plugin dropdown is the only field that still needs Prepare (the
  // resolver decides reuse-vs-create), so make that the highlighted action.
  fillFormFromSpec(spec);
  const status = $('#kernel-prepare-status');
  if (status) {
    status.textContent = 'Form auto-filled from the interview. Click Prepare to resolve the plugin (or pick one manually), then Launch.';
  }
  const pbtn = $('#btn-kernel-prepare');
  if (pbtn) pbtn.addEventListener('click', () => prepareKernel(spec));
  const btn = $('#btn-kernel-spec-fill');
  if (btn) btn.addEventListener('click', () => fillFormFromSpec(spec));
}

async function sendKernelChat() {
  const ta = $('#kernel-chat-text');
  if (!ta) return;
  const text = (ta.value || '').trim();
  if (!text) return;
  STATE.kernelChat = STATE.kernelChat || [];
  STATE.kernelChat.push({ role: 'user', content: text });
  ta.value = '';
  renderKernelChat();
  saveKernelInterview();
  const host = $('#kernel-chat');
  if (host) {
    const t = document.createElement('div');
    t.className = 'kernel-chat__msg kernel-chat__msg--assistant kernel-chat__msg--thinking';
    t.textContent = '…thinking';
    host.appendChild(t);
    host.scrollTop = host.scrollHeight;
  }
  try {
    const r = await api('/api/kernel/interview', { method: 'POST', body: JSON.stringify({ messages: STATE.kernelChat }) });
    if (r.done && r.spec) {
      STATE.kernelChat.push({ role: 'assistant', content: 'Got everything I need — spec below.' });
      renderKernelChat();
      showKernelSpec(r.spec);
    } else {
      STATE.kernelChat.push({ role: 'assistant', content: r.assistant || '(no response)' });
      renderKernelChat();
    }
    saveKernelInterview();
  } catch (err) {
    STATE.kernelChat.push({ role: 'assistant', content: 'Error: ' + (err.message || 'interview failed') });
    renderKernelChat();
    saveKernelInterview();
  }
}

document.getElementById('btn-kernel-chat-send').addEventListener('click', sendKernelChat);
document.getElementById('kernel-chat-text').addEventListener('keydown', (ev) => {
  if ((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter') { ev.preventDefault(); sendKernelChat(); }
});

// ===== Claude pane info card (worktree + session history + Resume) =====

function formatSessionMtime(ts) {
  if (!ts) return '';
  try {
    const d = new Date(ts * 1000);
    return d.toLocaleString();
  } catch { return ''; }
}

function shortSessionId(sid) {
  const s = String(sid || '');
  if (s.length <= 12) return s;
  return s.slice(0, 8);
}

// Build the worktree rows (path + branch + status + push/remove) into
// *wtHost*, and toggle *pushAllBtn* visibility. Shared by the Claude tab
// info card and the Kernel Lab worktree card so both stay in sync.
function renderWorktreeListInto(wtHost, pushAllBtn, meta, statuses, primaryLabel) {
  meta = meta || {};
  const worktrees = Array.isArray(meta.worktrees) && meta.worktrees.length
    ? meta.worktrees
    : (meta.worktree_path ? [meta.worktree_path] : []);
  const branches = Array.isArray(meta.branches) && meta.branches.length
    ? meta.branches
    : (meta.branch ? [meta.branch] : []);
  const statusByPath = {};
  for (const s of (Array.isArray(statuses) ? statuses : [])) {
    if (s && s.path) statusByPath[s.path] = s;
  }
  if (pushAllBtn) pushAllBtn.hidden = worktrees.length === 0;
  if (!wtHost) return;
  wtHost.innerHTML = '';
  if (!worktrees.length) {
    const hint = document.createElement('span');
    hint.className = 'claude-info__hint';
    hint.textContent = '(none — click + Add worktree, or git worktree add manually under .RUD/<slug>/work/)';
    wtHost.appendChild(hint);
    return;
  }
  worktrees.forEach((path, i) => {
    const row = document.createElement('div');
    row.className = 'wt-list-row' + (i === 0 ? ' wt-list-row--primary' : '');
    const main = document.createElement('div');
    main.className = 'wt-list-row__main';
    const pathEl = document.createElement('code');
    pathEl.className = 'wt-list-row__path';
    pathEl.textContent = path;
    if (i === 0) pathEl.title = primaryLabel || 'Primary worktree';
    main.appendChild(pathEl);
    const br = document.createElement('span');
    br.className = 'wt-list-row__branch';
    br.textContent = (branches[i] || '').trim() || '—';
    main.appendChild(br);
    const st = statusByPath[path];
    if (st) main.appendChild(renderWorktreeStatusBadge(st));
    row.appendChild(main);
    if (i === 0) {
      const pill = document.createElement('span');
      pill.className = 'wt-list-row__pill';
      pill.textContent = 'primary';
      row.appendChild(pill);
    }
    const push = document.createElement('button');
    push.type = 'button';
    push.className = 'wt-list-row__push';
    push.title = `git push -u origin ${(branches[i] || '').trim() || 'branch'}`;
    push.textContent = 'Push';
    push.addEventListener('click', () => pushWorktree(path, push));
    row.appendChild(push);
    const rm = document.createElement('button');
    rm.type = 'button';
    rm.className = 'wt-list-row__remove';
    rm.title = 'Remove this worktree (git worktree remove + delete dir)';
    rm.setAttribute('aria-label', 'Remove worktree');
    rm.textContent = '×';
    rm.addEventListener('click', () => removeWorktree(path));
    row.appendChild(rm);
    wtHost.appendChild(row);
  });
}

// Render the worktree card inside the Kernel Lab task view.
function renderKernelWorktrees(meta, statuses) {
  renderWorktreeListInto(
    document.getElementById('kernel-worktree-list'),
    document.getElementById('btn-kernel-worktree-push-all'),
    meta,
    statuses,
    'Primary — kernel runs/agents use this worktree',
  );
}

function renderClaudeInfo(meta, claude, statuses) {
  meta = meta || {};
  claude = claude || {};
  const tmuxEl = $('#claude-info-tmux');
  const pillEl = $('#claude-info-tmux-state');
  const sessHost = $('#claude-info-sessions');
  renderWorktreeListInto(
    $('#claude-info-worktree-list'),
    document.getElementById('btn-worktree-push-all'),
    meta,
    statuses,
    'Primary — the Claude pane opens in this worktree',
  );
  loadMonitor();
  if (tmuxEl) tmuxEl.textContent = claude.tmux_target || meta.tmux_interview_target || '(not started)';
  if (pillEl) {
    const alive = !!claude.tmux_alive;
    pillEl.textContent = alive ? 'alive' : 'down';
    pillEl.dataset.state = alive ? 'alive' : 'down';
  }
  if (!sessHost) return;
  sessHost.innerHTML = '';
  const sessions = Array.isArray(claude.sessions) ? claude.sessions : [];
  if (!sessions.length) {
    const span = document.createElement('span');
    span.className = 'claude-info__hint';
    span.textContent = 'No Claude sessions captured yet. Start Claude to bind one.';
    sessHost.appendChild(span);
    return;
  }
  for (const s of sessions) {
    const row = document.createElement('div');
    row.className = 'claude-session';
    row.dataset.sessionId = s.id || '';
    const idEl = document.createElement('code');
    idEl.className = 'claude-session__id';
    idEl.title = s.id || '';
    idEl.textContent = shortSessionId(s.id);
    const meta_ = document.createElement('span');
    meta_.className = 'claude-session__meta';
    const parts = [];
    if (s.mtime) parts.push(formatSessionMtime(s.mtime));
    if (s.size) parts.push(`${Math.max(1, Math.round(s.size / 1024))} KB`);
    if (!s.path) parts.push('on-disk file not found');
    meta_.textContent = parts.join(' · ') || '(no on-disk transcript yet)';
    const resume = document.createElement('button');
    resume.type = 'button';
    resume.className = 'btn btn--sm';
    resume.textContent = 'Resume';
    resume.disabled = !s.id || (claude.agent_running === true);
    resume.title = claude.agent_running
      ? `Stop the running pane command (${claude.pane_command || 'agent'}) before resuming.`
      : 'Resume in a fresh or idle tmux pane.';
    resume.addEventListener('click', () => resumeClaudeSession(s.id));
    row.appendChild(idEl);
    row.appendChild(meta_);
    row.appendChild(resume);
    sessHost.appendChild(row);
  }
}

// ===== Worktree picker modal =====

async function openWorktreeModal() {
  if (!STATE.slug) return;
  const modal = $('#worktree-modal');
  if (!modal) return;
  modal.hidden = false;
  $('#wt-modal-branch').textContent = `zhongzhu/${STATE.slug}`;
  $('#wt-modal-dest').textContent = `.RUD/${STATE.slug}/work/<repo>/`;
  const host = $('#wt-candidates');
  const status = $('#wt-status');
  status.textContent = '';
  host.innerHTML = '<div class="claude-info__hint">Scanning project root for git repos…</div>';
  try {
    const d = await api(`/api/tasks/${encodeURIComponent(STATE.slug)}/worktree-candidates`);
    renderWorktreeCandidates(d.candidates || [], d.projectRoot || '');
  } catch (err) {
    host.innerHTML = `<div class="status-bad">${escapeHtml(err.message || 'failed')}</div>`;
  }
}

function renderWorktreeCandidates(candidates, projectRoot) {
  const host = $('#wt-candidates');
  host.innerHTML = '';
  if (!candidates.length) {
    const help = document.createElement('div');
    help.className = 'claude-info__hint';
    help.innerHTML = `No git repos found at <code>${escapeHtml(projectRoot)}</code> or its immediate subdirectories. <br>Either register a git-repo path as the project, or <code>git worktree add</code> manually into <code>.RUD/${escapeHtml(STATE.slug)}/work/</code> and reopen the Claude tab.`;
    host.appendChild(help);
    return;
  }
  for (const c of candidates) {
    const row = document.createElement('div');
    row.className = 'wt-candidate' + (c.already_created ? ' wt-candidate--done' : '');
    const info = document.createElement('div');
    info.className = 'wt-candidate__info';
    const name = document.createElement('div');
    name.className = 'wt-candidate__name';
    name.innerHTML = `<strong>${escapeHtml(c.name)}</strong> <span class="wt-candidate__kind">${escapeHtml(c.kind)}</span>`;
    if (c.already_created) {
      name.innerHTML += ' <span class="wt-candidate__kind wt-candidate__kind--done">already added</span>';
    }
    info.appendChild(name);
    const src = document.createElement('div');
    src.className = 'wt-candidate__path';
    src.innerHTML = `<span>source </span><code>${escapeHtml(c.path)}</code>`;
    info.appendChild(src);
    if (c.destination) {
      const dst = document.createElement('div');
      dst.className = 'wt-candidate__path';
      dst.innerHTML = `<span>landing </span><code>${escapeHtml(c.destination)}</code>`;
      info.appendChild(dst);
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn--primary btn--sm';
    btn.textContent = c.already_created ? 'Added' : 'Create';
    btn.disabled = !!c.already_created;
    if (!c.already_created) {
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        $('#wt-status').textContent = `Creating ${c.name}…`;
        try {
          const r = await api(`/api/tasks/${encodeURIComponent(STATE.slug)}/worktree`, {
            method: 'POST',
            body: JSON.stringify({ source_repo: c.path }),
          });
          if (!r.ok) throw new Error(r.error || 'create failed');
          $('#wt-status').textContent = `Created at ${r.worktree_path}`;
          // Refresh both the Claude info card and the modal candidate list
          // so the user can keep adding more.
          await selectTask(STATE.slug);
          const fresh = await api(`/api/tasks/${encodeURIComponent(STATE.slug)}/worktree-candidates`);
          renderWorktreeCandidates(fresh.candidates || [], fresh.projectRoot || projectRoot);
        } catch (err) {
          $('#wt-status').textContent = err.message || 'create failed';
          btn.disabled = false;
        }
      });
    }
    row.appendChild(info);
    row.appendChild(btn);
    host.appendChild(row);
  }
}

function renderWorktreeStatusBadge(st) {
  const span = document.createElement('span');
  span.className = 'wt-status';
  const parts = [];
  if (st.error) {
    span.classList.add('wt-status--error');
    span.textContent = `error: ${st.error}`;
    return span;
  }
  if (st.clean) {
    span.classList.add('wt-status--clean');
    parts.push('● clean');
  } else {
    span.classList.add('wt-status--dirty');
    const breakdown = [];
    if (st.staged) breakdown.push(`${st.staged} staged`);
    if (st.unstaged) breakdown.push(`${st.unstaged} modified`);
    if (st.untracked) breakdown.push(`${st.untracked} untracked`);
    parts.push(breakdown.join(', ') || `${st.dirty_count} changes`);
  }
  if (st.has_remote) {
    if (st.ahead) parts.push(`↑${st.ahead}`);
    if (st.behind) parts.push(`↓${st.behind}`);
    if (!st.ahead && !st.behind) parts.push('in sync');
  } else {
    parts.push('no remote');
    span.classList.add('wt-status--noremote');
  }
  span.textContent = parts.join(' · ');
  return span;
}

async function pushWorktree(path, btn) {
  if (!STATE.slug || !path) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Pushing…';
  try {
    const r = await api(
      `/api/tasks/${encodeURIComponent(STATE.slug)}/worktree/push`,
      { method: 'POST', body: JSON.stringify({ path }) },
    );
    if (!r.ok) throw new Error(r.error || r.message || 'push failed');
    btn.textContent = 'Pushed';
    setTimeout(() => {
      btn.textContent = original;
      btn.disabled = false;
    }, 1500);
    // Refresh the info card so the ahead/behind badge updates.
    await refreshTaskTemplates();
  } catch (err) {
    btn.textContent = original;
    btn.disabled = false;
    alert(err.message || 'push failed');
  }
}

async function pushAllWorktrees() {
  if (!STATE.slug) return;
  const btn = document.getElementById('btn-worktree-push-all');
  if (!btn) return;
  if (!confirm('Push all worktree branches to origin?')) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Pushing all…';
  try {
    const r = await api(
      `/api/tasks/${encodeURIComponent(STATE.slug)}/worktrees/push-all`,
      { method: 'POST', body: '{}' },
    );
    const lines = (r.results || []).map((row) => {
      const tag = row.ok ? 'ok' : 'failed';
      return `${tag}: ${row.path} → ${row.branch || '(no branch)'}\n  ${row.message || row.error || ''}`;
    });
    alert(`Pushed ${r.results.filter((x) => x.ok).length}/${r.count}\n\n${lines.join('\n\n')}`);
    await refreshTaskTemplates();
  } catch (err) {
    alert(err.message || 'push-all failed');
  } finally {
    btn.textContent = original;
    btn.disabled = false;
  }
}

async function removeWorktree(path) {
  if (!STATE.slug || !path) return;
  if (!confirm(`Remove worktree?\n\n${path}\n\nRuns "git worktree remove" and deletes the directory.`)) return;
  try {
    await api(
      `/api/tasks/${encodeURIComponent(STATE.slug)}/worktree?path=${encodeURIComponent(path)}`,
      { method: 'DELETE' },
    );
    await selectTask(STATE.slug);
  } catch (err) {
    alert(err.message || 'remove failed');
  }
}

function closeWorktreeModal() {
  const m = $('#worktree-modal');
  if (m) m.hidden = true;
}

document.getElementById('btn-create-worktree').addEventListener('click', openWorktreeModal);
document.getElementById('btn-wt-close').addEventListener('click', closeWorktreeModal);
document.getElementById('btn-wt-cancel').addEventListener('click', closeWorktreeModal);
$('#worktree-modal').addEventListener('click', (event) => {
  if (event.target.id === 'worktree-modal') closeWorktreeModal();
});

async function refreshClaudeSessions() {
  if (!STATE.slug) return;
  if (STATE.pollInFlight.sessions) return;
  const slug = STATE.slug;
  STATE.pollInFlight.sessions = true;
  try {
    const d = await apiWithRetry(
      '/api/tasks/' + encodeURIComponent(slug) + '/claude-sessions',
      {},
      { attempts: 2, delayMs: 300 },
    );
    if (STATE.slug !== slug) return;
    renderClaudeInfo(STATE.currentMeta || {}, d, STATE.worktreeStatuses || []);
  } catch (err) {
    console.debug('refreshClaudeSessions failed', err);
  } finally {
    STATE.pollInFlight.sessions = false;
  }
}

async function resumeClaudeSession(sessionId) {
  if (!STATE.slug || !sessionId) return;
  if (!confirm(`Resume Claude session ${sessionId} in a fresh or idle tmux pane?`)) return;
  try {
    const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/claude/resume', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!r.ok) throw new Error(r.error || 'resume failed');
    $('#inp-interview-target').value = r.target || '';
    setTmuxOutputText(`Resuming Claude session ${sessionId}\nNew tmux target: ${r.target || '(pending)'}`);
    await refreshInterviewPreview(true);
    await refreshClaudeSessions();
  } catch (err) {
    alert(err.message || 'resume failed');
  }
}

document.getElementById('btn-interview-start').addEventListener('click', startInterviewPane);
document.getElementById('btn-interview-paste').addEventListener('click', pasteInterviewPrompt);
document.getElementById('btn-run-goal').addEventListener('click', runGoalFromPlan);
document.getElementById('btn-write-result').addEventListener('click', writeResultToPlan);
document.getElementById('btn-changes-refresh').addEventListener('click', () => refreshChangesView(true));

(() => {
  const toggle = document.getElementById('monitor-toggle');
  const input = document.getElementById('monitor-pattern');
  if (toggle) toggle.addEventListener('change', () => setMonitor(toggle.checked));
  if (input) {
    input.addEventListener('change', () => {
      if (toggle && toggle.checked) setMonitor(true);
    });
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); input.blur(); if (toggle) { toggle.checked = true; } setMonitor(true); }
    });
  }
})();

document.getElementById('btn-interview-stop').addEventListener('click', async () => {
  if (!STATE.slug) return;
  if (!confirm('Stop deep-interview? This will kill the associated tmux session.')) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/interview/stop', {
    method: 'POST',
    body: '{}',
  });
  $('#inp-interview-target').value = '';
  $('#interview-target-label').textContent = 'Not started';
  setTmuxOutputText(`Stopped ${r.tmux_session || ''}\n${r.tmux_message || ''}`);
});

document.getElementById('btn-interview-send').addEventListener('click', () => sendPaneText(true));

document.getElementById('interview-in').addEventListener('input', () => {
  savePaneDraftForTask();
});

document.querySelectorAll('.pane-actions [data-key]').forEach((btn) => {
  btn.addEventListener('click', () => sendPaneKey(btn.dataset.key));
});

document.getElementById('btn-delete-task').addEventListener('click', deleteSelectedTask);

document.getElementById('btn-new-task').addEventListener('click', async () => {
  const title = $('#new-title').value.trim();
  const general_goal = $('#new-goal').value.trim();
  const skillsEl = document.getElementById('new-skills');
  const skills_path = skillsEl ? skillsEl.value.trim() : '';
  const agentSel = document.getElementById('new-agent-select');
  const agent = agentSel ? agentSel.value : 'claude';
  const btn = $('#btn-new-task');
  const status = $('#new-task-status');
  if (!title || !general_goal) {
    status.textContent = 'Title and general goal are required.';
    return;
  }
  btn.disabled = true;
  status.textContent = 'Creating…';
  try {
    const isKernel = agent === 'kernel';
    const body = { title, general_goal, agent: isKernel ? 'claude' : agent };
    if (isKernel) body.kind = 'kernel';
    if (skills_path) body.skills_path = skills_path;
    const { meta } = await api('/api/tasks', { method: 'POST', body: JSON.stringify(body) });
    resetCreateForm();
    closeCreateModal();
    await loadTasks();
    await selectTask(meta.slug);
  } catch (e) {
    status.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
});

(async function init() {
  buildTabs();
  initMarkdownPreviews();
  initFullscreenPreviews();
  try {
    await loadProjectsList();
    await loadProject();
    await loadTmuxSessions();
    await loadTasks();
  } catch (e) {
    console.error(e);
    alert(e.message);
  }
})();
