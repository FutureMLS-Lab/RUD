/**
 * loom web client.
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
function taskBackendLabel(meta) {
  meta = meta || {};
  const base = `${agentLabel(meta.agent)}${meta.interview_model ? ' · ' + meta.interview_model : ''}`;
  return meta.kind === 'aris' ? `ARIS · ${base}` : base;
}

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
  // Per-task unsent text in the 中文/compose box (same client-only rationale).
  composeDrafts: {},
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
  // True while a monitor enable/disable request is in flight, so the 4s
  // poll's loadMonitor() doesn't reset the toggle the user just clicked.
  monitorBusy: false,
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
    // The terminal may have fit to a default width while this tab was hidden;
    // now that it's visible, refit so Claude fills the full screen width (and
    // the tmux pane resizes to match).
    termHandleResize();
    deferIdle(refreshInterviewPreview);
    deferIdle(refreshTaskTemplates);
    deferIdle(refreshClaudeSessions);
    deferIdle(loadMonitor);
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
  const addBtn = document.getElementById('btn-add-project');
  if (addBtn) addBtn.title = STATE.launchRoot ? `Add a folder inside ${STATE.launchRoot}` : 'Add a folder';
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
  const repoEl = document.getElementById('new-project-repo');
  if (repoEl) repoEl.value = '';
  try {
    await loadProjectsList();
  } catch (e) {
    $('#add-project-status').textContent = e.message;
  }
  const titleEl = $('#add-project-modal-title');
  if (titleEl) titleEl.textContent = STATE.launchRoot ? `Add a folder in ${STATE.launchRoot}` : 'Add a folder';
  renderAddProjectChips();
  setAddProjectMode('existing');
  requestAnimationFrame(() => $('#new-project-path').focus());
}

function setAddProjectMode(mode) {
  STATE.addProjectMode = mode;
  document.querySelectorAll('#add-project-modes .add-project-mode').forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  const repoRow = document.getElementById('add-project-repo-row');
  const label = document.getElementById('new-project-path-label');
  const hint = document.getElementById('add-project-path-hint');
  const pathInput = document.getElementById('new-project-path');
  const saveBtn = document.getElementById('btn-add-project-save');
  const root = (STATE.launchRoot || '').trim();
  const prefix = root ? root.replace(/\/+$/, '') + '/' : '';
  if (repoRow) repoRow.hidden = mode !== 'clone';
  if (mode === 'empty') {
    if (label) label.textContent = 'New folder path';
    if (pathInput) pathInput.placeholder = prefix ? prefix + 'my-new-project' : '/path/inside/launch-dir';
    if (hint) hint.textContent = root ? `Creates the folder inside ${root} and registers it.` : 'Creates the folder and registers it.';
    if (saveBtn) saveBtn.textContent = 'Create & add';
    if (pathInput && prefix && !pathInput.value.trim()) pathInput.value = prefix;
  } else if (mode === 'clone') {
    if (label) label.textContent = 'Clone into (destination folder)';
    if (pathInput) pathInput.placeholder = prefix ? prefix + 'repo' : '/path/inside/launch-dir';
    if (hint) hint.textContent = root ? `git clone the URL into this folder inside ${root}, then register it.` : 'git clone the URL into this folder, then register it.';
    if (saveBtn) saveBtn.textContent = 'Clone & add';
    if (pathInput && prefix && !pathInput.value.trim()) pathInput.value = prefix;
  } else {
    if (label) label.textContent = 'Project directory';
    if (pathInput) pathInput.placeholder = prefix ? prefix + 'my-repo' : '/home/you/MyRepo';
    if (hint) hint.textContent = 'Register an existing folder (any absolute path).';
    if (saveBtn) saveBtn.textContent = 'Add';
  }
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
  const mode = STATE.addProjectMode || 'existing';
  const path = $('#new-project-path').value.trim();
  const repoEl = document.getElementById('new-project-repo');
  const repo_url = repoEl ? repoEl.value.trim() : '';
  const status = $('#add-project-status');
  const btn = $('#btn-add-project-save');
  if (mode === 'clone' && !repo_url) {
    status.textContent = 'Enter a git repo URL to clone.';
    return;
  }
  if (!path) {
    status.textContent = mode === 'existing' ? 'Enter a directory path.' : 'Enter a destination folder path.';
    return;
  }
  btn.disabled = true;
  status.textContent = mode === 'clone'
    ? 'Cloning… (large repos can take a while)'
    : (mode === 'empty' ? 'Creating…' : 'Adding…');
  try {
    const created = await apiNoProject('/api/projects', {
      method: 'POST',
      body: JSON.stringify({ path, mode, repo_url }),
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
  if (!ul) return;
  ul.innerHTML = '';
  try {
    if (!STATE.projectId) {
      ul.innerHTML = '<li class="task-list__empty">Select a project to list loom tmux sessions for that root.</li>';
      return;
    }
    const q = `?project=${encodeURIComponent(STATE.projectId)}`;
    const d = await apiNoProject(`/api/tmux/sessions${q}`);
    const list = d.sessions || [];
    if (!list.length) {
      ul.innerHTML = '<li class="task-list__empty">No loom tmux sessions for this project (or tmux not installed).</li>';
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
    $('#task-backend').textContent = taskBackendLabel(cached);
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
  $('#task-backend').textContent = taskBackendLabel(d.meta);
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
    disconnectTerminal();
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
  if (pasteBtn) pasteBtn.textContent = (meta && meta.kind === 'aris') ? 'Start ARIS loop' : 'Start Deep Interview';
  if (stopBtn) stopBtn.textContent = `Stop ${label}`;
  const heading = document.querySelector('.tab-panel[data-panel="claude"] .terminal-card__bar h4');
  if (heading) heading.textContent = `${label} Terminal`;
  if (!TERM.connected && !termTarget()) {
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
  const key = paneDraftKey(slug);
  if (!key) return;
  const input = document.getElementById('interview-in');
  if (input) STATE.paneDrafts[key] = input.value;
  const compose = document.getElementById('term-compose-input');
  if (compose) STATE.composeDrafts[key] = compose.value;
}

function restorePaneDraftForTask(slug = STATE.slug) {
  const key = paneDraftKey(slug);
  const input = document.getElementById('interview-in');
  if (input) input.value = key ? (STATE.paneDrafts[key] || '') : '';
  const compose = document.getElementById('term-compose-input');
  if (compose) {
    compose.value = key ? (STATE.composeDrafts[key] || '') : '';
    // Resize the auto-growing textarea to fit the restored draft.
    compose.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

function clearPaneDraftForTask(slug = STATE.slug) {
  const key = paneDraftKey(slug);
  if (!key) return;
  STATE.paneDrafts[key] = '';
  STATE.composeDrafts[key] = '';
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
function setTmuxOutputText(text) {
  // Status messages only show when no live PTY stream is attached; once the
  // terminal is streaming, the real output owns the screen.
  const term = ensureTerminal();
  if (!term || TERM.connected) return;
  try {
    term.reset();
    term.write(String(text == null ? '' : text).replace(/\r?\n/g, '\r\n'));
  } catch (e) {}
}

function scrollTmuxOutputToBottom() { /* xterm.js auto-scrolls on write */ }

function revealInterviewTerminal(block = 'center') {
  const card = document.querySelector('.terminal-card--interview') || document.getElementById('interview-out');
  if (!card) return;
  requestAnimationFrame(() => {
    card.scrollIntoView({ block, inline: 'nearest', behavior: 'smooth' });
  });
  // Showing the terminal can change its available width (tab becomes visible,
  // sidebar drawer closes); refit so it matches the now-visible size.
  setTimeout(() => termHandleResize(), 80);
}

async function refreshInterviewPreview() {
  // With xterm + a live PTY stream there's nothing to poll: just ensure the
  // terminal is attached to the current pane (idempotent; reconnects if dropped).
  const target = termTarget();
  if (!target) { disconnectTerminal(); return; }
  connectTerminal(target, false);
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
    const baseKind = wt.base_kind || '';
    const base = wt.base
      ? (' · ' + (baseKind === 'fork' ? 'from ' : 'base ') + escapeHtml(wt.base))
      : '';
    listParts.push('<div class="changes-wt">');
    listParts.push(
      '<div class="changes-wt__head" title="' + escapeHtml(wt.path) + '">' +
        '<span class="changes-wt__name">' + escapeHtml(wtName) + '</span>' +
        '<span class="changes-wt__branch">' + escapeHtml(branch) + base + '</span>' +
        (wt.base
          ? '<button type="button" class="changes-wt__merge" data-path="' + escapeHtml(wt.path) +
            '" title="Merge this branch into the source repo base branch (no push)">Merge ↩</button>'
          : '') +
        '<button type="button" class="changes-wt__review" data-path="' + escapeHtml(wt.path) +
          '" title="AI review (Bugbot-style) of this worktree diff vs your rules / the task skills">Review ⚖</button>' +
      '</div>');
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
  body.querySelectorAll('.changes-wt__merge').forEach((btn) => {
    btn.addEventListener('click', () => mergeWorktree(btn.dataset.path, btn));
  });
  body.querySelectorAll('.changes-wt__review').forEach((btn) => {
    btn.addEventListener('click', () => reviewWorktree(btn.dataset.path, btn));
  });
  renderChangesDiffPanel();
}

async function reviewWorktree(path, btn) {
  if (!STATE.slug || !path) return;
  const rulesEl = document.getElementById('changes-rules');
  const rules = rulesEl ? rulesEl.value : '';
  const host = document.getElementById('changes-diff');
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Reviewing…';
  if (host) host.innerHTML = '<div class="changes-empty">Running review… (calls claude -p; ~10–60s)</div>';
  try {
    const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/review', {
      method: 'POST',
      body: JSON.stringify({ path, rules }),
    });
    if (!r.ok) throw new Error(r.error || 'review failed');
    if (host) {
      host.innerHTML =
        '<div class="changes-diff__head">' +
          '<span class="changes-diff__path">Review — ' + escapeHtml(changesBaseName(path)) + '</span>' +
          '<span class="changes-diff__scope">claude</span>' +
        '</div>' +
        '<div class="markdown-preview changes-review">' + renderMarkdown(r.review || '') + '</div>';
    }
  } catch (err) {
    if (host) host.innerHTML = '<div class="changes-empty">Review failed: ' + escapeHtml(err.message) + '</div>';
  } finally {
    btn.textContent = original;
    btn.disabled = false;
  }
}

async function mergeWorktree(path, btn) {
  if (!STATE.slug || !path) return;
  if (!confirm(
    "Merge this worktree's branch into the source repo's base branch?\n\n" +
    'Runs "git merge --no-ff" in the source repo: it refuses if that repo has ' +
    'uncommitted changes, aborts on conflicts, and does NOT push. Only commits ' +
    'already on the branch are merged.'
  )) return;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Merging…';
  try {
    const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worktree/merge', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    if (!r.ok) {
      const conflicts = (r.conflicts && r.conflicts.length)
        ? '\n\nConflicts:\n- ' + r.conflicts.join('\n- ')
        : '';
      alert((r.error || 'merge failed') + conflicts);
      btn.textContent = original;
      btn.disabled = false;
      return;
    }
    btn.textContent = 'Merged \u2713';
    refreshChangesView(true);
    setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1500);
  } catch (err) {
    alert(err.message || 'merge failed');
    btn.textContent = original;
    btn.disabled = false;
  }
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
  const toggle = document.getElementById('monitor-toggle');
  if (!toggle) return;
  // The toggle itself shows on/off state - no extra status text.
  toggle.checked = !!(d && d.running);
}

async function loadMonitor() {
  if (!STATE.slug) return;
  const slug = STATE.slug;
  try {
    const d = await api('/api/tasks/' + encodeURIComponent(slug) + '/monitor');
    // Don't clobber the toggle/input while the user is mid-action or has
    // switched tasks - the in-flight setMonitor() is the source of truth.
    if (STATE.slug !== slug || STATE.monitorBusy) return;
    applyMonitorState(d);
  } catch (err) {
    console.debug('loadMonitor failed', err);
  }
}

async function setMonitor(enabled) {
  if (!STATE.slug) return;
  const slug = STATE.slug;
  STATE.monitorBusy = true;
  try {
    let d;
    if (enabled) {
      // No pattern from the UI - the backend uses its hardcoded default.
      d = await api('/api/tasks/' + encodeURIComponent(slug) + '/monitor', {
        method: 'POST',
        body: JSON.stringify({}),
      });
    } else {
      d = await api('/api/tasks/' + encodeURIComponent(slug) + '/monitor', { method: 'DELETE' });
    }
    if (STATE.slug !== slug) return;
    applyMonitorState(d);
  } catch (err) {
    console.debug('setMonitor failed', err);
    STATE.monitorBusy = false;
    loadMonitor();
  } finally {
    STATE.monitorBusy = false;
  }
}

// ===== Real terminal (xterm.js) bound to the tmux pane via a live PTY stream.
// Output: GET /api/tmux/stream chunks the actual terminal bytes (a `tmux attach`
// PTY), rendered by xterm with true colors/cursor/TUI. Input: xterm.onData ->
// /api/tmux/send-literal sends the raw byte sequences, so typing, arrows, Enter,
// Esc, Ctrl-C, paste and IME all reach Claude exactly like a real terminal. =====
const TERM = {
  term: null,
  fit: null,
  target: '',
  abort: null,
  connected: false,
  inputQueue: '',
  inputSending: false,
  resizeTimer: null,
  lastSelection: '',
};

function termTarget() {
  const el = document.getElementById('inp-interview-target');
  return el ? el.value.trim() : '';
}

// --- clipboard (works over plain http:// where navigator.clipboard is blocked) -
function termClipboardWrite(text) {
  if (!text) return;
  if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text)
      .then(() => termFlashCopied())
      .catch(() => { if (termExecCopy(text)) termFlashCopied(); });
  } else {
    if (termExecCopy(text)) termFlashCopied();
  }
}

function termExecCopy(text) {
  let ok = false;
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    // Must be IN the viewport (not off-screen) or execCommand('copy') is a
    // silent no-op in several browsers. Keep it invisible but on-screen.
    ta.style.cssText =
      'position:fixed;left:0;top:0;width:2em;height:2em;padding:0;border:0;' +
      'margin:0;opacity:0;background:transparent;z-index:-1;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { ta.setSelectionRange(0, text.length); } catch (e) {}
    ok = document.execCommand('copy');
    document.body.removeChild(ta);
  } catch (e) { console.debug('copy failed', e); ok = false; }
  if (TERM.term) { try { TERM.term.focus(); } catch (e) {} }
  return ok;
}

function termCopySelection() {
  if (!TERM.term) return;
  // The live stream can clear xterm's selection on redraw, so fall back to the
  // last non-empty selection we saw.
  const sel = TERM.term.getSelection() || TERM.lastSelection || '';
  if (sel) termClipboardWrite(sel);
}

// Terminal-style copy/paste shortcuts. Returns false to stop xterm from also
// sending the key to the pane.
function termKeyEvent(e, term) {
  if (e.type !== 'keydown') return true;
  const k = (e.key || '').toLowerCase();
  // Copy: Cmd+C (mac), Ctrl+Shift+C, or Ctrl+C while text is selected.
  if ((e.metaKey && k === 'c')
      || (e.ctrlKey && e.shiftKey && k === 'c')
      || (e.ctrlKey && !e.shiftKey && k === 'c' && term.hasSelection())) {
    termCopySelection();
    return false;
  }
  // Paste: only suppress xterm's ^V on Ctrl+V (Linux/Win). Cmd+V (mac) makes
  // xterm send nothing, and intercepting it here swallows the browser's native
  // 'paste' event - so let Cmd+V through; the 'paste' listener does the paste.
  if (e.ctrlKey && !e.metaKey && k === 'v') {
    return false;
  }
  return true;
}

function ensureTerminal() {
  if (TERM.term) return TERM.term;
  const host = document.getElementById('interview-term');
  if (!host || typeof Terminal === 'undefined') return null;
  const term = new Terminal({
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace',
    fontSize: 13,
    lineHeight: 1.15,
    cursorBlink: true,
    cursorInactiveStyle: 'outline',
    macOptionIsMeta: true,
    scrollback: 8000,
    theme: {
      // Warm-dark: Claude's TUI assumes a dark terminal, so white/bright text
      // needs a dark background to stay readable.
      background: '#211d1a', foreground: '#e7ddcf',
      cursor: '#f59e0b', cursorAccent: '#211d1a',
      selectionBackground: 'rgba(245,158,11,0.30)',
      black: '#2b2620', red: '#e06c5a', green: '#9ec46a', yellow: '#e0af68',
      blue: '#7aa2f7', magenta: '#c79bf0', cyan: '#79c7c7', white: '#d8cfc2',
      brightBlack: '#7a6f60', brightRed: '#f08a7a', brightGreen: '#b6d98a', brightYellow: '#f0c987',
      brightBlue: '#9bb8fa', brightMagenta: '#d4b3f5', brightCyan: '#9bd9d9', brightWhite: '#fdf6ea',
    },
  });
  let fit = null;
  try { fit = new FitAddon.FitAddon(); term.loadAddon(fit); } catch (err) { console.debug('fit addon missing', err); }
  term.open(host);
  try { if (fit) fit.fit(); } catch (e) {}
  term.onData((data) => termQueueInput(data));
  term.attachCustomKeyEventHandler((e) => termKeyEvent(e, term));
  term.onSelectionChange(() => {
    const s = term.getSelection();
    if (s) TERM.lastSelection = s;
  });
  // Copy-on-select (tmux / iTerm style): copy the moment a selection finishes.
  // This is the only reliable copy path here — the live PTY stream redraws and
  // clears xterm's selection before a later Cmd/Ctrl+C keydown can read it, so
  // copying at mouseup time is what actually works. Shortcuts below stay as a
  // fallback for keyboard-only selection.
  host.addEventListener('mouseup', () => {
    const s = term.getSelection();
    if (s && s.trim()) { TERM.lastSelection = s; termClipboardWrite(s); }
  });
  // Native paste: clipboardData works even over plain http:// where the async
  // navigator.clipboard API is blocked. Capture phase so we beat xterm's own
  // paste handler and avoid a double paste.
  if (term.textarea) {
    term.textarea.addEventListener('paste', (e) => {
      const cd = e.clipboardData || window.clipboardData;
      if (!cd) return;
      const t = cd.getData('text');
      if (t) { e.preventDefault(); e.stopImmediatePropagation(); termPaste(t); }
    }, true);
  }
  // Mouse/touchpad wheel -> scroll tmux's own scrollback (copy-mode), instead of
  // letting xterm convert the wheel into arrow keys for the full-screen app.
  // Scrolling back down to the bottom auto-exits copy-mode (live output resumes).
  // Negative dy = up (older output), positive dy = down. Coalesced so a flick
  // doesn't fire dozens of requests.
  let scrollAccum = 0;
  let scrollTimer = null;
  function termScrollBy(dy) {
    if (!TERM.connected || !termTarget()) return;
    scrollAccum += dy;
    if (scrollTimer) return;
    scrollTimer = setTimeout(() => {
      const total = scrollAccum; scrollAccum = 0; scrollTimer = null;
      if (!total) return;
      const lines = Math.max(1, Math.min(80, Math.round(Math.abs(total) / 24)));
      api('/api/tmux/scroll', {
        method: 'POST',
        body: JSON.stringify({ target: termTarget(), dir: total < 0 ? 'up' : 'down', lines }),
      }).catch(() => {});
    }, 40);
  }
  host.addEventListener('wheel', (e) => {
    if (!TERM.connected || !termTarget()) return;
    e.preventDefault();
    e.stopPropagation();
    termScrollBy(e.deltaMode === 1 ? e.deltaY * 18 : e.deltaY);
  }, { passive: false, capture: true });
  // Touch (phones/tablets): a single-finger drag scrolls tmux scrollback the
  // same way the wheel does. A long-press (finger held still) instead arms text
  // selection, where the following drag is translated into xterm's own mouse
  // selection so copy-on-select still fires. Two-finger gestures are left to the
  // browser. Without this, touch drags did nothing (xterm only listens for
  // mouse/wheel), so the pane couldn't be scrolled or selected on mobile.
  const screenEl = () => (term.element && term.element.querySelector('.xterm-screen')) || term.element;
  function synthMouse(type, t) {
    try {
      const el = screenEl();
      if (!el) return;
      el.dispatchEvent(new MouseEvent(type, {
        bubbles: true, cancelable: true, view: window,
        clientX: t.clientX, clientY: t.clientY,
        button: 0, buttons: type === 'mouseup' ? 0 : 1,
      }));
    } catch (_) {}
  }
  const LONG_PRESS_MS = 380;
  const MOVE_SLOP = 10; // px before a press is treated as a drag (scroll)
  let touchMode = '';   // '' | 'scroll' | 'select'
  let touchStartX = 0, touchStartY = 0, touchLastY = 0;
  let pressTimer = null;
  function clearPressTimer() { if (pressTimer) { clearTimeout(pressTimer); pressTimer = null; } }
  host.addEventListener('touchstart', (e) => {
    if (!TERM.connected || !termTarget()) return;
    if (e.touches.length !== 1) { clearPressTimer(); touchMode = ''; return; }
    const t = e.touches[0];
    touchStartX = t.clientX; touchStartY = t.clientY; touchLastY = t.clientY;
    touchMode = '';
    clearPressTimer();
    pressTimer = setTimeout(() => {
      pressTimer = null;
      touchMode = 'select';
      try { term.clearSelection(); } catch (_) {}
      synthMouse('mousedown', { clientX: touchStartX, clientY: touchStartY });
      if (navigator.vibrate) { try { navigator.vibrate(15); } catch (_) {} }
    }, LONG_PRESS_MS);
    // preventDefault here stops the browser from panning the page and from
    // synthesizing its own mouse events (which would double up with synthMouse).
    e.preventDefault();
  }, { passive: false });
  host.addEventListener('touchmove', (e) => {
    if (e.touches.length !== 1) return;
    const t = e.touches[0];
    if (touchMode === '') {
      // Decide scroll vs (pending) select based on how far the finger moved.
      if (Math.abs(t.clientY - touchStartY) > MOVE_SLOP
          || Math.abs(t.clientX - touchStartX) > MOVE_SLOP) {
        clearPressTimer();
        touchMode = 'scroll';
      } else {
        return; // still might become a long-press select
      }
    }
    if (touchMode === 'scroll') {
      e.preventDefault();
      // Finger down (clientY grows) reveals older output above -> scroll up.
      termScrollBy(touchLastY - t.clientY);
      touchLastY = t.clientY;
    } else if (touchMode === 'select') {
      e.preventDefault();
      synthMouse('mousemove', t);
    }
  }, { passive: false });
  function endTouch(e) {
    clearPressTimer();
    if (touchMode === 'select') {
      const last = (e.changedTouches && e.changedTouches[0]) || { clientX: touchStartX, clientY: touchLastY };
      synthMouse('mouseup', last);
      // mouseup on the host copies the selection (copy-on-select); nudge it in
      // case the synthetic event didn't bubble to that listener.
      const s = (() => { try { return term.getSelection(); } catch (_) { return ''; } })();
      if (s && s.trim()) { TERM.lastSelection = s; termClipboardWrite(s); }
    } else if (touchMode === '') {
      // A quick tap (no drag, no long-press). We swallowed the synthesized click
      // in touchstart, so focus the terminal here to raise the soft keyboard.
      try { term.focus(); } catch (_) {}
    }
    touchMode = '';
  }
  host.addEventListener('touchend', endTouch, { passive: true });
  host.addEventListener('touchcancel', () => { clearPressTimer(); touchMode = ''; }, { passive: true });
  TERM.term = term;
  TERM.fit = fit;
  const live = document.getElementById('interview-live');
  host.addEventListener('focusin', () => { if (live) live.hidden = false; });
  host.addEventListener('focusout', () => { if (live) live.hidden = true; });
  // Clicking the host padding (outside the xterm screen) should still focus the
  // terminal so keys go to Claude. Guarded to the host itself so click-drag text
  // selection on the screen is left to xterm.
  host.addEventListener('pointerdown', (e) => {
    if (e.target === host) { try { term.focus(); } catch (_) {} }
  });
  try {
    const ro = new ResizeObserver(() => termHandleResize());
    ro.observe(host);
  } catch (e) {}
  window.addEventListener('resize', termHandleResize);
  // Monospace metrics finalize after the web font loads; refit so column count
  // (and the tmux pane width) matches the real glyph width.
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => termHandleResize()).catch(() => {});
  }
  return term;
}

function termHandleResize() {
  if (!TERM.fit || !TERM.term) return;
  if (TERM.resizeTimer) clearTimeout(TERM.resizeTimer);
  TERM.resizeTimer = setTimeout(() => {
    TERM.resizeTimer = null;
    const host = document.getElementById('interview-term');
    // Skip while the terminal isn't really laid out (a tab switch / mobile
    // URL-bar animation / scroll repaint can momentarily report a tiny width),
    // so we never fit to a sliver.
    if (!host || host.clientWidth < 120) return;
    const prevCols = TERM.term.cols;
    const prevRows = TERM.term.rows;
    try { TERM.fit.fit(); } catch (e) {}
    // Reconnect only when the size actually changed, so tmux resizes the pane to
    // match xterm (Claude re-renders to fill the width). Kept deliberately simple
    // — extra "smart" refit passes caused reconnect churn that dropped the client
    // and left the pane stuck at a stale width.
    if (TERM.connected && TERM.target
        && (TERM.term.cols !== prevCols || TERM.term.rows !== prevRows)) {
      connectTerminal(TERM.target, true);
    }
  }, 300);
}

async function termQueueInput(data) {
  TERM.inputQueue += data;
  if (TERM.inputSending) return;
  TERM.inputSending = true;
  try {
    while (TERM.inputQueue) {
      const chunk = TERM.inputQueue;
      TERM.inputQueue = '';
      const target = termTarget();
      if (!target) break;
      try {
        await api('/api/tmux/send-literal', { method: 'POST', body: JSON.stringify({ target, text: chunk }) });
      } catch (err) { console.debug('input send failed', err); }
    }
  } finally {
    TERM.inputSending = false;
  }
}

// Send raw bytes to the pane (used by paste + the mobile key bar). Reuses the
// same batched queue as keystrokes so ordering with live typing is preserved.
function termSendLiteral(text) {
  if (text == null || text === '') return;
  termQueueInput(String(text));
}

// Paste into the pane. A short single line goes through the fast literal path
// (keeps ordering with live typing); anything multi-line or large goes through
// tmux bracketed paste so the TUI (Claude) treats it as ONE paste and doesn't
// submit on every embedded newline (and we avoid the send-literal size cap).
function termPaste(text) {
  if (!text) return;
  const target = termTarget();
  if (!target) return;
  if (!/[\r\n]/.test(text) && text.length <= 2000) {
    termSendLiteral(text);
    return;
  }
  api('/api/tmux/send-text', {
    method: 'POST',
    body: JSON.stringify({ target, text, submit: false }),
  }).catch((err) => console.debug('paste failed', err));
}

// The live PTY stream repaints on its own, so there's nothing to poll here; this
// just nudges the read-only markdown preview after a compose-box message in case
// it changed PLAN.md (or another scanned file).
function termScheduleRefresh() {
  if (termScheduleRefresh._t) clearTimeout(termScheduleRefresh._t);
  termScheduleRefresh._t = setTimeout(() => {
    termScheduleRefresh._t = null;
    try { refreshInterviewPreview(true); } catch (e) {}
  }, 800);
}

// Small transient "已复制" toast shown after a successful clipboard copy.
function termFlashCopied() {
  const card = document.querySelector('.terminal-card--interview');
  if (!card) return;
  let el = card.querySelector('.term-copied-toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'term-copied-toast';
    el.textContent = '已复制';
    card.appendChild(el);
  }
  el.classList.add('is-show');
  if (termFlashCopied._t) clearTimeout(termFlashCopied._t);
  termFlashCopied._t = setTimeout(() => el.classList.remove('is-show'), 900);
}

function disconnectTerminal() {
  if (TERM.abort) { try { TERM.abort.abort(); } catch (e) {} TERM.abort = null; }
  TERM.connected = false;
  TERM.target = '';
}

async function connectTerminal(target, force = false) {
  const term = ensureTerminal();
  if (!term || !target) return;
  if (!force && TERM.connected && TERM.target === target) return;
  // If the host isn't laid out yet (tab just shown / mid-layout), don't fit to a
  // tiny width and pin the pane to ~20 cols. Wait until it has a real width,
  // keeping any existing stream alive in the meantime.
  const hostEl = document.getElementById('interview-term');
  if (hostEl && hostEl.clientWidth < 120) {
    TERM._sizeWaits = (TERM._sizeWaits || 0) + 1;
    if (TERM._sizeWaits <= 50) {
      clearTimeout(TERM._sizeWaitTimer);
      TERM._sizeWaitTimer = setTimeout(() => connectTerminal(target, force), 100);
      return;
    }
  }
  TERM._sizeWaits = 0;
  disconnectTerminal();
  TERM.target = target;
  TERM.connected = true;
  const ctrl = new AbortController();
  TERM.abort = ctrl;
  try { if (TERM.fit) TERM.fit.fit(); } catch (e) {}
  const cols = term.cols || 80;
  const rows = term.rows || 24;
  try { term.reset(); } catch (e) {}
  try {
    const resp = await fetch(
      '/api/tmux/stream?target=' + encodeURIComponent(target) + '&cols=' + cols + '&rows=' + rows,
      { signal: ctrl.signal, cache: 'no-store' },
    );
    if (!resp.ok || !resp.body) {
      if (TERM.abort === ctrl) {
        TERM.connected = false; TERM.abort = null;
        // Session not running (task not started, or the pane was stopped/died).
        // Show a friendly note instead of tmux's raw "can't find session: …".
        setTmuxOutputText('tmux is not alive — click Start Claude to launch the pane.');
      }
      return;
    }
    const reader = resp.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value && TERM.term && TERM.target === target) TERM.term.write(value);
    }
  } catch (err) {
    if (err && err.name !== 'AbortError') console.debug('terminal stream error', err);
  } finally {
    if (TERM.abort === ctrl) { TERM.connected = false; TERM.abort = null; }
  }
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
  aris: 'Autonomous research loop: the agent mines ideas from the base repo, runs a worktree experiment per idea, and folds results back into PLAN.md.',
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
(() => {
  const modesEl = document.getElementById('add-project-modes');
  if (!modesEl) return;
  modesEl.addEventListener('click', (e) => {
    const b = e.target.closest('.add-project-mode');
    if (!b || !b.dataset.mode) return;
    setAddProjectMode(b.dataset.mode);
    const focusEl = b.dataset.mode === 'clone'
      ? document.getElementById('new-project-repo')
      : document.getElementById('new-project-path');
    if (focusEl) focusEl.focus();
  });
})();
$('#add-project-modal').addEventListener('click', (event) => {
  if (event.target.id === 'add-project-modal') closeAddProjectModal();
});

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
  if (!$('#kernel-source-modal').hidden) closeKernelSourceModal();
  else if (!$('#preview-modal').hidden) closeFullscreenPreview();
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

document.getElementById('btn-kernel-source-close').addEventListener('click', closeKernelSourceModal);
document.getElementById('btn-kernel-source-copy').addEventListener('click', copyKernelSource);
document.getElementById('btn-kernel-source-download').addEventListener('click', downloadKernelSource);
document.getElementById('kernel-source-modal').addEventListener('click', (event) => {
  if (event.target.id === 'kernel-source-modal') closeKernelSourceModal();
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
  // Shape is agent-decided now, so don't fill the field. Surface the plugin's
  // template (when known) as a placeholder hint for the optional override.
  const plugin = $('#kernel-plugin').value;
  const tpl = (STATE.kernelShapeTemplates || {})[plugin];
  const el = $('#kernel-shape');
  if (el) el.placeholder = tpl ? JSON.stringify(tpl, null, 2) : '(agent decides — optional override)';
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
  return [r.id, r.state || '', plugin, best, st.agents_running || 0, (st.archive || []).length, r.verified].join('|');
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
    const legacyBadge = r.legacy ? ' <span class="kernel-legacy-badge">legacy</span>' : '';
    li.innerHTML =
      `<div class="kernel-run__head"><span class="kernel-run__plugin">${escapeHtml(cfg.plugin || plugin || '?')} · ${escapeHtml(cfg.target || (r.spec && r.spec.target) || '?')}</span>` +
      `<span class="kernel-run__state" data-state="${escapeHtml(r.state || '')}">${escapeHtml(r.state || '')}</span></div>` +
      `<div class="kernel-run__meta">best ${kernelSpeedupText(best)}${agents}${unvBadge}${legacyBadge}</div>`;
    li.addEventListener('click', () => selectKernelRun(r.id));
    ul.appendChild(li);
  });
}

async function loadKernelRuns() {
  const ul = $('#kernel-run-list');
  if (!ul) return;
  try {
    const d = await api('/api/kernel/runs?task=' + encodeURIComponent(STATE.slug || ''));
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

function kernelFmtUs(v) {
  return (typeof v === 'number' && isFinite(v)) ? v.toFixed(1) : '—';
}

// Inline-SVG speedup-over-time chart from the run's archive (every accepted
// kernel). The line is the running best; dots are individual accepted kernels
// colored per agent. No chart dependency - keeps the vendored stack lean.
function kernelTimelineSvg(entries) {
  if (!entries || !entries.length) return '';
  const W = 600, H = 130, padL = 8, padR = 8, padT = 12, padB = 16;
  const n = entries.length;
  const speeds = entries.map((e) => Number(e.speedup) || 0);
  const maxS = Math.max(...speeds, 1);
  const minS = Math.min(...speeds, 0);
  const xAt = (i) => padL + (n <= 1 ? (W - padL - padR) / 2 : (i / (n - 1)) * (W - padL - padR));
  const yAt = (s) => {
    const t = maxS === minS ? 0.5 : (s - minS) / (maxS - minS);
    return padT + (1 - t) * (H - padT - padB);
  };
  const palette = ['#6366f1', '#14b8a6', '#e0af68', '#e06c5a', '#c79bf0', '#79c7c7', '#9ec46a', '#f08a7a'];
  let bestSoFar = -Infinity;
  const linePts = entries.map((e, i) => {
    bestSoFar = Math.max(bestSoFar, Number(e.speedup) || 0);
    return `${xAt(i).toFixed(1)},${yAt(bestSoFar).toFixed(1)}`;
  });
  const dots = entries.map((e, i) => {
    const c = palette[(Number(e.agent_index) || 0) % palette.length];
    const title = escapeHtml(`agent ${e.agent_index} · ${(Number(e.speedup) || 0).toFixed(2)}× · ${e.achieved_at || ''}`);
    return `<circle cx="${xAt(i).toFixed(1)}" cy="${yAt(Number(e.speedup) || 0).toFixed(1)}" r="3.2" fill="${c}"><title>${title}</title></circle>`;
  }).join('');
  return `<svg class="kernel-timeline__svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="speedup over time">`
    + `<polyline class="kernel-timeline__line" fill="none" stroke="#6366f1" stroke-width="2" points="${linePts.join(' ')}"/>`
    + dots
    + `<text class="kernel-timeline__axis" x="${padL}" y="${padT}">${maxS.toFixed(2)}×</text>`
    + `<text class="kernel-timeline__axis" x="${padL}" y="${H - padB + 12}">${minS.toFixed(2)}×</text>`
    + `</svg>`;
}

function renderKernelRunDetail(r) {
  const host = $('#kernel-run-detail');
  if (!host) return;
  const cfg = r.config || {};
  const st = r.status || {};
  const best = st.best || null;
  const bests = (st.agent_bests || []).slice().sort((a, b) => (b.speedup || 0) - (a.speedup || 0));
  const archive = (st.archive || []).slice().sort((a, b) => String(a.achieved_at).localeCompare(String(b.achieved_at)));
  const plugin = r.plugin || cfg.plugin || '';
  const target = cfg.target || (r.spec && r.spec.target) || '?';
  const unverified = !!plugin && ((STATE.kernelUnverified || []).includes(plugin) || r.verified === false);
  const vbadge = plugin ? (unverified ? ' <span class="kernel-unverified">⚠ unverified reference</span>' : ' <span class="kernel-verified">✓ verified</span>') : '';
  const legacy = r.legacy ? ' <span class="kernel-legacy-badge">legacy</span>' : '';
  const targetSpeedup = (st.target_speedup != null ? st.target_speedup : cfg.target_speedup) || null;
  const bestSpeed = best && typeof best.speedup === 'number' ? best.speedup : null;
  const shapeText = cfg.shape == null
    ? (cfg.shape_source === 'template' ? 'plugin default' : 'agent-decided')
    : (typeof cfg.shape === 'string' ? cfg.shape : JSON.stringify(cfg.shape));
  const shapeSrc = cfg.shape_source ? ` <span class="kernel-shape-src">(${escapeHtml(cfg.shape_source)})</span>` : '';

  let html = `<div class="kernel-detail__head"><h4>${escapeHtml(plugin || '?')} <span class="kernel-detail__target">${escapeHtml(target)}</span>${vbadge}${legacy}</h4>`;
  if (['launching', 'running'].includes(r.state)) {
    html += `<button type="button" class="btn btn--danger btn--sm" id="btn-kernel-stop">Stop</button>`;
  }
  html += '</div>';

  // Status chips
  const agentsRunning = st.agents_running || 0;
  const agentsTotal = (st.agents || []).length;
  html += '<div class="kernel-chips">';
  html += `<span class="kernel-chip kernel-chip--state" data-state="${escapeHtml(r.state || '')}">${escapeHtml(r.state || '—')}</span>`;
  if (agentsTotal || ['launching', 'running'].includes(r.state)) html += `<span class="kernel-chip">${agentsRunning}/${agentsTotal} agents</span>`;
  html += `<span class="kernel-chip">${st.improvements || archive.length || 0} improvements</span>`;
  if (cfg.build_mode) html += '<span class="kernel-chip kernel-chip--mode">build mode</span>';
  html += '</div>';

  // Best-kernel card + target progress
  if (best) {
    html += '<div class="kernel-best">';
    html += '<div class="kernel-best__main">';
    html += `<div class="kernel-best__speed">${bestSpeed != null ? bestSpeed.toFixed(2) + '×' : '—'}</div>`;
    html += `<div class="kernel-best__sub">${kernelFmtUs(best.kernel_us)} µs vs ${kernelFmtUs(best.baseline_us)} µs baseline${best.agent_index != null ? ` · agent ${escapeHtml(String(best.agent_index))}` : ''}</div>`;
    html += '</div>';
    html += '<div class="kernel-best__actions">';
    if (best.job_id) html += `<button type="button" class="btn btn--sm kernel-view-src" data-job="${escapeHtml(best.job_id)}" data-label="best kernel">View source</button>`;
    html += '<button type="button" class="btn btn--sm" id="btn-kernel-best-full">Best kernel ▸</button>';
    html += '</div></div>';
    if (targetSpeedup && bestSpeed != null) {
      const pct = Math.max(0, Math.min(100, (bestSpeed / targetSpeedup) * 100));
      html += `<div class="kernel-progress" title="best ${bestSpeed.toFixed(2)}× of target ${Number(targetSpeedup).toFixed(2)}×">`
        + `<div class="kernel-progress__bar${pct >= 100 ? ' is-done' : ''}" style="width:${pct.toFixed(0)}%"></div>`
        + `<span class="kernel-progress__label">${bestSpeed.toFixed(2)}× / target ${Number(targetSpeedup).toFixed(2)}×</span></div>`;
    }
  } else if (['running', 'launching'].includes(r.state)) {
    html += `<div class="kernel-best kernel-best--empty">${cfg.build_mode ? 'No correct kernel yet…' : 'No improving kernel yet…'}</div>`;
  }

  // Meta grid
  html += '<div class="kernel-detail__grid">';
  html += `<span>Run ID</span><code>${escapeHtml(r.run_id || '—')}</code>`;
  html += `<span>Model</span><span>${escapeHtml(cfg.model || '')}</span>`;
  html += `<span>Shape</span><span><code>${escapeHtml(shapeText)}</code>${shapeSrc}</span>`;
  if (targetSpeedup) html += `<span>Target</span><span>${Number(targetSpeedup).toFixed(2)}×</span>`;
  html += '</div>';

  if (r.state === 'error' && r.error) html += `<p class="status-bad">${escapeHtml(r.error)}</p>`;

  // Per-agent leaderboard
  if (bests.length) {
    html += '<div class="kernel-section__title">Agent leaderboard</div>';
    html += '<table class="kernel-leaderboard"><thead><tr><th>#</th><th>Agent</th><th>Speedup</th><th>kernel µs</th><th>baseline µs</th><th></th></tr></thead><tbody>';
    bests.forEach((e, i) => {
      const src = e.job_id ? `<button type="button" class="kernel-view-src kernel-view-src--link" data-job="${escapeHtml(e.job_id)}" data-label="agent ${escapeHtml(String(e.agent_index))}">source</button>` : '';
      html += `<tr><td>${i + 1}</td><td>${escapeHtml(String(e.agent_index))}</td><td class="kernel-speedup">${(e.speedup || 0).toFixed(2)}×</td><td>${kernelFmtUs(e.kernel_us)}</td><td>${kernelFmtUs(e.baseline_us)}</td><td>${src}</td></tr>`;
    });
    html += '</tbody></table>';
  }

  // Improvement timeline
  if (archive.length) {
    html += '<div class="kernel-section__title">Improvement timeline</div>';
    html += '<div class="kernel-timeline">' + kernelTimelineSvg(archive) + '</div>';
  }

  // Build / run log
  if (['launching', 'running', 'error'].includes(r.state)) {
    const shown = STATE.kernelLogRunId === r.id ? (STATE.kernelLogText || '') : '';
    html += '<details class="kernel-buildlog"' + (r.state === 'error' ? ' open' : '') + '><summary>build / run log</summary>' +
      `<pre id="kernel-build-log">${escapeHtml(shown || '(loading log…)')}</pre></details>`;
  } else {
    const detail = formatKernelErrorDetail(r.error_detail);
    if (detail) html += `<details class="kernel-errdetail"><summary>details</summary><pre>${escapeHtml(detail)}</pre></details>`;
  }

  // Verify
  if (unverified && plugin) {
    html += '<div class="kernel-detail__verify"><button type="button" class="btn btn--sm" id="btn-kernel-verify">Mark reference verified</button>' +
      '<span class="tab-panel__hint"> review reference() (search TODO(review)) before marking</span></div>';
  }

  host.innerHTML = html;
  const stopBtn = $('#btn-kernel-stop');
  if (stopBtn) stopBtn.addEventListener('click', () => stopKernelRun(r.id));
  const verBtn = $('#btn-kernel-verify');
  if (verBtn) verBtn.addEventListener('click', () => markPluginVerified(plugin));
  const bestFull = $('#btn-kernel-best-full');
  if (bestFull) bestFull.addEventListener('click', () => openKernelBestModal(r.id));
  host.querySelectorAll('.kernel-view-src').forEach((btn) => {
    btn.addEventListener('click', () => openKernelSourceModal(r.id, btn.dataset.job, btn.dataset.label || 'kernel'));
  });
}

// ===== Kernel source viewer modal =====

function showKernelSourceModal(title, source, extraTabs) {
  const modal = document.getElementById('kernel-source-modal');
  if (!modal) return;
  const titleEl = document.getElementById('kernel-source-title');
  const pre = document.getElementById('kernel-source-pre');
  const tabsEl = document.getElementById('kernel-source-tabs');
  const tabs = [{ label: 'kernel', source: source || '' }, ...(extraTabs || []).filter((t) => t && t.source)];
  STATE.kernelSourceTabs = tabs;
  if (titleEl) titleEl.textContent = title || 'Kernel source';
  if (tabsEl) {
    if (tabs.length > 1) {
      tabsEl.hidden = false;
      tabsEl.innerHTML = tabs.map((t, i) => `<button type="button" class="kernel-src-tab${i === 0 ? ' is-active' : ''}" data-idx="${i}">${escapeHtml(t.label)}</button>`).join('');
      tabsEl.querySelectorAll('.kernel-src-tab').forEach((b) => b.addEventListener('click', () => {
        const idx = Number(b.dataset.idx) || 0;
        tabsEl.querySelectorAll('.kernel-src-tab').forEach((x) => x.classList.toggle('is-active', x === b));
        if (pre) pre.textContent = tabs[idx].source;
      }));
    } else {
      tabsEl.hidden = true;
      tabsEl.innerHTML = '';
    }
  }
  if (pre) pre.textContent = source || '(empty)';
  modal.hidden = false;
}

function closeKernelSourceModal() {
  const m = document.getElementById('kernel-source-modal');
  if (m) m.hidden = true;
}

async function openKernelSourceModal(runId, jobId, label) {
  if (!jobId) return;
  showKernelSourceModal((label || 'kernel') + ' source', 'Loading…');
  try {
    const d = await api('/api/kernel/runs/' + encodeURIComponent(runId) + '/kernel/' + encodeURIComponent(jobId));
    showKernelSourceModal((label || 'kernel') + ' source', d.ok === false ? (d.error || 'not found') : (d.source || '(empty)'));
  } catch (err) {
    showKernelSourceModal((label || 'kernel') + ' source', err.message || 'failed to load');
  }
}

async function openKernelBestModal(runId) {
  showKernelSourceModal('Best kernel', 'Loading…');
  try {
    const d = await api('/api/kernel/runs/' + encodeURIComponent(runId) + '/best-kernel');
    if (d.ok === false) { showKernelSourceModal('Best kernel', d.error || 'no best kernel yet'); return; }
    const extra = [];
    if (d.postprocessed_source) extra.push({ label: 'postprocessed', source: d.postprocessed_source });
    if (d.python_registration) extra.push({ label: 'registration', source: d.python_registration });
    const head = (typeof d.speedup === 'number' ? d.speedup.toFixed(2) + '× · ' : '') + (d.valid === false ? 'INVALID' : 'valid');
    showKernelSourceModal('Best kernel (' + head + ')', d.kernel_source || '(empty)', extra);
  } catch (err) {
    showKernelSourceModal('Best kernel', err.message || 'failed to load');
  }
}

function copyKernelSource() {
  const pre = document.getElementById('kernel-source-pre');
  if (!pre) return;
  const text = pre.textContent || '';
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).catch(() => {});
  } else {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }
}

function downloadKernelSource() {
  const pre = document.getElementById('kernel-source-pre');
  if (!pre) return;
  const blob = new Blob([pre.textContent || ''], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'kernel.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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
  let cached = (STATE.kernelRuns || []).find((r) => r.id === id) || null;
  if (cached) {
    renderKernelRunDetail(cached);
    STATE.kernelDetailSig = kernelRunSig(cached);
    if (['launching', 'running', 'error'].includes(cached.state)) refreshKernelBuildLog(id);
  } else {
    try {
      cached = await api('/api/kernel/runs/' + encodeURIComponent(id));
      if (STATE.kernelSelected !== id) return;
      renderKernelRunDetail(cached);
      STATE.kernelDetailSig = kernelRunSig(cached);
      if (['launching', 'running', 'error'].includes(cached.state)) refreshKernelBuildLog(id);
    } catch (err) {
      host.innerHTML = `<p class="status-bad">${escapeHtml(err.message)}</p>`;
      return;
    }
  }
  // Pull a fresh leaderboard (best / per-agent / improvement timeline) for any
  // started run - this works for stopped runs too, where the run detail wouldn't
  // re-fetch status on its own.
  if (cached && cached.run_id) {
    try {
      const lb = await api('/api/kernel/runs/' + encodeURIComponent(id) + '/leaderboard');
      if (STATE.kernelSelected !== id) return;
      if (lb && lb.ok !== false) {
        cached.status = lb;
        renderKernelRunDetail(cached);
        STATE.kernelDetailSig = kernelRunSig(cached);
      }
    } catch (e) { /* keep current render */ }
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
  // Shape is optional: blank => the agent proposes one at launch. Only parse and
  // validate when the user typed an explicit override.
  const rawShape = ($('#kernel-shape').value || '').trim();
  let shape = null;
  if (rawShape) {
    try { shape = JSON.parse(rawShape); }
    catch { status.textContent = 'Override shape must be valid JSON (or leave it blank)'; return; }
  }
  const model = $('#kernel-model').value.trim();
  if (!model) { status.textContent = 'Model is required'; return; }
  const buildMode = $('#kernel-build-mode') ? $('#kernel-build-mode').checked : false;
  const body = {
    plugin: $('#kernel-plugin').value,
    target: $('#kernel-target').value,
    model,
    slug: STATE.slug,
    n_agents: Number($('#kernel-nagents').value) || 1,
    starter_mode: $('#kernel-starter').value,
    auto_terminate: buildMode ? true : $('#kernel-auto-terminate').checked,
    build: $('#kernel-build').checked,
    build_mode: buildMode,
  };
  if (shape) body.shape = shape;
  const ts = $('#kernel-target-speedup').value;
  if (ts) body.target_speedup = Number(ts);
  if (buildMode && body.target_speedup === undefined) body.target_speedup = 0;
  status.textContent = shape
    ? 'Launching… (first run builds the image — this can take a few minutes)'
    : 'Launching… the agent is choosing a shape, then building (first run can take a few minutes)';
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
  if (spec.shape && $('#kernel-shape')) {
    // The interview agent inferred a shape — show it in the (optional) override
    // so the user can review or tweak it before launching.
    $('#kernel-shape').value = JSON.stringify(spec.shape, null, 2);
    const adv = document.getElementById('kernel-advanced-shape');
    if (adv) adv.open = true;
  }
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
    const r = await api('/api/kernel/prepare', { method: 'POST', body: JSON.stringify({ spec, slug: STATE.slug }) });
    const id = r.id;
    for (let i = 0; i < 240; i += 1) {
      await sleep(2500);
      const rec = await api('/api/kernel/runs/' + encodeURIComponent(id));
      if (rec.state === 'prepared') {
        await loadKernelPlugins();
        fillFormFromSpec(spec);
        if (rec.plugin && $('#kernel-plugin')) {
          $('#kernel-plugin').value = rec.plugin;
          // Refresh the override placeholder for the resolved plugin. The field
          // stays blank on purpose — the agent picks the shape at launch unless
          // the user types an override.
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
  if (pushAllBtn) pushAllBtn.hidden = worktrees.length <= 1;
  if (!wtHost) return;
  wtHost.innerHTML = '';
  if (!worktrees.length) {
    const hint = document.createElement('span');
    hint.className = 'claude-info__hint';
    hint.textContent = '(none — click + Add worktree, or git worktree add manually under .RUD/<slug>/work/)';
    wtHost.appendChild(hint);
    return;
  }
  // Compact picker: a dropdown to choose a worktree, its live status, and the
  // Push / remove buttons on the right acting on the *selected* worktree.
  const row = document.createElement('div');
  row.className = 'wt-picker';

  const sel = document.createElement('select');
  sel.className = 'wt-select';
  worktrees.forEach((path, i) => {
    const opt = document.createElement('option');
    opt.value = path;
    // Prefer the LIVE branch from `git status` (so a manual `git checkout`
    // inside the worktree is reflected); fall back to the persisted meta
    // branch only when a live status isn't available yet.
    let liveBranch = ((statusByPath[path] && statusByPath[path].branch) || '').trim();
    if (/\(no branch\)/.test(liveBranch) || liveBranch === 'HEAD') liveBranch = '(detached)';
    const br = liveBranch || (branches[i] || '').trim() || '—';
    opt.textContent = `${changesBaseName(path)} · ${br}${i === 0 ? ' · primary' : ''}`;
    opt.title = path;
    sel.appendChild(opt);
  });
  row.appendChild(sel);

  const badgeHost = document.createElement('span');
  badgeHost.className = 'wt-picker__status';
  const renderBadge = () => {
    badgeHost.innerHTML = '';
    const st = statusByPath[sel.value];
    if (st) badgeHost.appendChild(renderWorktreeStatusBadge(st));
  };
  sel.addEventListener('change', renderBadge);
  row.appendChild(badgeHost);

  const push = document.createElement('button');
  push.type = 'button';
  push.className = 'btn btn--sm wt-picker__push';
  push.textContent = 'Push';
  push.title = 'git push -u origin <selected branch>';
  push.addEventListener('click', () => pushWorktree(sel.value, push));
  row.appendChild(push);

  const rm = document.createElement('button');
  rm.type = 'button';
  rm.className = 'wt-list-row__remove';
  rm.title = 'Remove the selected worktree (git worktree remove + delete dir)';
  rm.setAttribute('aria-label', 'Remove selected worktree');
  rm.textContent = '×';
  rm.addEventListener('click', () => removeWorktree(sel.value));
  row.appendChild(rm);

  wtHost.appendChild(row);
  renderBadge();
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
  // Dropdown of sessions (newest first) + a single Resume button on the right,
  // mirroring the worktree picker. Resuming acts on the selected session.
  const running = claude.agent_running === true;
  const row = document.createElement('div');
  row.className = 'session-picker';

  const sel = document.createElement('select');
  sel.className = 'session-select';
  sel.setAttribute('aria-label', 'Claude session to resume');
  sessions.forEach((s) => {
    const opt = document.createElement('option');
    opt.value = s.id || '';
    const bits = [shortSessionId(s.id)];
    if (s.mtime) bits.push(formatSessionMtime(s.mtime));
    if (s.size) bits.push(`${Math.max(1, Math.round(s.size / 1024))} KB`);
    if (!s.path) bits.push('no transcript');
    opt.textContent = bits.join(' · ');
    opt.title = s.id || '';
    if (!s.id) opt.disabled = true;
    sel.appendChild(opt);
  });
  row.appendChild(sel);

  const resume = document.createElement('button');
  resume.type = 'button';
  resume.className = 'btn btn--sm session-picker__resume';
  resume.textContent = 'Resume';
  resume.disabled = running;
  resume.title = running
    ? `Stop the running pane command (${claude.pane_command || 'agent'}) before resuming.`
    : 'Resume the selected session in a fresh or idle tmux pane.';
  resume.addEventListener('click', () => { if (sel.value) resumeClaudeSession(sel.value); });
  row.appendChild(resume);

  sessHost.appendChild(row);
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

// Mobile/touch terminal keys: phone keyboards have no Esc/Ctrl-C/arrows/Tab, so
// this bar sends the raw byte sequences into the pane (send_pane_literal also
// leaves copy-mode first, so it works even after scrolling).
// Compose box: a normal input where the OS IME works reliably (xterm's built-in
// CJK input is buggy - repeats/loses characters). Enter sends the line into the
// pane and clears. English/keys can still be typed straight into the terminal.
(function initTermCompose() {
  const input = document.getElementById('term-compose-input');
  const sendBtn = document.getElementById('term-compose-send');
  const toggle = document.getElementById('term-compose-toggle');
  const box = document.getElementById('term-compose');
  if (!input) return;
  const MAX_H = 168;
  function autoGrow() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, MAX_H) + 'px';
  }
  function setOpen(open) {
    if (!box) return;
    box.hidden = !open;
    if (toggle) toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) { autoGrow(); input.focus(); }
    else if (TERM.term) { try { TERM.term.focus(); } catch (e) {} }
  }
  async function submitCompose() {
    const text = input.value;
    const target = termTarget();
    if (!text.trim() || !target) return;
    input.value = '';
    const key = paneDraftKey();
    if (key) STATE.composeDrafts[key] = '';
    autoGrow();
    try {
      await api('/api/tmux/send-text', {
        method: 'POST',
        body: JSON.stringify({ target, text, submit: true }),
      });
    } catch (err) { console.debug('compose send failed', err); }
    termScheduleRefresh();
    input.focus();
  }
  input.addEventListener('input', autoGrow);
  input.addEventListener('keydown', (e) => {
    // Enter sends; Shift+Enter inserts a newline (default textarea behaviour);
    // the Enter that commits an IME composition has isComposing=true and must
    // be left alone so Chinese input still works.
    if (e.key === 'Enter' && !e.isComposing && !e.shiftKey) {
      e.preventDefault();
      submitCompose();
    }
  });
  if (sendBtn) sendBtn.addEventListener('click', submitCompose);
  // Collapsed by default; the 中文 button reveals the reliable-IME compose box.
  if (toggle) toggle.addEventListener('click', () => setOpen(box && box.hidden));
})();

(function initTermMobileKeys() {
  const bar = document.getElementById('term-mobile-keys');
  if (!bar) return;
  const SEQ = {
    esc: '\x1b', 'c-c': '\x03', tab: '\t', enter: '\r',
    up: '\x1b[A', down: '\x1b[B', right: '\x1b[C', left: '\x1b[D',
  };
  bar.addEventListener('click', (e) => {
    const btn = e.target.closest('.term-key');
    if (!btn) return;
    const seq = SEQ[btn.dataset.key];
    if (seq != null) termSendLiteral(seq);
  });
})();

(() => {
  const toggle = document.getElementById('monitor-toggle');
  if (toggle) toggle.addEventListener('change', () => setMonitor(toggle.checked));
})();

// Kill the current task's tmux pane (kill-session + every brand/agent alias).
// Shared by the toolbar "Stop" button and the terminal-bar "关闭 tmux" button.
async function stopClaudePane() {
  if (!STATE.slug) return;
  if (!confirm('关闭当前 tmux 会话？正在运行的 agent 会被一并终止。')) return;
  const killBtn = document.getElementById('term-kill');
  const stopBtn = document.getElementById('btn-interview-stop');
  if (killBtn) killBtn.disabled = true;
  if (stopBtn) stopBtn.disabled = true;
  try {
    const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/claude/stop', {
      method: 'POST',
      body: '{}',
    });
    const tgt = $('#inp-interview-target');
    if (tgt) tgt.value = '';
    const lbl = $('#interview-target-label');
    if (lbl) lbl.textContent = 'Not started';
    disconnectTerminal();
    setTmuxOutputText(`已关闭 ${r.tmux_session || ''}\n${r.tmux_message || ''}`.trim());
    refreshInterviewPreview(true);
    refreshClaudeSessions();
  } catch (err) {
    alert((err && err.message) || '关闭 tmux 失败');
  } finally {
    if (killBtn) killBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = false;
  }
}

document.getElementById('btn-interview-stop').addEventListener('click', stopClaudePane);
const termKillBtn = document.getElementById('term-kill');
if (termKillBtn) termKillBtn.addEventListener('click', stopClaudePane);

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
    const special = agent === 'kernel' || agent === 'aris';
    const body = { title, general_goal, agent: special ? 'claude' : agent };
    if (agent === 'kernel') body.kind = 'kernel';
    else if (agent === 'aris') body.kind = 'aris';
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
