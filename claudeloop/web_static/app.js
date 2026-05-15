/**
 * claudeloop .RUD web client.
 */
const FILES = {
  plan: 'PLAN.md',
  task: 'TASK_PROMPT.md',
  success: 'SUCCESS_CONDITION.md',
  interview: 'INTERVIEW.md',
};

const TABS = [
  { id: 'plan', label: 'PLAN.md' },
  { id: 'task', label: 'TASK_PROMPT.md' },
  { id: 'success', label: 'SUCCESS_CONDITION.md' },
  { id: 'interview', label: 'Interview' },
  { id: 'ask', label: 'Ask' },
  { id: 'panes', label: 'Runner / Evaluator' },
  { id: 'worker', label: 'Worker' },
];

const STATE = {
  slug: null,
  projectId: null,
  projects: [],
  tasks: [],
  launchRoot: '',
  launchRootChildren: [],
  firstInterview: true,
  paneTimer: null,
};

let PROJECT_DRAG_ID = '';
let PROJECT_JUST_DRAGGED = false;
let TASK_DRAG_SLUG = '';
let TASK_JUST_DRAGGED = false;
let WORKTREE_SELECTION = null;
const WORKER_REPO = 'work-root';

function withProjectQuery(path) {
  if (!STATE.projectId) return path;
  if (path.startsWith('/api/projects')) return path;
  if (!path.startsWith('/api/project') && !path.startsWith('/api/tasks')) return path;
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}project=${encodeURIComponent(STATE.projectId)}`;
}

async function apiNoProject(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (opts.body !== undefined && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(path, {
    ...opts,
    headers,
  });
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text };
  }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function api(path, opts = {}) {
  const url = withProjectQuery(path);
  const headers = { ...(opts.headers || {}) };
  if (opts.body !== undefined && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(url, {
    ...opts,
    headers,
  });
  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text };
  }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function $(sel) {
  return document.querySelector(sel);
}

function showPanel(id) {
  document.querySelectorAll('.tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.tab === id);
  });
  document.querySelectorAll('.tab-panel').forEach((p) => {
    const on = p.dataset.panel === id;
    p.classList.toggle('active', on);
    p.hidden = !on;
  });
  if (['plan', 'task', 'success'].includes(id)) refreshTaskTemplates();
  if (id === 'interview') refreshInterviewPreview();
  if (id === 'ask') refreshAskPreview();
}

function buildTabs() {
  const nav = $('#main-tabs');
  nav.innerHTML = '';
  for (const t of TABS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tab' + (t.id === 'plan' ? ' active' : '');
    b.dataset.tab = t.id;
    b.textContent = t.label;
    b.addEventListener('click', () => {
      showPanel(t.id);
      if (t.id === 'panes') refreshPanePreview();
    });
    nav.appendChild(b);
  }
}

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
    $('#hdr-skills').textContent = '—';
    return;
  }
  const d = await api('/api/project');
  const meta = (STATE.projects || []).find((x) => x.id === STATE.projectId);
  const pathLine = d.projectRoot || '';
  $('#hdr-project').textContent = meta ? `${meta.name} — ${pathLine}` : pathLine;
  $('#hdr-skills').textContent = d.skillsPath || '';
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

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
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
    return line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((cell) => cell.trim());
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
      if (listType !== 'ul') {
        flushList();
        listType = 'ul';
        out.push('<ul>');
      }
      out.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      if (listType !== 'ol') {
        flushList();
        listType = 'ol';
        out.push('<ol>');
      }
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

function updateMarkdownPreview(which) {
  const editor = $(`#editor-${which}`);
  const preview = $(`#preview-${which}`);
  if (!editor || !preview) return;
  preview.innerHTML = renderMarkdown(editor.value);
}

function updateAllMarkdownPreviews() {
  ['plan', 'task', 'success', 'interview'].forEach(updateMarkdownPreview);
}

function initMarkdownPreviews() {
  ['plan', 'task', 'success', 'interview'].forEach((which) => {
    const editor = $(`#editor-${which}`);
    if (editor) editor.addEventListener('input', () => updateMarkdownPreview(which));
  });
  updateAllMarkdownPreviews();
}

function previewTitle(which) {
  const names = {
    plan: 'PLAN.md',
    task: 'TASK_PROMPT.md',
    success: 'SUCCESS_CONDITION.md',
    interview: 'INTERVIEW.md',
  };
  const taskTitle = $('#task-title')?.textContent?.trim() || 'Task';
  return `${names[which] || 'Preview'} · ${taskTitle}`;
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
  } catch {
    // Browsers may block fullscreen if the gesture was not accepted; the modal still works.
  }
}

async function closeFullscreenPreview() {
  const modal = $('#preview-modal');
  if (!modal) return;
  modal.hidden = true;
  document.body.classList.remove('preview-open');
  if (document.fullscreenElement) {
    try {
      await document.exitFullscreen();
    } catch {
      // Ignore fullscreen teardown errors.
    }
  }
}

function printFullscreenPreview() {
  const modal = $('#preview-modal');
  if (!modal || modal.hidden) return;
  window.print();
}

function initFullscreenPreviews() {
  ['plan', 'task', 'success', 'interview'].forEach((which) => {
    const preview = $(`#preview-${which}`);
    if (!preview) return;
    preview.title = 'Double-click to open fullscreen preview';
    preview.addEventListener('dblclick', () => openFullscreenPreview(which));
  });
}

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
  const tasks = STATE.tasks || [];
  if (!tasks.length) {
    const li = document.createElement('li');
    li.className = 'task-list__empty';
    li.textContent = STATE.projectId ? 'No tasks yet' : 'Select or add a project';
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
      else selectTask(t.slug);
    });
    ul.appendChild(li);
  }
}

function clearTaskSelection() {
  STATE.slug = null;
  STATE.firstInterview = true;
  if (STATE.paneTimer) {
    clearInterval(STATE.paneTimer);
    STATE.paneTimer = null;
  }
  document.querySelectorAll('#task-list li').forEach((li) => li.classList.remove('active'));
  $('#worktree-status').textContent = '';
  $('#task-view').hidden = true;
  $('#task-empty').hidden = false;
}

async function selectTask(slug) {
  STATE.slug = slug;
  STATE.firstInterview = true;
  document.querySelectorAll('#task-list li').forEach((li) => {
    li.classList.toggle('active', li.dataset.slug === slug);
  });
  const d = await api('/api/tasks/' + encodeURIComponent(slug));
  $('#task-empty').hidden = true;
  $('#task-view').hidden = false;
  $('#task-title').textContent = d.meta.title || slug;
  $('#task-slug').textContent = d.meta.slug;
  $('#task-backend').textContent = `interview: ${d.meta.interview_backend || 'cli'} · model: ${d.meta.interview_model || ''}`;
  $('#task-goal').textContent = d.meta.general_goal || '';
  $('#worktree-status').textContent = '';
  $('#editor-plan').value = d.templates[FILES.plan] || '';
  $('#editor-task').value = d.templates[FILES.task] || '';
  $('#editor-success').value = d.templates[FILES.success] || '';
  $('#editor-interview').value = d.interview || '';
  updateAllMarkdownPreviews();
  $('#inp-interview-target').value = d.meta.tmux_interview_target || '';
  $('#inp-ask-target').value = d.meta.tmux_ask_target || '';
  $('#inp-runner-target').value = d.meta.tmux_runner_target || '';
  $('#inp-eval-target').value = d.meta.tmux_evaluator_target || '';
  $('#interview-target-label').textContent = d.meta.tmux_interview_target || 'Not started';
  $('#ask-target-label').textContent = d.meta.tmux_ask_target || 'Not started';
  $('#runner-target-label').textContent = d.meta.tmux_runner_target || 'Not started';
  $('#eval-target-label').textContent = d.meta.tmux_evaluator_target || 'Not started';
  $('#interview-out').textContent = d.meta.tmux_interview_target
    ? 'Loading interview pane…'
    : (d.interview || 'Click Start deep-interview to launch the Claude Code interview pane.');
  fillRepos(d.work_repos || [], d.meta.work_dirs || []);
  updateWorktreeStatusList(d.work_repos || []);
  buildTabs();
  showPanel('plan');
  await refreshLog();
  refreshInterviewPreview();
  refreshAskPreview();
  refreshPanePreview();
  startPanePolling();
}

async function deleteSelectedTask() {
  if (!STATE.slug) return;
  const slug = STATE.slug;
  const title = $('#task-title')?.textContent || slug;
  const ok = confirm(
    `Delete task "${title}" (${slug})?\n\n` +
    `This permanently removes .RUD/${slug}/, including task files, worktrees, runs, and logs. ` +
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
  const d = await api('/api/tasks/' + encodeURIComponent(STATE.slug));
  $('#editor-plan').value = d.templates[FILES.plan] || '';
  $('#editor-task').value = d.templates[FILES.task] || '';
  $('#editor-success').value = d.templates[FILES.success] || '';
  $('#editor-interview').value = d.interview || '';
  updateAllMarkdownPreviews();
}

function fillRepos(workRepos, workDirs) {
  // Worktree list is displayed in the status row; worker/ask run from the work root.
}

function updateWorktreeStatusList(workRepos) {
  const status = $('#worktree-status');
  if (!status) return;
  const names = Array.isArray(workRepos) ? workRepos : [];
  if (!names.length) {
    status.textContent = 'No worktrees created yet.';
    status.title = '';
    return;
  }
  const text = `Created worktrees: ${names.join(', ')}`;
  status.textContent = text;
  status.title = names.join('\n');
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

async function refreshPanePreview() {
  const rT = $('#inp-runner-target').value.trim();
  const eT = $('#inp-eval-target').value.trim();
  const preR = $('#pre-runner');
  const preE = $('#pre-eval');
  if (!rT) preR.textContent = '(runner not bound)';
  else {
    try {
      const d = await api('/api/tmux/capture?target=' + encodeURIComponent(rT) + '&lines=100');
      preR.textContent = d.ok ? d.text : d.error || '(error)';
    } catch (err) {
      preR.textContent = err.message;
    }
  }
  if (!eT) preE.textContent = '(evaluator not bound)';
  else {
    try {
      const d = await api('/api/tmux/capture?target=' + encodeURIComponent(eT) + '&lines=100');
      preE.textContent = d.ok ? d.text : d.error || '(error)';
    } catch (err) {
      preE.textContent = err.message;
    }
  }
}

async function refreshInterviewPreview() {
  const target = $('#inp-interview-target').value.trim();
  const out = $('#interview-out');
  if (!target) return;
  try {
    const d = await api('/api/tmux/capture?target=' + encodeURIComponent(target) + '&lines=160');
    out.textContent = d.ok ? d.text : d.error || '(error)';
  } catch (err) {
    out.textContent = err.message;
  }
}

async function refreshAskPreview() {
  const target = $('#inp-ask-target').value.trim();
  const out = $('#ask-out');
  if (!target) return;
  try {
    const d = await api('/api/tmux/capture?target=' + encodeURIComponent(target) + '&lines=160');
    out.textContent = d.ok ? d.text : d.error || '(error)';
  } catch (err) {
    out.textContent = err.message;
  }
}

function paneTarget(which) {
  if (which === 'runner') return $('#inp-runner-target').value.trim();
  if (which === 'eval') return $('#inp-eval-target').value.trim();
  if (which === 'ask') return $('#inp-ask-target').value.trim();
  return $('#inp-interview-target').value.trim();
}

function paneInput(which) {
  if (which === 'runner') return $('#pane-runner-input');
  if (which === 'eval') return $('#pane-eval-input');
  if (which === 'ask') return $('#ask-in');
  return $('#interview-in');
}

async function sendPaneText(which, submit = false) {
  const target = paneTarget(which);
  if (!target) {
    alert(`Bind the ${which} pane first.`);
    return;
  }
  const input = paneInput(which);
  const text = input.value;
  if (!text && !submit) return;
  await api('/api/tmux/send-text', {
    method: 'POST',
    body: JSON.stringify({ target, text, submit }),
  });
  input.value = '';
  if (which === 'interview') await refreshInterviewPreview();
  else if (which === 'ask') await refreshAskPreview();
  else await refreshPanePreview();
}

async function sendPaneKey(which, key) {
  const target = paneTarget(which);
  if (!target) {
    alert(`Bind the ${which} pane first.`);
    return;
  }
  await api('/api/tmux/send-key', {
    method: 'POST',
    body: JSON.stringify({ target, key }),
  });
  if (which === 'interview') await refreshInterviewPreview();
  else if (which === 'ask') await refreshAskPreview();
  else await refreshPanePreview();
}

function startPanePolling() {
  if (STATE.paneTimer) clearInterval(STATE.paneTimer);
  STATE.paneTimer = setInterval(() => {
    if (!STATE.slug) return;
    const pan = document.querySelector('.tab-panel[data-panel="panes"]');
    if (pan && !pan.hidden) refreshPanePreview();
    const interview = document.querySelector('.tab-panel[data-panel="interview"]');
    if (interview && !interview.hidden) {
      refreshInterviewPreview();
      refreshTaskTemplates();
    }
    const ask = document.querySelector('.tab-panel[data-panel="ask"]');
    if (ask && !ask.hidden) refreshAskPreview();
    const worker = document.querySelector('.tab-panel[data-panel="worker"]');
    if (worker && !worker.hidden) refreshLog();
  }, 4000);
}

async function refreshLog() {
  if (!STATE.slug) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worker/log?' + new URLSearchParams({ repo: WORKER_REPO }));
  $('#log-pre').textContent = r.tail || '(no output)';
  updateWorkerStatus(r.status || {});
}

function updateWorkerStatus(status) {
  const card = document.getElementById('worker-status-card');
  const session = status.session || {};
  const running = Boolean(status.running);
  const saved = String(session.status || '').toLowerCase();
  const completed = Number(session.completed_iteration || 0);
  const maxRounds = Number(session.max_rounds || 0);
  const current = Number(session.current_round || (running && maxRounds ? completed + 1 : completed));

  let phase = 'idle';
  let label = 'Idle';
  if (running) {
    phase = 'running';
    label = 'Running';
  } else if (saved === 'running') {
    phase = 'stopped';
    label = 'Stopped';
  } else if (saved === 'success') {
    phase = 'complete';
    label = 'Complete';
  } else if (saved === 'failed') {
    phase = 'failed';
    label = 'Failed';
  } else if (saved === 'error') {
    phase = 'error';
    label = 'Error';
  } else if (saved === 'paused') {
    phase = 'paused';
    label = 'Paused';
  } else if (saved && saved !== 'idle' && saved !== 'unknown') {
    label = saved.charAt(0).toUpperCase() + saved.slice(1);
  }

  let pct = maxRounds ? Math.max(0, Math.min(100, (completed / maxRounds) * 100)) : 0;
  if (saved === 'success' || phase === 'complete') {
    pct = 100;
  }

  const badge = $('#worker-status-text');
  if (badge) {
    badge.textContent = label;
    badge.className = `worker-status-badge worker-status-badge--${phase}`;
  }
  if (card) {
    card.dataset.workerPhase = phase;
  }

  $('#worker-status-meta').textContent = status.pid
    ? `pid ${status.pid}${session.updated_at ? ' · updated ' + session.updated_at : ''}`
    : 'No controller process found.';
  $('#worker-round-text').textContent = maxRounds
    ? `Round ${Math.min(current, maxRounds)} / ${maxRounds} · completed ${completed}`
    : 'Round - / -';

  const bar = $('#worker-progress-bar');
  const wrap = document.getElementById('worker-progress');
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.classList.toggle('worker-progress__bar--done', phase === 'complete');
    bar.classList.toggle('worker-progress__bar--bad', phase === 'failed' || phase === 'error');
  }
  if (wrap) {
    wrap.classList.toggle('worker-progress--done', phase === 'complete');
    wrap.classList.toggle('worker-progress--bad', phase === 'failed' || phase === 'error');
  }
}

function formatWorktreeResults(results) {
  const rows = Array.isArray(results) ? results : [];
  const okCount = rows.filter((x) => x && x.ok).length;
  const failCount = rows.length - okCount;
  const summary = failCount
    ? `Worktree: ${okCount} ok, ${failCount} failed`
    : `Worktree ready: ${okCount}`;
  const detail = rows.map((r) => {
    const repo = r.repo_key || r.work_dir || '(unknown repo)';
    const status = r.ok ? 'ok' : 'failed';
    const reason = r.reason ? ` — ${r.reason}` : '';
    const path = r.worktree ? ` (${r.worktree})` : '';
    return `${status}: ${repo}${path}${reason}`;
  }).join('\n');
  return { summary, detail, failCount };
}

async function createWorktrees(workDirs = null) {
  if (!STATE.slug) return;
  const status = $('#worktree-status');
  status.textContent = 'Creating worktree…';
  status.title = '';
  const opts = { method: 'POST', body: workDirs ? JSON.stringify({ work_dirs: workDirs }) : '{}' };
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worktrees', opts);
  const d = await api('/api/tasks/' + encodeURIComponent(STATE.slug));
  fillRepos(d.work_repos || [], d.meta.work_dirs || []);
  updateWorktreeStatusList(d.work_repos || []);
  const formatted = formatWorktreeResults(r.results || []);
  status.textContent = `${status.textContent} · ${formatted.summary}`;
  status.title = [status.title, formatted.detail].filter(Boolean).join('\n\n');
  if (formatted.failCount) {
    alert(`${formatted.summary}\n\n${formatted.detail}`);
  }
}

async function openWorktreeSelectionIfNeeded() {
  if (!STATE.slug) return;
  const status = $('#worktree-status');
  status.textContent = 'Checking repositories…';
  status.title = '';
  const candidates = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worktree-candidates');
  if (candidates.needsSelection) {
    const d = await api('/api/tasks/' + encodeURIComponent(STATE.slug));
    updateWorktreeStatusList(d.work_repos || []);
    openWorktreeSelectModal(candidates);
    return;
  }
  await createWorktrees();
}

function openWorktreeSelectModal(candidates) {
  WORKTREE_SELECTION = candidates || {};
  const modal = $('#worktree-select-modal');
  const list = $('#worktree-select-list');
  const status = $('#worktree-select-status');
  list.innerHTML = '';
  status.textContent = '';
  for (const group of (WORKTREE_SELECTION.groups || [])) {
    const box = document.createElement('div');
    box.className = 'worktree-select-group';
    const title = document.createElement('h3');
    title.textContent = group.workDir || '(unknown directory)';
    box.appendChild(title);
    if (group.kind === 'repo') {
      const p = document.createElement('p');
      p.className = 'modal-subhint';
      p.textContent = `Git repo: ${(group.repos && group.repos[0] && group.repos[0].name) || group.workDir}. This will be created automatically.`;
      box.appendChild(p);
    } else if (!group.repos || !group.repos.length) {
      const p = document.createElement('p');
      p.className = 'modal-subhint status-bad';
      p.textContent = group.reason || 'No direct child git repositories found.';
      box.appendChild(p);
    } else {
      const p = document.createElement('p');
      p.className = 'modal-subhint';
      p.textContent = 'Direct child repos:';
      box.appendChild(p);
      for (const repo of group.repos) {
        const label = document.createElement('label');
        label.className = 'worktree-select-row';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = repo.path || '';
        cb.checked = true;
        cb.dataset.worktreeRepo = '1';
        const span = document.createElement('span');
        span.textContent = `${repo.name || repo.repoKey || repo.path} — ${repo.path}`;
        label.appendChild(cb);
        label.appendChild(span);
        box.appendChild(label);
      }
    }
    list.appendChild(box);
  }
  modal.hidden = false;
}

function closeWorktreeSelectModal() {
  const modal = $('#worktree-select-modal');
  if (modal) modal.hidden = true;
  WORKTREE_SELECTION = null;
}

async function createSelectedWorktrees() {
  const selected = Array.from(document.querySelectorAll('#worktree-select-list input[data-worktree-repo]:checked'))
    .map((x) => x.value)
    .filter(Boolean);
  const auto = [];
  for (const group of ((WORKTREE_SELECTION && WORKTREE_SELECTION.groups) || [])) {
    const repos = group.repos || [];
    if (group.kind === 'repo' && repos[0] && repos[0].path) auto.push(repos[0].path);
    else if (group.kind === 'container' && repos.length === 1 && repos[0].path) auto.push(repos[0].path);
  }
  const workDirs = Array.from(new Set([...auto, ...selected]));
  if (!workDirs.length) {
    $('#worktree-select-status').textContent = 'Select at least one repo.';
    return;
  }
  $('#btn-worktree-select-create').disabled = true;
  $('#worktree-select-status').textContent = 'Creating…';
  try {
    closeWorktreeSelectModal();
    await createWorktrees(workDirs);
  } catch (e) {
    $('#worktree-status').textContent = e.message;
  } finally {
    $('#btn-worktree-select-create').disabled = false;
  }
}

function openCreateModal() {
  if (!STATE.projectId) {
    alert('Select or add a project first.');
    return;
  }
  const modal = $('#create-modal');
  modal.hidden = false;
  $('#new-task-status').textContent = '';
  requestAnimationFrame(() => $('#new-title').focus());
}

function closeCreateModal() {
  $('#create-modal').hidden = true;
}

function resetCreateForm() {
  $('#new-title').value = '';
  $('#new-goal').value = '';
  $('#new-task-status').textContent = '';
}

document.getElementById('btn-add-project').addEventListener('click', openAddProjectModal);
document.getElementById('btn-add-project-close').addEventListener('click', closeAddProjectModal);
document.getElementById('btn-add-project-cancel').addEventListener('click', closeAddProjectModal);
document.getElementById('btn-add-project-save').addEventListener('click', submitAddProject);
$('#add-project-modal').addEventListener('click', (event) => {
  if (event.target.id === 'add-project-modal') closeAddProjectModal();
});

document.getElementById('btn-tmux-refresh').addEventListener('click', loadTmuxSessions);
document.getElementById('btn-tasks-refresh').addEventListener('click', loadTasks);
document.getElementById('btn-create-open').addEventListener('click', openCreateModal);
document.getElementById('btn-empty-create').addEventListener('click', openCreateModal);
document.getElementById('btn-create-close').addEventListener('click', closeCreateModal);
document.getElementById('btn-create-cancel').addEventListener('click', closeCreateModal);
document.getElementById('create-modal').addEventListener('click', (event) => {
  if (event.target.id === 'create-modal') closeCreateModal();
});
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!$('#preview-modal').hidden) closeFullscreenPreview();
  else if (!$('#create-modal').hidden) closeCreateModal();
  else if (!$('#add-project-modal').hidden) closeAddProjectModal();
  else if (!$('#worktree-select-modal').hidden) closeWorktreeSelectModal();
});

document.getElementById('btn-worktree-select-close').addEventListener('click', closeWorktreeSelectModal);
document.getElementById('btn-worktree-select-cancel').addEventListener('click', closeWorktreeSelectModal);
document.getElementById('btn-worktree-select-create').addEventListener('click', createSelectedWorktrees);
$('#worktree-select-modal').addEventListener('click', (event) => {
  if (event.target.id === 'worktree-select-modal') closeWorktreeSelectModal();
});

document.getElementById('btn-preview-close').addEventListener('click', closeFullscreenPreview);
document.getElementById('btn-preview-exit-fullscreen').addEventListener('click', closeFullscreenPreview);
document.getElementById('btn-preview-print').addEventListener('click', printFullscreenPreview);
document.getElementById('preview-modal').addEventListener('click', (event) => {
  if (event.target.id === 'preview-modal') closeFullscreenPreview();
});

document.getElementById('btn-save-plan').addEventListener('click', () => saveTemplate(FILES.plan, '#editor-plan', '#status-plan'));
document.getElementById('btn-save-task').addEventListener('click', () => saveTemplate(FILES.task, '#editor-task', null));
document.getElementById('btn-save-success').addEventListener('click', () => saveTemplate(FILES.success, '#editor-success', null));

async function startInterviewPane() {
  if (!STATE.slug) return;
  showPanel('interview');
  $('#interview-out').textContent = 'Starting Claude Code interview pane…\nPrompt will be pasted automatically in about 5 seconds.';
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/interview/start', {
    method: 'POST',
    body: '{}',
  });
  $('#inp-interview-target').value = r.target || '';
  $('#interview-target-label').textContent = r.target || 'Not started';
  await refreshInterviewPreview();
  setTimeout(refreshInterviewPreview, 6500);
  setTimeout(refreshInterviewPreview, 10000);
}

document.getElementById('btn-interview-start').addEventListener('click', async () => {
  await startInterviewPane();
});

document.getElementById('btn-interview-stop').addEventListener('click', async () => {
  if (!STATE.slug) return;
  if (!confirm('Stop deep-interview? This will kill the associated tmux session.')) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/interview/stop', {
    method: 'POST',
    body: '{}',
  });
  $('#inp-interview-target').value = '';
  $('#interview-target-label').textContent = 'Not started';
  $('#interview-out').textContent = `Stopped ${r.tmux_session}\n${r.tmux_message}`;
});

document.getElementById('btn-interview-send').addEventListener('click', async () => {
  if (!STATE.slug) return;
  await sendPaneText('interview', true);
});

async function startAskPane() {
  if (!STATE.slug) return;
  showPanel('ask');
  $('#ask-out').textContent = 'Starting Claude Opus ask pane…\nContext prompt will be pasted automatically in about 5 seconds.';
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/ask/start', {
    method: 'POST',
    body: JSON.stringify({ repo: WORKER_REPO }),
  });
  $('#inp-ask-target').value = r.target || '';
  $('#ask-target-label').textContent = r.target || 'Not started';
  await refreshAskPreview();
  setTimeout(refreshAskPreview, 6500);
  setTimeout(refreshAskPreview, 10000);
}

document.getElementById('btn-ask-start').addEventListener('click', async () => {
  await startAskPane();
});

document.getElementById('btn-ask-stop').addEventListener('click', async () => {
  if (!STATE.slug) return;
  if (!confirm('Stop ask pane? This will kill the associated tmux session.')) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/ask/stop', {
    method: 'POST',
    body: '{}',
  });
  $('#inp-ask-target').value = '';
  $('#ask-target-label').textContent = 'Not started';
  $('#ask-out').textContent = `Stopped ${r.tmux_session}\n${r.tmux_message}`;
});

document.getElementById('btn-ask-send').addEventListener('click', async () => {
  if (!STATE.slug) return;
  await sendPaneText('ask', true);
});

document.querySelectorAll('[data-send-text]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    await sendPaneText(btn.dataset.sendText, false);
  });
});

document.querySelectorAll('[data-send-text-enter]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    await sendPaneText(btn.dataset.sendTextEnter, true);
  });
});

document.querySelectorAll('.pane-actions [data-key]').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const pane = btn.closest('.pane-actions')?.dataset.pane;
    if (!pane) return;
    await sendPaneKey(pane, btn.dataset.key);
  });
});

document.getElementById('btn-worktrees').addEventListener('click', async () => {
  try {
    await openWorktreeSelectionIfNeeded();
  } catch (e) {
    $('#worktree-status').textContent = e.message;
  }
});

document.getElementById('btn-worker-start').addEventListener('click', async () => {
  if (!STATE.slug) return;
  const body = {
    repo: WORKER_REPO,
    mode: $('#worker-mode').value,
    model: $('#worker-model').value,
    max_iters: Number($('#worker-iters').value || 200),
  };
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worker/start', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  alert(r.ok ? 'Started: ' + r.log_path : r.error);
  if (r.ok && body.mode === 'tmux') {
    await selectTask(STATE.slug);
    showPanel('panes');
  }
  await refreshLog();
});

document.getElementById('btn-worker-stop').addEventListener('click', async () => {
  if (!STATE.slug) return;
  if (!confirm('Stop this task? This will stop the claudeloop process and kill the associated tmux session.')) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worker/stop', {
    method: 'POST',
    body: JSON.stringify({ repo: WORKER_REPO }),
  });
  alert(`Stopped: ${r.tmux_session}\nprocess: ${r.process_message}\ntmux: ${r.tmux_message}`);
  $('#inp-runner-target').value = '';
  $('#inp-eval-target').value = '';
  $('#runner-target-label').textContent = 'Not started';
  $('#eval-target-label').textContent = 'Not started';
  await refreshLog();
  await refreshPanePreview();
});

function formatPublishResults(r) {
  const rows = Array.isArray(r.results) ? r.results : [r];
  return rows.map((row) => {
    const repo = row.repo || '(repo)';
    if (row.ok) {
      return `ok: ${repo} -> ${row.branch || '(branch)'}${row.committed ? ' (committed)' : ' (no changes)'}`;
    }
    return `failed: ${repo}${row.branch ? ' -> ' + row.branch : ''}\n${row.error || 'unknown error'}`;
  }).join('\n\n');
}

document.getElementById('btn-worker-push').addEventListener('click', async () => {
  if (!STATE.slug) return;
  if (!confirm('Commit/push all git worktrees under this task work/ directory to their claudeloop task branches?')) return;
  const r = await api('/api/tasks/' + encodeURIComponent(STATE.slug) + '/worker/push', {
    method: 'POST',
    body: '{}',
  });
  alert(`Publish ${r.ok ? 'completed' : 'failed'} (${r.count || (r.results || []).length || 1} repos)\n\n${formatPublishResults(r)}`);
  await refreshLog();
});

document.getElementById('btn-worker-refresh').addEventListener('click', refreshLog);

document.getElementById('btn-delete-task').addEventListener('click', deleteSelectedTask);

document.getElementById('btn-new-task').addEventListener('click', async () => {
  const title = $('#new-title').value.trim();
  const general_goal = $('#new-goal').value.trim();
  const btn = $('#btn-new-task');
  const status = $('#new-task-status');
  if (!title || !general_goal) {
    status.textContent = 'Title and general goal are required.';
    return;
  }
  btn.disabled = true;
  status.textContent = 'Creating…';
  try {
    const body = { title, general_goal };
    const { meta } = await api('/api/tasks', { method: 'POST', body: JSON.stringify(body) });
    resetCreateForm();
    closeCreateModal();
    await loadTasks();
    await selectTask(meta.slug);
    showPanel('plan');
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
