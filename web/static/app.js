// ── Clock ──────────────────────────────────────────────────────────────────
function tick() {
  const n = new Date();
  document.getElementById('clock').textContent =
    [n.getHours(), n.getMinutes(), n.getSeconds()]
      .map(x => String(x).padStart(2,'0')).join(':');
}
setInterval(tick, 1000); tick();

// ── Formatters ──────────────────────────────────────────────────────────────
function fmtUptime(s) { return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`; }
function fmtCountdown(iso) {
  const ms = new Date(iso.replace(' ','T')) - new Date();
  if (ms <= 0) return 'SCADUTO';
  const h = Math.floor(ms/3600000), m = Math.floor((ms%3600000)/60000);
  return h > 0 ? `tra ${h}h ${m}m` : `tra ${m}min`;
}
function fmtDate(iso) { return iso ? iso.substring(0,16).replace('T',' ') : '—'; }
function fmtTok(n) {
  if (!n) return '0';
  return n >= 1000 ? (n/1000).toFixed(1) + 'k' : String(n);
}
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function pad(n) { return String(n).padStart(2,'0'); }

// ── Status ──────────────────────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const d = await fetch('/api/status').then(r => r.json());
    document.getElementById('s-act').textContent = d.activity || '—';
    const roomEl = document.getElementById('s-room');
    roomEl.textContent = d.in_room === null ? '—' : (d.in_room ? 'IN STANZA' : 'ASSENTE');
    roomEl.className   = 'stat-val ' + (d.in_room ? 'on' : 'off');
    const deskEl = document.getElementById('s-desk');
    deskEl.textContent = d.at_desk === null ? '—' : (d.at_desk ? 'ALLA SCRIVANIA' : 'NO');
    deskEl.className   = 'stat-val ' + (d.at_desk ? 'on' : 'off');
    document.getElementById('s-upd').textContent   = d.last_update || '—';
    document.getElementById('m-ev').textContent    = d.total_events?.toLocaleString() ?? '—';
    document.getElementById('m-inf').textContent   = d.avg_inference_ms ? `${Math.round(d.avg_inference_ms)}ms` : '—';
    document.getElementById('m-cache').textContent = d.cache ? `${d.cache.hits} hit · ${d.cache.entries} entry` : '—';
    document.getElementById('m-up').textContent    = d.uptime_s ? fmtUptime(d.uptime_s) : '—';
    document.getElementById('m-tok-s').textContent = d.tokens ? fmtTok(d.tokens.session.total) : '—';
    document.getElementById('m-tok-t').textContent = d.tokens ? fmtTok(d.tokens.total.total)   : '—';
  } catch {
    document.getElementById('status-dot').style.cssText = 'background:var(--red);box-shadow:0 0 10px var(--red)';
    document.getElementById('status-txt').textContent = 'OFFLINE';
  }
}

// ── Reminders ───────────────────────────────────────────────────────────────
async function fetchReminders() {
  const items = await fetch('/api/reminders').then(r => r.json());
  const el = document.getElementById('rem-list');
  if (!items.length) { el.innerHTML = '<div class="empty">Nessun promemoria</div>'; return; }
  el.innerHTML = items.map(r => `
    <div class="rem-item" id="r${r.id}">
      <div class="rem-top">
        <span class="rem-time">${fmtDate(r.trigger_time)}</span>
        <span>
          <button class="btn-done" onclick="done(${r.id})">FATTO ✓</button>
          <button class="btn-del" onclick="delReminder(${r.id})" title="Elimina">✕</button>
        </span>
      </div>
      <div class="rem-text">${r.text}</div>
      <div class="rem-tag">
        ${r.category || ''}${r.repeat && r.repeat !== 'none' ? ' · ' + r.repeat : ''}
        · <span style="color:var(--cyan)">${fmtCountdown(r.trigger_time)}</span>
      </div>
    </div>`).join('');
}
async function done(id) {
  await fetch(`/api/reminders/${id}/done`, {method:'POST'});
  document.getElementById('r'+id)?.remove();
}
async function delReminder(id) {
  await fetch(`/api/reminders/${id}`, {method:'DELETE'});
  document.getElementById('r'+id)?.remove();
}

// ── Add reminder form ────────────────────────────────────────────────────────
async function addReminder() {
  const text = document.getElementById('add-rem-text').value.trim();
  const dt   = document.getElementById('add-rem-dt').value;
  const cat  = document.getElementById('add-rem-cat').value;
  if (!text || !dt) return;
  const res = await fetch('/api/reminders', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({text, trigger_time: dt, category: cat}),
  });
  if (res.ok) {
    document.getElementById('add-rem-text').value = '';
    beep(660, 0.3, 0.2);
    fetchReminders();
  }
}

// ── Fired reminders notification ─────────────────────────────────────────────
const _seenFired = new Set();

async function checkFired() {
  try {
    const items = await fetch('/api/reminders/fired').then(r => r.json());
    for (const item of items) {
      if (!_seenFired.has(item.id)) {
        _seenFired.add(item.id);
        showFiredBanner(item);
      }
    }
  } catch {}
}

function showFiredBanner(item) {
  document.getElementById('fired-text').textContent = item.text;
  document.getElementById('fired-banner').style.display = 'flex';
  item.category === 'sveglia' ? beepLoop(6, 800) : beep(880, 0.4, 0.5);
  setTimeout(dismissBanner, 30000);
  fetchReminders();
}
function dismissBanner() {
  document.getElementById('fired-banner').style.display = 'none';
}

// ── Web Audio beep ───────────────────────────────────────────────────────────
let _audioCtx = null;
function _getCtx() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return _audioCtx;
}
function beep(freq = 880, vol = 0.4, dur = 0.3) {
  try {
    const ctx = _getCtx();
    const osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.type = 'sine'; osc.frequency.value = freq;
    gain.gain.setValueAtTime(vol, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
    osc.start(); osc.stop(ctx.currentTime + dur);
  } catch {}
}
function beepLoop(times, intervalMs = 1000) {
  let n = 0;
  const id = setInterval(() => { beep(880,0.5,0.4); if (++n >= times) clearInterval(id); }, intervalMs);
}

// ── Chat ─────────────────────────────────────────────────────────────────────
function appendMsg(role, text) {
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + (role === 'user' ? 'user' : 'nico');
  if (role === 'nico') div.innerHTML = '<div class="msg-label">N·I·C·O</div>' + escHtml(text);
  else div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
async function sendChat() {
  const inp = document.getElementById('chat-input');
  const btn = document.getElementById('chat-send');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = ''; btn.disabled = true;
  appendMsg('user', text);
  try {
    const res = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}),
    });
    const data = await res.json();
    appendMsg('nico', data.response || data.error || 'Nessuna risposta.');
    fetchReminders();
  } catch { appendMsg('nico', 'Errore di connessione.'); }
  finally { btn.disabled = false; inp.focus(); }
}

// ── Notes ────────────────────────────────────────────────────────────────────
async function fetchNotes() {
  const q = document.getElementById('note-q').value;
  const url = q ? `/api/notes?q=${encodeURIComponent(q)}` : '/api/notes';
  const items = await fetch(url).then(r => r.json());
  const el = document.getElementById('note-list');
  if (!items.length) { el.innerHTML = '<div class="empty">Nessuna nota</div>'; return; }
  el.innerHTML = items.map(n => `
    <div class="note-item">
      <div class="note-text">${n.text}</div>
      <div class="note-meta">${fmtDate(n.created_at)}${n.category ? ' · ' + n.category : ''}</div>
    </div>`).join('');
}

// ── Sessions ─────────────────────────────────────────────────────────────────
async function fetchSessions() {
  const items = await fetch('/api/sessions').then(r => r.json());
  const el = document.getElementById('sess-list');
  if (!items.length) { el.innerHTML = '<div class="empty">Nessuna sessione</div>'; return; }
  el.innerHTML = items.map(s => `
    <div class="sess-item">
      <span class="sess-act">${s.activity || '—'}</span>
      <span class="sess-dur">${s.duration_min ? Math.round(s.duration_min)+'min' : 'in corso'}</span>
      <span class="sess-time">${fmtDate(s.start)}</span>
    </div>`).join('');
}

// ── Stopwatch ────────────────────────────────────────────────────────────────
let _swRunning = false, _swStart = 0, _swAccum = 0, _swInterval = null;

function fmtMs(ms) {
  const m = Math.floor(ms/60000), s = Math.floor((ms%60000)/1000), cs = Math.floor((ms%1000)/10);
  return `${pad(m)}:${pad(s)}.${pad(cs)}`;
}
function swToggle() {
  if (_swRunning) {
    _swRunning = false; _swAccum += Date.now() - _swStart; clearInterval(_swInterval);
    document.getElementById('sw-btn').textContent = 'START';
    document.getElementById('sw-btn').className   = 'util-btn start';
  } else {
    _swRunning = true; _swStart = Date.now();
    _swInterval = setInterval(() => {
      document.getElementById('sw-display').textContent = fmtMs(_swAccum + Date.now() - _swStart);
    }, 33);
    document.getElementById('sw-btn').textContent = 'STOP';
    document.getElementById('sw-btn').className   = 'util-btn stop';
  }
}
function swReset() {
  if (_swRunning) swToggle();
  _swAccum = 0;
  document.getElementById('sw-display').textContent = '00:00.00';
}

// ── Timer ────────────────────────────────────────────────────────────────────
let _tiRunning = false, _tiEnd = 0, _tiInterval = null;

function timerToggle() {
  if (_tiRunning) {
    clearInterval(_tiInterval); _tiRunning = false;
    document.getElementById('timer-btn').textContent = 'START';
    document.getElementById('timer-btn').className   = 'util-btn start';
    return;
  }
  const m = parseInt(document.getElementById('timer-min').value) || 0;
  const s = parseInt(document.getElementById('timer-sec').value) || 0;
  const dur = (m * 60 + s) * 1000;
  if (!dur) return;
  _tiEnd = Date.now() + dur; _tiRunning = true;
  document.getElementById('timer-btn').textContent = 'STOP';
  document.getElementById('timer-btn').className   = 'util-btn stop';
  _tiInterval = setInterval(() => {
    const rem = _tiEnd - Date.now();
    if (rem <= 0) {
      clearInterval(_tiInterval); _tiRunning = false;
      document.getElementById('timer-display').textContent = '00:00';
      document.getElementById('timer-btn').textContent = 'START';
      document.getElementById('timer-btn').className   = 'util-btn start';
      beepLoop(6, 800);
      const disp = document.getElementById('timer-display');
      disp.style.color = 'var(--red)';
      setTimeout(() => { disp.style.color = ''; }, 6000);
      return;
    }
    const mm = Math.floor(rem/60000), ss = Math.floor((rem%60000)/1000);
    document.getElementById('timer-display').textContent = `${pad(mm)}:${pad(ss)}`;
  }, 500);
}
function timerReset() {
  if (_tiRunning) timerToggle();
  document.getElementById('timer-display').textContent = '00:00';
  document.getElementById('timer-display').style.color = '';
}

// ── Poll ─────────────────────────────────────────────────────────────────────
async function refresh() {
  await Promise.allSettled([fetchStatus(), fetchReminders(), fetchNotes(), fetchSessions(), checkFired()]);
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('note-q').addEventListener('keyup', e => { if(e.key==='Enter') fetchNotes(); });
  document.getElementById('chat-input').addEventListener('keyup', e => { if(e.key==='Enter') sendChat(); });
  document.getElementById('add-rem-text').addEventListener('keyup', e => { if(e.key==='Enter') addReminder(); });
});

refresh();
setInterval(refresh, 5000);
