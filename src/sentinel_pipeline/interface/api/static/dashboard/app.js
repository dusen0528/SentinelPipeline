let ws = null;
const streams = {};
const modules = {};
const maxEvents = 50;

function renderStreams() {
  const tbody = document.querySelector('#streams tbody');
  tbody.innerHTML = '';
  Object.values(streams).forEach(s => {
    const tr = document.createElement('tr');
    const cls = s.status === 'RUNNING' ? 'status-running'
              : s.status === 'ERROR' ? 'status-error'
              : s.status === 'RECONNECTING' ? 'status-reconnecting'
              : 'status-stopped';
    const lastTs = s.last_frame_ts ? new Date(s.last_frame_ts * 1000).toLocaleString() : '';
    tr.innerHTML = `
      <td>${s.stream_id || ''}</td>
      <td class="${cls}">${s.status || ''}</td>
      <td>${s.fps ?? ''}</td>
      <td>${s.error_count ?? ''}</td>
      <td>${lastTs}</td>`;
    tbody.appendChild(tr);
  });
}

function renderModules() {
  const tbody = document.querySelector('#modules tbody');
  tbody.innerHTML = '';
  Object.values(modules).forEach(m => {
    tbody.innerHTML += `
      <tr>
        <td>${m.name || ''}</td>
        <td>${m.success_count ?? 0}</td>
        <td>${m.error_count ?? 0}</td>
        <td>${m.timeout_count ?? 0}</td>
      </tr>`;
  });
}

function addEvent(ev) {
  const ul = document.querySelector('#events');
  const li = document.createElement('li');
  li.textContent = JSON.stringify(ev);
  ul.prepend(li);
  while (ul.children.length > maxEvents) {
    ul.removeChild(ul.lastChild);
  }
}

function connect() {
  const urlInput = document.getElementById('wsUrl').value || location.origin.replace('http', 'ws') + '/ws/admin';
  const token = document.getElementById('token').value;
  let url = urlInput;
  if (token) {
    const delim = url.includes('?') ? '&' : '?';
    url = url + delim + 'token=' + encodeURIComponent(token);
  }
  ws = new WebSocket(url);
  const statusEl = document.getElementById('status');
  statusEl.textContent = 'connecting...';
  statusEl.classList.remove('connected', 'disconnected');
  ws.onopen = () => {
    statusEl.textContent = 'connected';
    statusEl.classList.remove('disconnected');
    statusEl.classList.add('connected');
  };
  ws.onclose = () => {
    statusEl.textContent = 'disconnected';
    statusEl.classList.remove('connected');
    statusEl.classList.add('disconnected');
  };
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'stream_update') {
        streams[msg.stream_id] = msg;
        renderStreams();
      } else if (msg.type === 'module_stats') {
        if (msg.modules) {
          Object.entries(msg.modules).forEach(([name, stats]) => {
            modules[name] = {
              name,
              success_count: stats.success_count ?? stats.success ?? stats.successes ?? 0,
              error_count: stats.error_count ?? stats.errors ?? 0,
              timeout_count: stats.timeout_count ?? stats.timeouts ?? 0,
            };
          });
          renderModules();
        }
      } else if (msg.type === 'event') {
        addEvent(msg);
      }
    } catch (e) {
      console.error(e);
    }
  };
}

