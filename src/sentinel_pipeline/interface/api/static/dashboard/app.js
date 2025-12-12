let ws = null;
let reconnectTimer = null;
const streams = {};
const modules = {};
const eventsData = [];
const maxEvents = 50;

let streamChart = null;
let moduleChart = null;
const streamChartMaxPoints = 50;
const moduleChartMaxPoints = 50;
const streamColors = {};

const chartBgPlugin = {
  id: 'chartBg',
  beforeDraw(chart) {
    const area = chart.chartArea;
    if (!area) return;
    const { ctx } = chart;
    ctx.save();
    const bg = getComputedStyle(document.documentElement)
      .getPropertyValue('--card-bg')
      .trim() || '#ffffff';
    ctx.fillStyle = bg;
    ctx.fillRect(area.left, area.top, area.right - area.left, area.bottom - area.top);
    ctx.restore();
  },
};

function formatTime(tsSec, { withDate = false } = {}) {
  const d = tsSec !== undefined ? new Date(tsSec * 1000) : new Date();
  const opts = withDate
    ? { year: '2-digit', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Seoul' }
    : { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Seoul' };
  return d.toLocaleString('ko-KR', opts);
}

function setConnectionState(text, cls) {
  const statusEl = document.getElementById('status');
  statusEl.textContent = text;
  statusEl.classList.remove('connected', 'disconnected', 'reconnecting', 'error');
  if (cls) statusEl.classList.add(cls);
}

function renderStreams() {
  const tbody = document.querySelector('#streams tbody');
  tbody.innerHTML = '';
  Object.values(streams)
    .sort((a, b) => (a.stream_id || '').localeCompare(b.stream_id || ''))
    .forEach(s => {
      const tr = document.createElement('tr');
      const cls = s.status === 'RUNNING' ? 'status-running'
                : s.status === 'ERROR' ? 'status-error'
                : s.status === 'RECONNECTING' ? 'status-reconnecting'
                : 'status-stopped';
      const lastTs = s.last_frame_ts ? formatTime(s.last_frame_ts, { withDate: true }) : '';
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
  Object.values(modules)
    .sort((a, b) => (a.name || '').localeCompare(b.name || ''))
    .forEach(m => {
      tbody.innerHTML += `
        <tr>
          <td>${m.name || ''}</td>
          <td>${m.success_count ?? 0}</td>
          <td>${m.error_count ?? 0}</td>
          <td>${m.timeout_count ?? 0}</td>
        </tr>`;
    });
}

function renderEvents() {
  const ul = document.querySelector('#events');
  ul.innerHTML = '';
  eventsData.slice(-maxEvents).reverse().forEach(ev => {
    const ts =
      ev.ts !== undefined
        ? formatTime(ev.ts, { withDate: true })
        : formatTime(undefined, { withDate: true });
    ev.ts !== undefined
      ? formatTime(ev.ts, { withDate: true })
      : formatTime(undefined, { withDate: true });
    const types = Array.isArray(ev.types) ? ev.types.join(', ') : (ev.type || '');
    const stream = ev.stream_id || ev.stream || '';
    const moduleName = ev.module || '';
    const count = ev.count !== undefined ? `count=${ev.count}` : '';
    const text = `[${ts}] ${types} stream=${stream} module=${moduleName} ${count}`.trim();
    const li = document.createElement('li');
    li.textContent = text;
    ul.appendChild(li);
  });
}

function addEvent(ev) {
  eventsData.push(ev);
  renderEvents();
}

function initCharts() {
  if (typeof Chart !== 'undefined') {
    Chart.register(chartBgPlugin);
  }
  const streamCtx = document.getElementById('streamChart');
  const moduleCtx = document.getElementById('moduleChart');
  if (streamCtx) {
    streamChart = new Chart(streamCtx, {
      type: 'line',
      data: { labels: [''], datasets: [] },
      options: {
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { display: true },
          y: { beginAtZero: true },
        },
        plugins: { legend: { display: false } },
      },
    });
  }
  if (moduleCtx) {
    moduleChart = new Chart(moduleCtx, {
      type: 'line',
      data: {
        labels: [''],
        datasets: [
          { label: 'Success', data: [0], borderColor: '#34d399', tension: 0.2, fill: false, pointRadius: 2, borderWidth: 2 },
          { label: 'Errors', data: [0], borderColor: '#f87171', tension: 0.2, fill: false, pointRadius: 2, borderWidth: 2 },
          { label: 'Timeouts', data: [0], borderColor: '#f5a524', tension: 0.2, fill: false, pointRadius: 2, borderWidth: 2 },
        ],
      },
      options: {
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: { display: true },
          y: { beginAtZero: true },
        },
        plugins: { legend: { display: true, position: 'bottom' } },
      },
    });
  }
}

function updateStreamChart(msg) {
  if (!streamChart) return;
  if (streamChart.data.labels[0] === '' && streamChart.data.labels.length > 0) {
    streamChart.data.labels = [];
    streamChart.data.datasets = [];
  }
  const ts = msg.last_frame_ts ? formatTime(msg.last_frame_ts) : formatTime();
  streamChart.data.labels.push(ts);

  const streamId = msg.stream_id || 'unknown';
  let ds = streamChart.data.datasets.find(d => d.label === streamId);
  if (!ds) {
    const palette = ['#4fd1c5', '#7dd3fc', '#a78bfa', '#f472b6', '#fbbf24', '#34d399', '#60a5fa'];
    if (!streamColors[streamId]) {
      const idx = Object.keys(streamColors).length % palette.length;
      streamColors[streamId] = palette[idx];
    }
    const color = streamColors[streamId];
    ds = { label: streamId, data: [], borderColor: color, tension: 0.2, pointRadius: 2, borderWidth: 2 };
    streamChart.data.datasets.push(ds);
  }
  ds.data.push(msg.fps ?? 0);

  if (streamChart.data.labels.length > streamChartMaxPoints) {
    streamChart.data.labels.shift();
    streamChart.data.datasets.forEach(d => d.data.shift());
  }
  streamChart.update('none');
}

function updateModuleChart(msg) {
  if (!moduleChart) return;
  if (moduleChart.data.labels[0] === '' && moduleChart.data.labels.length > 0) {
    moduleChart.data.labels = [];
    moduleChart.data.datasets.forEach((d) => (d.data = []));
  }
  const ts = msg.ts ? formatTime(msg.ts) : formatTime();
  let success = 0, errors = 0, timeouts = 0;
  if (msg.modules) {
    Object.values(msg.modules).forEach((stats) => {
      success += stats.success_count ?? stats.success ?? stats.successes ?? 0;
      errors += stats.error_count ?? stats.errors ?? 0;
      timeouts += stats.timeout_count ?? stats.timeouts ?? 0;
    });
  }
  moduleChart.data.labels.push(ts);
  moduleChart.data.datasets[0].data.push(success);
  moduleChart.data.datasets[1].data.push(errors);
  moduleChart.data.datasets[2].data.push(timeouts);
  if (moduleChart.data.labels.length > moduleChartMaxPoints) {
    moduleChart.data.labels.shift();
    moduleChart.data.datasets.forEach(d => d.data.shift());
  }
  moduleChart.update('none');
}

function initGridstack() {
  if (typeof GridStack === 'undefined') {
    console.warn('GridStack not loaded; skipping draggable/resizable layout');
    return;
  }

  const positionResizeHandles = () => {
    document.querySelectorAll('.grid-stack .ui-resizable-se').forEach((el) => {
      el.style.position = 'absolute';
      el.style.right = '27px';
      el.style.top = '25px';
      el.style.bottom = 'auto';
      el.style.left = 'auto';
      el.style.transform = 'none';
    });
  };

  const grid = GridStack.init(
    {
      float: true,
      column: 12,
      cellHeight: 70,
      margin: 16,
      resizable: { handles: 'e, se, s, sw, w, ne, n, nw' },
      disableOneColumnMode: true,
    },
    '.grid-stack'
  );
  if (!grid) return;
  const resizeCharts = () => {
    if (streamChart) streamChart.resize();
    if (moduleChart) moduleChart.resize();
  };
  grid.on('change', resizeCharts);
  grid.on('resizestop', resizeCharts);
  grid.on('resizestop', positionResizeHandles);
  grid.on('change', positionResizeHandles);

  // 최초 한 번 적용 (GridStack 초기 렌더 이후에 DOM이 만들어지므로 약간 늦게 호출)
  setTimeout(positionResizeHandles, 50);
}

function connect() {
  clearTimeout(reconnectTimer);
  if (ws) {
    try { ws.close(); } catch (e) {}
  }

  const urlInput = document.getElementById('wsUrl').value || location.origin.replace('http', 'ws') + '/ws/admin';
  const token = document.getElementById('token').value.trim();
  let url = urlInput;
  if (token) {
    const delim = url.includes('?') ? '&' : '?';
    url = url + delim + 'token=' + encodeURIComponent(token);
  }
  const protocols = token ? ['api-key', token] : undefined;

  try {
    ws = new WebSocket(url, protocols);
  } catch (e) {
    console.error('WebSocket init failed', e);
    setConnectionState('error', 'error');
    return;
  }

  setConnectionState('connecting...', 'reconnecting');
  ws.onopen = () => setConnectionState('connected', 'connected');
  ws.onerror = () => setConnectionState('error', 'error');
  ws.onclose = () => setConnectionState('disconnected', 'disconnected');
  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'stream_update') {
        streams[msg.stream_id] = msg;
        renderStreams();
        updateStreamChart(msg);
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
          updateModuleChart(msg);
        }
      } else if (msg.type === 'event') {
        addEvent(msg);
      }
    } catch (e) {
      console.error(e);
    }
  };
}

document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initGridstack();
});
