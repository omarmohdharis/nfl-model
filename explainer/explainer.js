// ============================================================================
// Model data
// ============================================================================

function buildFallbackModelData() {
  const arr = new Array(100).fill(0);
  for (let yl = 1; yl <= 99; yl++) {
    if (yl <= 15) arr[yl] = -1.6 + (yl - 1) * 0.107;
    else if (yl <= 50) arr[yl] = (yl - 15) / 18;
    else arr[yl] = (yl - 15) / 18 + (yl - 50) * 0.02;
    if (yl >= 90) arr[yl] += (yl - 90) * 0.18;
  }
  const fg = {};
  for (let yl = 51; yl <= 99; yl++) {
    const distance = (100 - yl) + 17;
    const z = 6.16 - 0.103 * distance;
    fg[yl] = 1 / (1 + Math.exp(-z));
  }
  const punt = {};
  for (let yl = 1; yl <= 67; yl++) {
    punt[yl] = yl <= 5 ? -2.4 : -2.5 + (yl - 1) * 0.038;
  }
  const conv = [];
  for (let ytg = 1; ytg <= 15; ytg++) {
    for (let yl = 1; yl <= 99; yl++) {
      const nearGoalLine = yl >= 90 ? 1 : 0;
      const z = 0.91 - 0.186 * ytg - 0.001 * yl - 0.286 * nearGoalLine;
      conv.push({ ytg, yl, p: 1 / (1 + Math.exp(-z)) });
    }
  }
  return normalizeModelData({ V: arr, fg, punt, conv });
}

function lookupToArray(lookup, maxIndex = 99) {
  const arr = new Array(maxIndex + 1).fill(null);
  Object.entries(lookup || {}).forEach(([key, value]) => {
    const index = parseInt(key, 10);
    arr[index] = value === null || Number.isNaN(value) ? null : Number(value);
  });
  return arr;
}

function normalizeModelData(raw) {
  const V = Array.isArray(raw.V) ? raw.V.slice() : lookupToArray(raw.V);
  const fg = lookupToArray(raw.fg);
  const punt = lookupToArray(raw.punt);
  const conv = new Map();

  (raw.conv || []).forEach((row) => {
    conv.set(`${row.ytg}-${row.yl}`, row.p === null ? null : Number(row.p));
  });

  return { V, fg, punt, conv };
}

function loadModelData() {
  const request = new XMLHttpRequest();
  try {
    request.open('GET', 'model_data.json', false);
    request.overrideMimeType('application/json');
    request.send(null);
    if (request.status === 200 || request.status === 0) {
      return normalizeModelData(JSON.parse(request.responseText));
    }
  } catch (error) {
    console.warn('Using fallback model data because model_data.json could not be loaded.', error);
  }
  return buildFallbackModelData();
}

const modelData = loadModelData();
const V = modelData.V;
const fgByYardline = modelData.fg;
const puntByYardline = modelData.punt;
const conversionBySituation = modelData.conv;
const fgDistances = fgByYardline
  .map((p, yl) => (p === null || p === undefined ? null : 117 - yl))
  .filter((distance) => Number.isFinite(distance))
  .sort((a, b) => a - b);

function pMakeFG(distance) {
  const ownYardline = 117 - distance;
  return pMakeFGByYL(ownYardline);
}

function pMakeFGByYL(ownYardline) {
  const p = fgByYardline[ownYardline];
  return p === null || p === undefined ? null : p;
}

function eVPunt(ownYardline) {
  const ev = puntByYardline[ownYardline];
  return ev === null || ev === undefined ? null : ev;
}

function pConvert(ydstogo, ownYardline) {
  const p = conversionBySituation.get(`${ydstogo}-${ownYardline}`);
  return p === null || p === undefined ? null : p;
}

// Football-themed chart palette
const COLORS = {
  fieldGreen: '#1f4d2d',
  fieldGreenLine: '#2d6f44',
  gold: '#c8a035',
  pigskin: '#8a4a2a',
  penaltyRed: '#b03028',
  ink: '#1a1a18',
};

// Detect dark mode for chart colors
const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
const chartTextColor = isDark ? '#b8b8b1' : '#4a4a44';
const chartGridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';

Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
Chart.defaults.color = chartTextColor;
Chart.defaults.borderColor = chartGridColor;

// ============================================================================
// Section 2: Value function explorer
// ============================================================================

const ylSlider = document.getElementById('yl-slider');
const ylDisplay = document.getElementById('yl-display');
const vDisplay = document.getElementById('v-display');
const marker1 = document.getElementById('marker1');

function formatYardline(yl) {
  if (yl < 50) return `Own ${yl}`;
  if (yl === 50) return 'Midfield';
  return `Opp ${100 - yl}`;
}

function updateValueDisplay() {
  const yl = parseInt(ylSlider.value);
  ylDisplay.textContent = formatYardline(yl);
  const v = V[yl];
  vDisplay.textContent = (v >= 0 ? '+' : '') + v.toFixed(2);
  vDisplay.classList.toggle('negative', v < 0);
  marker1.style.left = yl + '%';
}

ylSlider.addEventListener('input', updateValueDisplay);
updateValueDisplay();

function drawFieldHashMarks(playId) {
  const field = document.getElementById(playId);
  for (let i = 10; i < 100; i += 10) {
    const line = document.createElement('div');
    line.className = 'field-yardline' + (i % 50 === 0 ? ' major' : '');
    line.style.left = i + '%';
    field.appendChild(line);
    if (i === 25 || i === 50 || i === 75) {
      const label = document.createElement('div');
      label.className = 'field-yardline-label';
      label.style.left = i + '%';
      label.textContent = i === 50 ? '50' : (i < 50 ? i : 100 - i);
      field.appendChild(label);
    }
  }
}
drawFieldHashMarks('field1-play');
drawFieldHashMarks('field2-play');

// ============================================================================
// Section 2: Full V curve chart
// ============================================================================

new Chart(document.getElementById('value-chart'), {
  type: 'line',
  data: {
    labels: Array.from({length: 99}, (_, i) => i + 1),
    datasets: [{
      label: 'Expected net points (V)',
      data: V.slice(1, 100),
      borderColor: COLORS.fieldGreen,
      backgroundColor: 'rgba(31, 77, 45, 0.12)',
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2.5,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: COLORS.fieldGreen,
        titleFont: { family: "'Source Serif 4', Georgia, serif", size: 14, weight: '600' },
        bodyFont: { family: "'JetBrains Mono', monospace", size: 13 },
        padding: 10,
        callbacks: {
          title: (ctx) => formatYardline(ctx[0].dataIndex + 1),
          label: (ctx) => `V = ${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)} pts`,
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Own yard line (1 = own goal, 99 = opp goal)', font: { size: 12 } },
        ticks: { maxTicksLimit: 11 },
        grid: { color: chartGridColor },
      },
      y: {
        title: { display: true, text: 'V (expected net points)', font: { size: 12 } },
        grid: {
          color: (ctx) => ctx.tick.value === 0 ? (isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)') : chartGridColor,
        },
      },
    },
  },
});

// ============================================================================
// Section 3: Value iteration animation
// ============================================================================

const iterStates = [
  { own1: 0.0, own25: 0.0, mid: 0.0, opp10: 0.0 },
  { own1: 0.0, own25: 0.4, mid: 1.6, opp10: 4.2 },
  { own1: -0.5, own25: 0.0, mid: 1.4, opp10: 4.0 },
  { own1: -1.2, own25: -0.2, mid: 1.3, opp10: 3.9 },
  { own1: -1.4, own25: -0.3, mid: 1.3, opp10: 3.8 },
];

const iterTexts = [
  'All values start at zero. We have no information yet — this is just our arbitrary starting guess. Click "Iter 1" to see what one pass through the data does.',
  'After iteration 1, only direct scoring matters. Opp 10 jumps to 4.2 because plays starting there often end in TDs or FGs. Own 1 is still 0 because punts give zero immediate points and we are still using V[next]=0 from the initial guess.',
  'Now information starts flowing backward. Midfield drops slightly because we have learned (from iteration 1) that giving up the ball mid-field often leads to opponent scoring. Own 1 dips slightly negative.',
  'Iteration 3 propagates the information one more step. Own 1 has now learned that punting from there gives the opponent decent field position, and the opponent often scores. Own 1 ≈ -1.2.',
  'After ~30 iterations, the values stop changing. Every yard line has fully absorbed the consequences of all downstream events, no matter how many possessions away. This is the converged value function.',
];

const updatedCells = [
  [],
  ['opp10', 'mid', 'own25'],
  ['mid', 'own25', 'own1'],
  ['own25', 'own1'],
  ['own1'],
];

function setIter(n) {
  document.querySelectorAll('.iter-btn').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.iter) === n);
  });
  const state = iterStates[n];
  document.querySelectorAll('.v-cell').forEach((cell) => {
    const key = cell.dataset.yl;
    const val = state[key];
    cell.querySelector('.v-cell-value').textContent =
      (val > 0 ? '+' : '') + val.toFixed(1);
    cell.classList.remove('updated', 'negative');
    if (updatedCells[n].includes(key)) cell.classList.add('updated');
    if (val < 0) cell.classList.add('negative');
  });
  document.getElementById('iter-text').textContent = iterTexts[n];
}

// ============================================================================
// Section 4: Component model charts
// ============================================================================

new Chart(document.getElementById('fg-chart'), {
  type: 'line',
  data: {
    labels: fgDistances,
    datasets: [{
      label: 'P(make)',
      data: fgDistances.map((distance) => pMakeFG(distance)),
      borderColor: COLORS.penaltyRed,
      backgroundColor: 'rgba(176, 48, 40, 0.10)',
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2.5,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: COLORS.penaltyRed,
        titleFont: { family: "'Source Serif 4', Georgia, serif", size: 14, weight: '600' },
        bodyFont: { family: "'JetBrains Mono', monospace", size: 13 },
        callbacks: {
          title: (ctx) => `${ctx[0].label}-yard kick`,
          label: (ctx) => `${(ctx.parsed.y * 100).toFixed(1)}% make rate`,
        },
      },
    },
    scales: {
      x: { title: { display: true, text: 'Kick distance (yards)', font: { size: 12 } } },
      y: {
        title: { display: true, text: 'Make probability', font: { size: 12 } },
        min: 0, max: 1,
        ticks: { callback: (v) => (v * 100) + '%' },
      },
    },
  },
});

new Chart(document.getElementById('punt-chart'), {
  type: 'line',
  data: {
    labels: puntByYardline
      .map((ev, yl) => (ev === null || ev === undefined ? null : yl))
      .filter((yl) => yl !== null),
    datasets: [{
      label: 'E[V after punt]',
      data: puntByYardline.filter((ev) => ev !== null && ev !== undefined),
      borderColor: COLORS.fieldGreen,
      backgroundColor: 'rgba(31, 77, 45, 0.12)',
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2.5,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: COLORS.fieldGreen,
        callbacks: {
          title: (ctx) => formatYardline(parseInt(ctx[0].label, 10)),
          label: (ctx) => `Punt EV = ${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)} pts`,
        },
      },
    },
    scales: {
      x: { title: { display: true, text: "Kicking team's own yard line", font: { size: 12 } } },
      y: {
        title: { display: true, text: 'Expected net points after punt', font: { size: 12 } },
        grid: {
          color: (ctx) => ctx.tick.value === 0 ? (isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)') : chartGridColor,
        },
      },
    },
  },
});

new Chart(document.getElementById('conv-chart'), {
  type: 'line',
  data: {
    labels: Array.from({length: 15}, (_, i) => i + 1),
    datasets: [{
      label: 'P(convert)',
      data: Array.from({length: 15}, (_, i) => pConvert(i + 1, 50)),
      borderColor: COLORS.fieldGreen,
      backgroundColor: 'rgba(31, 77, 45, 0.10)',
      fill: true,
      tension: 0.3,
      pointRadius: 4,
      pointBackgroundColor: COLORS.gold,
      pointBorderColor: COLORS.fieldGreen,
      pointBorderWidth: 2,
      borderWidth: 2.5,
    }],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: COLORS.fieldGreen,
        titleFont: { family: "'Source Serif 4', Georgia, serif", size: 14, weight: '600' },
        bodyFont: { family: "'JetBrains Mono', monospace", size: 13 },
        callbacks: {
          title: (ctx) => `4th-and-${ctx[0].label}`,
          label: (ctx) => `${(ctx.parsed.y * 100).toFixed(1)}% conversion rate`,
        },
      },
    },
    scales: {
      x: { title: { display: true, text: 'Yards to go', font: { size: 12 } } },
      y: {
        title: { display: true, text: 'Conversion probability', font: { size: 12 } },
        min: 0, max: 1,
        ticks: { callback: (v) => (v * 100) + '%' },
      },
    },
  },
});

function showComponent(which) {
  ['fg', 'punt', 'conv'].forEach((c) => {
    document.getElementById('component-' + c).style.display =
      c === which ? '' : 'none';
  });
  document.querySelectorAll('.pill').forEach((p) => {
    const text = p.textContent.toLowerCase();
    const match =
      (which === 'fg' && text.includes('field goal')) ||
      (which === 'punt' && text.includes('punt')) ||
      (which === 'conv' && text.includes('conversion'));
    p.classList.toggle('active', match);
  });
}

// ============================================================================
// Section 5: Decision calculator
// ============================================================================

const TD_VALUE = 6.984;
const FG_VALUE = 3.0;
const KICKOFF_RETURN_YL = 25;
const V_AFTER_OUR_SCORE = -V[KICKOFF_RETURN_YL];

function evGoForIt(ownYL, ytg) {
  const p = pConvert(ytg, ownYL);
  if (p === null) return null;
  const newYL = ownYL + ytg;
  let vIfConvert;
  if (newYL >= 100) vIfConvert = TD_VALUE + V_AFTER_OUR_SCORE;
  else vIfConvert = V[newYL];
  if (vIfConvert === null || vIfConvert === undefined) return null;
  const oppOwnYL = 100 - ownYL;
  const vIfFail = -V[oppOwnYL];
  if (V[oppOwnYL] === null || V[oppOwnYL] === undefined) return null;
  return p * vIfConvert + (1 - p) * vIfFail;
}

function evFG(ownYL) {
  if (ownYL < 50) return null;
  const p = pMakeFGByYL(ownYL);
  if (p === null) return null;
  const vIfMade = FG_VALUE + V_AFTER_OUR_SCORE;
  const distToEZ = 100 - ownYL;
  let oppOwnYL;
  if (distToEZ < 20) oppOwnYL = 20;
  else oppOwnYL = 100 - (ownYL - 7);
  const vIfMissed = -V[oppOwnYL];
  if (V[oppOwnYL] === null || V[oppOwnYL] === undefined) return null;
  return p * vIfMade + (1 - p) * vIfMissed;
}

const decYL = document.getElementById('dec-yl');
const decYTG = document.getElementById('dec-ytg');
const decYLDisp = document.getElementById('dec-yl-display');
const decYTGDisp = document.getElementById('dec-ytg-display');
const marker2 = document.getElementById('marker2');

function updateDecision() {
  const yl = parseInt(decYL.value);
  const ytg = parseInt(decYTG.value);
  decYLDisp.textContent = formatYardline(yl);
  decYTGDisp.textContent = ytg;
  marker2.style.left = yl + '%';

  const evGo = evGoForIt(yl, ytg);
  const evFg = evFG(yl);
  const evPt = eVPunt(yl);

  const options = [
    { name: 'go_for_it', ev: evGo, label: 'Go for it' },
    { name: 'field_goal', ev: evFg, label: 'Attempt a field goal' },
    { name: 'punt', ev: evPt, label: 'Punt' },
  ].filter((o) => o.ev !== null && !isNaN(o.ev));

  const best = options.reduce((a, b) => (a.ev > b.ev ? a : b));

  const allEvs = options.map((o) => o.ev);
  const minEv = Math.min(...allEvs, -3);
  const maxEv = Math.max(...allEvs, 3);
  const range = maxEv - minEv || 1;
  const zeroPct = ((0 - minEv) / range) * 100;

  ['go', 'fg', 'pt'].forEach((id) => {
    const zero = document.getElementById('zero-' + id);
    if (zero) zero.style.left = zeroPct + '%';
  });

  function setBar(barId, valId, ev) {
    const bar = document.getElementById(barId);
    const valEl = document.getElementById(valId);
    if (ev === null || isNaN(ev)) {
      bar.style.width = '0';
      valEl.textContent = '—';
      return;
    }
    const pct = ((ev - minEv) / range) * 100;
    if (ev >= 0) {
      bar.style.left = zeroPct + '%';
      bar.style.width = (pct - zeroPct) + '%';
    } else {
      bar.style.left = pct + '%';
      bar.style.width = (zeroPct - pct) + '%';
    }
    valEl.textContent = (ev >= 0 ? '+' : '') + ev.toFixed(2);
  }

  setBar('bar-go', 'ev-go', evGo);
  setBar('bar-fg', 'ev-fg', evFg);
  setBar('bar-pt', 'ev-pt', evPt);

  document.querySelectorAll('.ev-bar-fill').forEach((b) => b.classList.remove('recommended'));
  document.getElementById(
    best.name === 'go_for_it' ? 'bar-go' :
    best.name === 'field_goal' ? 'bar-fg' : 'bar-pt'
  ).classList.add('recommended');

  document.getElementById('rec-text').textContent = best.label;
}

decYL.addEventListener('input', updateDecision);
decYTG.addEventListener('input', updateDecision);
updateDecision();
