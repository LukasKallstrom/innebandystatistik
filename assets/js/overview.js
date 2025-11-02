import { loadGoalieDataset } from './data.js';
import {
  attachNavHighlight,
  computeDateRange,
  computeUniqueTeams,
  computeZeroShotGoalies,
  createElement,
  createOption,
  formatNumber,
  formatPercent,
  parseSearchTerms,
  sortData,
} from './utils.js';

const dateFormatter = new Intl.DateTimeFormat('sv-SE', { dateStyle: 'medium' });
const updatedFormatter = new Intl.DateTimeFormat('sv-SE', {
  dateStyle: 'long',
  timeStyle: 'short',
});
const updatedTitleFormatter = new Intl.DateTimeFormat('sv-SE', {
  dateStyle: 'full',
  timeStyle: 'long',
});

const timelineState = {
  goalieMap: {},
  traceOrder: [],
  traceIndex: new Map(),
};

const timelineElements = {
  chart: null,
  teamSelect: null,
  searchInput: null,
  resetButton: null,
  resizeListenerAttached: false,
};

let timelineListenersRegistered = false;
let isRefreshing = false;

function populateStats({ teams, goalies, shots, range, zeroShotCount }) {
  const teamEl = document.querySelector('[data-stat="team-count"]');
  const goalieEl = document.querySelector('[data-stat="goalie-count"]');
  const shotsEl = document.querySelector('[data-stat="shot-volume"]');
  const coverageEl = document.querySelector('[data-stat="coverage"]');
  const zeroShotEl = document.querySelector('[data-stat="zero-shot-count"]');

  if (teamEl) teamEl.textContent = formatNumber(teams);
  if (goalieEl) goalieEl.textContent = formatNumber(goalies);
  if (shotsEl) shotsEl.textContent = formatNumber(shots);
  if (coverageEl) {
    coverageEl.textContent = range
      ? `${dateFormatter.format(range.start)} – ${dateFormatter.format(range.end)}`
      : 'Ingen tidslinje tillgänglig';
  }
  if (zeroShotEl) zeroShotEl.textContent = formatNumber(zeroShotCount);
}

function renderZeroShotTags(goalies) {
  const container = document.getElementById('zero-shot-tags');
  if (!container) return;
  container.innerHTML = '';
  if (!goalies.length) {
    container.textContent = 'Alla målvakter har registrerade skott.';
    return;
  }
  goalies.slice(0, 14).forEach((goalie) => {
    const tag = createElement('span', { className: 'tag', text: goalie.goalie });
    container.appendChild(tag);
  });
  if (goalies.length > 14) {
    const extra = createElement('span', {
      className: 'tag',
      text: `+${goalies.length - 14} fler`,
    });
    container.appendChild(extra);
  }
}

function renderTopGoalies(summary, goalieMap) {
  const container = document.getElementById('top-goalies');
  if (!container) return;
  container.innerHTML = '';
  const ranked = summary
    .filter((row) => row.total_shots >= 25)
    .sort((a, b) => b.final_save_pct - a.final_save_pct)
    .slice(0, 4);
  ranked.forEach((row) => {
    const card = createElement('article', { className: 'card' });
    card.innerHTML = `
      <span class="badge">${formatPercent(row.final_save_pct, 1)}</span>
      <h3 class="card__title">${row.goalie}</h3>
      <p class="card__subtitle">${goalieMap[row.goalie] || row.team || 'Okänt lag'}</p>
      <div class="card__meta">${formatNumber(row.total_shots)} skott | ${formatNumber(row.games_played)} matcher</div>
    `;
    container.appendChild(card);
  });
}

function renderSeasonInsights(summary) {
  const list = document.getElementById('insights-list');
  if (!list) return;
  list.innerHTML = '';

  const teams = new Map();
  summary.forEach((row) => {
    const key = row.team || 'Okänt lag';
    if (!teams.has(key)) {
      teams.set(key, { shots: 0, saves: 0, goals: 0 });
    }
    const entry = teams.get(key);
    entry.shots += row.total_shots || 0;
    entry.saves += row.total_saves || 0;
    entry.goals += row.goals_against || 0;
  });

  const mostShots = [...teams.entries()].sort((a, b) => b[1].shots - a[1].shots)[0];
  const highestSavePct = sortData(
    [...summary],
    'final_save_pct',
    false,
  ).find((row) => row.total_shots >= 30);

  if (mostShots) {
    const [team, values] = mostShots;
    const item = createElement('li');
    item.innerHTML = `<strong>${team}</strong> har mött flest skott med ${formatNumber(
      values.shots,
    )} registrerade avslut.`;
    list.appendChild(item);
  }

  if (highestSavePct) {
    const item = createElement('li');
    item.innerHTML = `<strong>${highestSavePct.goalie}</strong> leder ligan i räddningsprocent (${formatPercent(
      highestSavePct.final_save_pct,
      1,
    )}) efter ${formatNumber(highestSavePct.total_shots)} skott.`;
    list.appendChild(item);
  }

  const averageShots = summary.length
    ? summary.reduce((acc, row) => acc + (row.avg_shots_per_game || 0), 0) / summary.length
    : 0;
  const item = createElement('li');
  item.innerHTML = `Snittbelastningen per match är ${formatNumber(averageShots, {
    maximumFractionDigits: 1,
    minimumFractionDigits: 1,
  })} skott mot mål.`;
  list.appendChild(item);
}

function cloneTimelineData(timeline) {
  return (timeline?.data ?? []).map((trace) => ({ ...trace }));
}

function ensureTimelineElements() {
  if (!timelineElements.chart) {
    timelineElements.chart = document.getElementById('timeline-chart');
  }
  if (!timelineElements.teamSelect) {
    timelineElements.teamSelect = document.getElementById('timeline-team');
  }
  if (!timelineElements.searchInput) {
    timelineElements.searchInput = document.getElementById('timeline-search');
  }
  if (!timelineElements.resetButton) {
    timelineElements.resetButton = document.getElementById('timeline-reset');
  }
}

function populateTimelineTeams(traceOrder, goalieMap, previousValue = 'ALL') {
  ensureTimelineElements();
  const select = timelineElements.teamSelect;
  if (!select) return;

  const uniqueTeams = computeUniqueTeams(
    traceOrder.map((goalie) => ({ team: goalieMap[goalie] })).filter((row) => row.team),
  );

  const desiredValue = uniqueTeams.includes(previousValue) ? previousValue : 'ALL';

  select.innerHTML = '';
  select.appendChild(createOption('ALL', 'Alla lag'));
  uniqueTeams.forEach((team) => select.appendChild(createOption(team, team)));
  select.value = desiredValue;
}

function applyTimelineFilters() {
  ensureTimelineElements();
  const chart = timelineElements.chart;
  if (!chart) return;

  const teamValue = timelineElements.teamSelect?.value || 'ALL';
  const searchTerms = parseSearchTerms(timelineElements.searchInput?.value || '');

  timelineState.traceOrder.forEach((goalie) => {
    const index = timelineState.traceIndex.get(goalie);
    if (index === undefined) return;
    const team = timelineState.goalieMap[goalie] || 'Okänt lag';
    const matchesTeam = teamValue === 'ALL' || team === teamValue;
    const matchesSearch = !searchTerms.length || searchTerms.some((term) => goalie.toLowerCase().includes(term));
    const visible = matchesTeam && matchesSearch;
    Plotly.restyle(chart, { visible: visible ? true : 'legendonly' }, [index]);
  });
}

function setAllTracesVisible() {
  const chart = timelineElements.chart;
  if (!chart) return;
  timelineState.traceIndex.forEach((_, goalie) => {
    const index = timelineState.traceIndex.get(goalie);
    if (index === undefined) return;
    Plotly.restyle(chart, { visible: true }, [index]);
  });
}

function handleTimelineReset() {
  ensureTimelineElements();
  if (timelineElements.teamSelect) {
    timelineElements.teamSelect.value = 'ALL';
  }
  if (timelineElements.searchInput) {
    timelineElements.searchInput.value = '';
  }
  setAllTracesVisible();
}

function registerTimelineEvents() {
  ensureTimelineElements();
  if (timelineListenersRegistered) {
    return;
  }

  timelineElements.teamSelect?.addEventListener('change', applyTimelineFilters);
  timelineElements.searchInput?.addEventListener('input', () => {
    window.requestAnimationFrame(applyTimelineFilters);
  });
  timelineElements.resetButton?.addEventListener('click', () => {
    handleTimelineReset();
    applyTimelineFilters();
  });

  timelineListenersRegistered = true;
}

async function renderTimeline(timeline, goalieMap, traceOrder) {
  ensureTimelineElements();
  const chartElement = timelineElements.chart;
  if (!chartElement) {
    return;
  }

  const previousTeam = timelineElements.teamSelect?.value || 'ALL';
  const previousSearch = timelineElements.searchInput?.value || '';

  const data = cloneTimelineData(timeline);
  const layout = {
    ...(timeline?.layout || {}),
    plot_bgcolor: 'rgba(15, 23, 42, 0.65)',
    paper_bgcolor: 'rgba(8, 12, 24, 0.95)',
    font: { family: 'Inter, sans-serif', color: '#e2e8f0' },
    margin: { l: 70, r: 20, t: 60, b: 60 },
    legend: {
      ...(timeline?.layout?.legend || {}),
      orientation: 'h',
      x: 0,
      y: -0.25,
      font: { size: 11, color: '#e2e8f0' },
    },
    hoverlabel: {
      bgcolor: '#0f172a',
      bordercolor: '#38bdf8',
    },
    xaxis: {
      ...(timeline?.layout?.xaxis || {}),
      gridcolor: 'rgba(148, 163, 184, 0.2)',
      linecolor: 'rgba(148, 163, 184, 0.35)',
    },
    yaxis: {
      ...(timeline?.layout?.yaxis || {}),
      gridcolor: 'rgba(148, 163, 184, 0.2)',
      linecolor: 'rgba(148, 163, 184, 0.35)',
      tickformat: ',.0%',
    },
  };

  const config = {
    ...(timeline?.config || {}),
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png',
      filename: 'goalie-savepct-timeline',
    },
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
  };

  await Plotly.react(chartElement, data, layout, config);

  timelineState.traceOrder = [...traceOrder];
  timelineState.goalieMap = { ...goalieMap };
  timelineState.traceIndex = new Map();
  data.forEach((trace, index) => {
    timelineState.traceIndex.set(trace.name, index);
  });

  populateTimelineTeams(traceOrder, goalieMap, previousTeam);
  if (timelineElements.searchInput) {
    timelineElements.searchInput.value = previousSearch;
  }

  registerTimelineEvents();

  Plotly.d3.select(chartElement).on('plotly_doubleclick', () => {
    handleTimelineReset();
    applyTimelineFilters();
  });

  if (!timelineElements.resizeListenerAttached) {
    window.addEventListener('resize', () => {
      if (timelineElements.chart) {
        Plotly.Plots.resize(timelineElements.chart);
      }
    });
    timelineElements.resizeListenerAttached = true;
  }

  applyTimelineFilters();
}

function renderLastUpdated(updatedAt) {
  const element = document.getElementById('last-updated');
  if (!element) return;

  if (!updatedAt) {
    element.textContent = 'Okänt';
    element.removeAttribute('datetime');
    element.removeAttribute('title');
    return;
  }

  const date = new Date(updatedAt);
  if (Number.isNaN(date.valueOf())) {
    element.textContent = 'Okänt';
    element.removeAttribute('datetime');
    element.removeAttribute('title');
    return;
  }

  element.textContent = updatedFormatter.format(date);
  element.setAttribute('datetime', date.toISOString());
  element.setAttribute('title', updatedTitleFormatter.format(date));
}

async function loadAndRender(options = {}) {
  const dataset = await loadGoalieDataset(options);
  const summary = Array.isArray(dataset.summary) ? dataset.summary : [];
  const goalieMap = dataset.goalieToTeam ?? {};
  const timeline = dataset.timeline ?? { data: [], layout: {}, config: {} };
  const traceOrder = Array.isArray(dataset.traceOrder) && dataset.traceOrder.length
    ? dataset.traceOrder
    : (timeline.data || []).map((trace) => trace.name);

  const teams = computeUniqueTeams(summary);
  const zeroShot = computeZeroShotGoalies(summary);
  const range = computeDateRange(timeline.data || []);
  const totalShots = summary.reduce((acc, row) => acc + (row.total_shots || 0), 0);

  populateStats({
    teams: teams.length,
    goalies: summary.length,
    shots: totalShots,
    range,
    zeroShotCount: zeroShot.length,
  });

  renderLastUpdated(dataset.updatedAt ?? null);
  renderZeroShotTags(zeroShot);
  renderTopGoalies(summary, goalieMap);
  renderSeasonInsights(summary);
  await renderTimeline(timeline, goalieMap, traceOrder);
}

async function handleRefresh(event) {
  const button = event.currentTarget;
  if (!(button instanceof HTMLButtonElement) || isRefreshing) {
    return;
  }

  isRefreshing = true;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = 'Uppdaterar…';

  try {
    await loadAndRender({ cacheBust: true });
  } catch (error) {
    console.error('Failed to refresh dataset', error);
  } finally {
    button.disabled = false;
    button.textContent = originalText || 'Uppdatera data';
    isRefreshing = false;
  }
}

async function init() {
  attachNavHighlight('overview');
  const refreshButton = document.getElementById('refresh-data');
  refreshButton?.addEventListener('click', handleRefresh);

  try {
    await loadAndRender();
  } catch (error) {
    console.error('Failed to initialise overview page', error);
    renderLastUpdated(null);
    const chartContainer = document.getElementById('timeline-chart');
    if (chartContainer) {
      chartContainer.textContent = 'Det gick inte att ladda tidslinjen just nu.';
    }
  }
}

document.addEventListener('DOMContentLoaded', init);
