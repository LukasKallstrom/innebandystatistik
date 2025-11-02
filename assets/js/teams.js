import { getSummaryData } from './data.js';
import {
  attachNavHighlight,
  createElement,
  formatNumber,
  formatPercent,
  parseSearchTerms,
  sortData,
} from './utils.js';

const state = {
  searchTerms: [],
  sortKey: 'save_pct',
  ascending: false,
};

let teams = [];

function aggregateTeams(summary) {
  const map = new Map();
  summary.forEach((row) => {
    const key = row.team || 'Okänt lag';
    if (!map.has(key)) {
      map.set(key, {
        team: key,
        goalieAppearances: 0,
        totalShots: 0,
        totalSaves: 0,
        goalsAgainst: 0,
      });
    }
    const entry = map.get(key);
    entry.goalieAppearances += 1;
    entry.totalShots += row.total_shots || 0;
    entry.totalSaves += row.total_saves || 0;
    entry.goalsAgainst += row.goals_against || 0;
  });
  return [...map.values()].map((team) => ({
    ...team,
    save_pct: team.totalShots ? team.totalSaves / team.totalShots : null,
  }));
}

function filterTeams() {
  if (!state.searchTerms.length) {
    return teams;
  }
  return teams.filter((team) =>
    state.searchTerms.some((term) => team.team.toLowerCase().includes(term)),
  );
}

function renderTable(rows) {
  const tbody = document.getElementById('team-table-body');
  const empty = document.getElementById('team-empty');
  if (!tbody || !empty) return;
  tbody.innerHTML = '';

  if (!rows.length) {
    empty.hidden = false;
    return;
  }

  empty.hidden = true;
  const fragment = document.createDocumentFragment();
  rows.forEach((team) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${team.team}</td>
      <td>${formatNumber(team.goalieAppearances)}</td>
      <td>${formatNumber(team.totalShots)}</td>
      <td>${formatNumber(team.totalSaves)}</td>
      <td>${formatNumber(team.goalsAgainst)}</td>
      <td>${formatPercent(team.save_pct, 1)}</td>
    `;
    fragment.appendChild(tr);
  });
  tbody.appendChild(fragment);
}

function updateSortIndicators() {
  document.querySelectorAll('th.sortable').forEach((th) => {
    const { key } = th.dataset;
    if (key === state.sortKey) {
      th.setAttribute('data-sort', state.ascending ? 'asc' : 'desc');
    } else {
      th.removeAttribute('data-sort');
    }
  });
}

function renderHighlights(rows) {
  const highlight = document.getElementById('team-highlight');
  if (!highlight) return;
  if (!rows.length) {
    highlight.textContent = 'Inga lag matchar nuvarande filter.';
    return;
  }
  const leader = rows[0];
  highlight.innerHTML = `
    <strong>${leader.team}</strong> håller högst räddningsprocent just nu (${formatPercent(
      leader.save_pct,
      1,
    )}).
  `;
}

function applyStateChange() {
  const rows = sortData(filterTeams(), state.sortKey, state.ascending);
  renderTable(rows);
  renderHighlights(rows);
}

function registerEvents() {
  const searchInput = document.getElementById('team-search');
  const resetButton = document.getElementById('team-reset');

  searchInput?.addEventListener('input', (event) => {
    state.searchTerms = parseSearchTerms(event.target.value);
    applyStateChange();
  });

  resetButton?.addEventListener('click', () => {
    state.searchTerms = [];
    state.sortKey = 'save_pct';
    state.ascending = false;
    if (searchInput) searchInput.value = '';
    updateSortIndicators();
    applyStateChange();
  });

  document.querySelectorAll('th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const { key } = th.dataset;
      if (!key) return;
      if (state.sortKey === key) {
        state.ascending = !state.ascending;
      } else {
        state.sortKey = key;
        state.ascending = key === 'team';
      }
      updateSortIndicators();
      applyStateChange();
    });
  });
}

function renderSummaryCards() {
  const container = document.getElementById('team-summary');
  if (!container) return;
  const totalShots = teams.reduce((acc, team) => acc + team.totalShots, 0);
  const totalSaves = teams.reduce((acc, team) => acc + team.totalSaves, 0);
  const totalGoals = teams.reduce((acc, team) => acc + team.goalsAgainst, 0);
  container.innerHTML = '';

  const cards = [
    {
      label: 'Totala skott',
      value: formatNumber(totalShots),
      meta: 'Ackumulerat över alla målvakter',
    },
    {
      label: 'Totala räddningar',
      value: formatNumber(totalSaves),
      meta: 'Försvarade avslut över säsongen',
    },
    {
      label: 'Insläppta mål',
      value: formatNumber(totalGoals),
      meta: 'Summerat från målvaktsstatistik',
    },
  ];

  cards.forEach((card) => {
    const el = createElement('article', { className: 'stat-card' });
    el.innerHTML = `
      <span class="stat-card__label">${card.label}</span>
      <span class="stat-card__value">${card.value}</span>
      <span class="stat-card__meta">${card.meta}</span>
    `;
    container.appendChild(el);
  });
}

async function init() {
  attachNavHighlight('teams');
  try {
    const summary = await getSummaryData();
    teams = aggregateTeams(summary);
    renderSummaryCards();
    registerEvents();
    updateSortIndicators();
    applyStateChange();
  } catch (error) {
    console.error('Failed to initialise team overview', error);
    const tableWrapper = document.getElementById('team-table');
    if (tableWrapper) {
      tableWrapper.textContent = 'Det gick inte att ladda lagöversikten.';
    }
  }
}

document.addEventListener('DOMContentLoaded', init);
